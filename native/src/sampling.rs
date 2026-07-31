//! Algorithm 3 hierarchical R2 layout and LOD overlap.

use std::collections::BTreeMap;

use glam::{Vec2, Vec3};

use crate::{
    constants::{LOD_OVERLAP_START, MAX_LOD},
    math::{decode_equal_area, r2, stable_rotation},
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodBlend {
    pub fine: u32,
    pub coarse: u32,
    pub coarse_weight: f32,
}

/// Chebyshev-distance LOD with a 10% overlap before each nominal boundary.
#[must_use]
pub fn lod_blend(position: Vec3, camera: Vec3, base_spacing: f32) -> LodBlend {
    let distance = (position - camera).abs().max_element();
    let normalized = (distance / (base_spacing * 16.0)).max(1.0);
    let nominal = normalized.log2().floor().clamp(0.0, MAX_LOD as f32) as u32;
    let next = (nominal + 1).min(MAX_LOD);
    if next == nominal {
        return LodBlend {
            fine: nominal,
            coarse: next,
            coarse_weight: 0.0,
        };
    }
    let boundary = base_spacing * 16.0 * 2.0_f32.powi((nominal + 1) as i32);
    let overlap_start = boundary * LOD_OVERLAP_START;
    let weight = ((distance - overlap_start) / (boundary - overlap_start)).clamp(0.0, 1.0);
    LodBlend {
        fine: nominal,
        coarse: next,
        coarse_weight: weight,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RayOwner {
    pub probe_key: u32,
    pub canonical_slot: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RankedRay {
    pub probe_key: u32,
    pub canonical_slot: u32,
    pub local_rank: u32,
    pub sequence_index: u64,
}

/// Deterministic executable form of Algorithm 3.
///
/// Probe allocation order is deliberately irrelevant. Inputs are sorted by
/// `(probe_key, canonical_slot)`, counts propagate to the caller-provided
/// hierarchical offset, and every surface sample receives a stable local rank.
#[must_use]
pub fn assign_hierarchical_ranks(owners: &[RayOwner], parent_offsets: &BTreeMap<u32, u64>) -> Vec<RankedRay> {
    let mut canonical = owners.to_vec();
    canonical.sort_unstable_by_key(|owner| (owner.probe_key, owner.canonical_slot));

    let mut local_counts = BTreeMap::<u32, u32>::new();
    let mut ranked = Vec::with_capacity(canonical.len());
    for owner in canonical {
        let local_rank = *local_counts.get(&owner.probe_key).unwrap_or(&0);
        local_counts.insert(owner.probe_key, local_rank + 1);
        let sequence_index =
            parent_offsets.get(&owner.probe_key).copied().unwrap_or(0) + u64::from(local_rank);
        ranked.push(RankedRay {
            probe_key: owner.probe_key,
            canonical_slot: owner.canonical_slot,
            local_rank,
            sequence_index,
        });
    }
    ranked.sort_unstable_by_key(|ray| ray.canonical_slot);
    ranked
}

#[must_use]
pub fn ray_direction(sequence_index: u64, frame: u32, normal: Vec3) -> Vec3 {
    let mut direction = decode_equal_area(r2(sequence_index, stable_rotation(frame)));
    if direction.dot(normal) < 0.0 {
        direction = -direction;
    }
    direction
}

#[must_use]
pub fn temporal_jitter(frame: u32) -> Vec2 {
    stable_rotation(frame)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_order_cannot_change_ranks() {
        let owners = [
            RayOwner {
                probe_key: 9,
                canonical_slot: 20,
            },
            RayOwner {
                probe_key: 4,
                canonical_slot: 1,
            },
            RayOwner {
                probe_key: 9,
                canonical_slot: 2,
            },
            RayOwner {
                probe_key: 4,
                canonical_slot: 30,
            },
        ];
        let mut shuffled = owners;
        shuffled.reverse();
        let offsets = BTreeMap::from([(4, 12), (9, 28)]);
        assert_eq!(
            assign_hierarchical_ranks(&owners, &offsets),
            assign_hierarchical_ranks(&shuffled, &offsets)
        );
    }

    #[test]
    fn lod_overlap_is_continuous() {
        let camera = Vec3::ZERO;
        let base = 1.0;
        let boundary = base * 16.0 * 2.0;
        let before = lod_blend(Vec3::splat(boundary * LOD_OVERLAP_START - 1.0e-4), camera, base);
        let start = lod_blend(Vec3::splat(boundary * LOD_OVERLAP_START), camera, base);
        let end = lod_blend(Vec3::splat(boundary - 1.0e-4), camera, base);
        assert_eq!(before.coarse_weight, 0.0);
        assert_eq!(start.coarse_weight, 0.0);
        assert!(end.coarse_weight > 0.999);
    }
}

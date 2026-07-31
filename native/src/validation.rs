//! Deterministic validation gates for math, BVH, scenes, and temporal stability.

use std::{collections::BTreeMap, path::Path};

use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::{
    bvh::Ray,
    constants::{CASCADE_COUNT, CASCADE_DIRECTIONS, TEMPORAL_ROTATION_PERIOD},
    math::{decode_equal_area, direction_center, direction_index, r2},
    sampling::{assign_hierarchical_ranks, RayOwner},
    scene::{Scene, SceneId},
};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ValidationReport {
    pub scene_count: usize,
    pub tested_triangles: usize,
    pub tested_rays: usize,
    pub bvh_misses: usize,
    pub non_finite_values: usize,
    pub temporal_mismatches: usize,
    pub cascade_direction_failures: usize,
    pub skipped_sponza: bool,
}

impl ValidationReport {
    #[must_use]
    pub fn passed(&self) -> bool {
        self.bvh_misses == 0
            && self.non_finite_values == 0
            && self.temporal_mismatches == 0
            && self.cascade_direction_failures == 0
            && self.scene_count >= 11
    }
}

pub fn audit(asset_root: impl AsRef<Path>, include_sponza: bool) -> ValidationReport {
    let mut report = ValidationReport::default();
    for cascade in 0..CASCADE_COUNT as u32 {
        for index in 0..CASCADE_DIRECTIONS[cascade as usize] {
            if direction_index(direction_center(index, cascade), cascade) != index {
                report.cascade_direction_failures += 1;
            }
        }
    }
    let owners: Vec<_> = (0..256)
        .map(|slot| RayOwner {
            probe_key: (slot * 17) % 31,
            canonical_slot: slot,
        })
        .collect();
    let reverse: Vec<_> = owners.iter().copied().rev().collect();
    let offsets = BTreeMap::new();
    if assign_hierarchical_ranks(&owners, &offsets) != assign_hierarchical_ranks(&reverse, &offsets) {
        report.temporal_mismatches += 1;
    }
    for frame in 0..TEMPORAL_ROTATION_PERIOD {
        for index in 0..512 {
            let direction = decode_equal_area(r2(index, crate::math::stable_rotation(frame)));
            if !direction.is_finite() || (direction.length() - 1.0).abs() > 1.0e-4 {
                report.non_finite_values += 1;
            }
        }
    }

    for id in SceneId::ALL {
        if id == SceneId::Sponza && !include_sponza {
            report.skipped_sponza = true;
            continue;
        }
        let Ok(scene) = Scene::load(id, asset_root.as_ref()) else {
            if id == SceneId::Sponza {
                report.skipped_sponza = true;
                continue;
            }
            report.non_finite_values += 1;
            continue;
        };
        report.scene_count += 1;
        report.tested_triangles += scene.source_triangle_count;
        let center = (scene.bvh.bounds.min + scene.bvh.bounds.max) * 0.5;
        let extent = scene.bvh.bounds.max - scene.bvh.bounds.min;
        let radius = extent.length().max(1.0);
        for sample in 0..64 {
            let direction = decode_equal_area(r2(sample, Vec2::ZERO));
            let origin = center + direction * radius;
            let ray = Ray::new(origin, center - origin);
            report.tested_rays += 1;
            let accelerated = scene.bvh.trace(ray);
            let reference = scene.bvh.trace_brute_force(ray);
            let matches = match (accelerated, reference) {
                (None, None) => true,
                (Some(left), Some(right)) => {
                    let tolerance = 2.0e-4 * right.distance.abs().max(1.0);
                    (left.distance - right.distance).abs() <= tolerance
                }
                _ => false,
            };
            if !matches {
                report.bvh_misses += 1;
            }
        }
        let light0 = scene.settings.animated_lights(0.0);
        let light1 = scene.settings.animated_lights(1.0);
        for value in [
            light0.sun_direction,
            light0.sun_radiance,
            light0.point_position,
            light0.point_radiance,
            light1.sun_direction,
            light1.sun_radiance,
            light1.point_position,
            light1.point_radiance,
            Vec3::splat(scene.settings.base_spacing),
        ] {
            if !value.is_finite() {
                report.non_finite_values += 1;
            }
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn procedural_audit_passes() {
        let report = audit(".", false);
        assert!(report.passed(), "{report:#?}");
        assert_eq!(report.scene_count, 11);
        assert!(report.tested_triangles > 40_000);
    }
}

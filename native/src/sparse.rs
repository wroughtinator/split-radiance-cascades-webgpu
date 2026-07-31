//! Deterministic CPU sparse probe tables used by validation and scene audits.

use std::collections::BTreeMap;

use glam::{IVec3, Vec3};
use thiserror::Error;

use crate::{
    constants::{HASH_PROBE_LIMIT, MAX_LOD},
    math::{hash32, pack_probe_key, probe_cell, probe_center},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProbeAddress {
    pub key: u64,
    pub compact_index: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Probe {
    pub key: u64,
    pub position: Vec3,
    pub lod: u32,
    pub ray_count: u32,
    pub hierarchical_offset: u64,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum SparseError {
    #[error("sparse probe compact capacity {capacity} exceeded")]
    CompactOverflow { capacity: usize },
    #[error("sparse probe hash failed after {limit} probes")]
    HashOverflow { limit: u32 },
}

#[derive(Debug)]
pub struct SparseProbeTable {
    slots: Vec<Option<ProbeAddress>>,
    probes: Vec<Probe>,
    compact_capacity: usize,
}

impl SparseProbeTable {
    #[must_use]
    pub fn new(hash_capacity: usize, compact_capacity: usize) -> Self {
        assert!(hash_capacity.is_power_of_two());
        Self {
            slots: vec![None; hash_capacity],
            probes: Vec::with_capacity(compact_capacity),
            compact_capacity,
        }
    }

    pub fn clear(&mut self) {
        self.slots.fill(None);
        self.probes.clear();
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.probes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.probes.is_empty()
    }

    #[must_use]
    pub fn probes(&self) -> &[Probe] {
        &self.probes
    }

    pub fn insert(&mut self, cell: IVec3, lod: u32, spacing: f32) -> Result<ProbeAddress, SparseError> {
        let key = pack_probe_key(cell, lod.min(MAX_LOD));
        let mask = self.slots.len() - 1;
        let start = hash32(key as u32 ^ hash32((key >> 32) as u32)) as usize & mask;
        for offset in 0..HASH_PROBE_LIMIT {
            let slot = (start + offset as usize) & mask;
            match self.slots[slot] {
                Some(address) if address.key == key => return Ok(address),
                Some(_) => {}
                None => {
                    if self.probes.len() >= self.compact_capacity {
                        return Err(SparseError::CompactOverflow {
                            capacity: self.compact_capacity,
                        });
                    }
                    let address = ProbeAddress {
                        key,
                        compact_index: self.probes.len() as u32,
                    };
                    self.probes.push(Probe {
                        key,
                        position: probe_center(cell, spacing),
                        lod,
                        ray_count: 0,
                        hierarchical_offset: 0,
                    });
                    self.slots[slot] = Some(address);
                    return Ok(address);
                }
            }
        }
        Err(SparseError::HashOverflow {
            limit: HASH_PROBE_LIMIT,
        })
    }

    #[must_use]
    pub fn lookup_key(&self, key: u64) -> Option<ProbeAddress> {
        let mask = self.slots.len() - 1;
        let start = hash32(key as u32 ^ hash32((key >> 32) as u32)) as usize & mask;
        for offset in 0..HASH_PROBE_LIMIT {
            match self.slots[(start + offset as usize) & mask] {
                Some(address) if address.key == key => return Some(address),
                Some(_) => {}
                None => return None,
            }
        }
        None
    }

    #[must_use]
    pub fn lookup(&self, position: Vec3, spacing: f32, lod: u32) -> Option<ProbeAddress> {
        self.lookup_key(pack_probe_key(probe_cell(position, spacing), lod))
    }

    pub fn add_ray(&mut self, address: ProbeAddress) {
        self.probes[address.compact_index as usize].ray_count += 1;
    }

    /// Canonical prefix assignment. Hash allocation order cannot influence it.
    pub fn assign_offsets(&mut self, base_offset: u64) -> u64 {
        let mut order: Vec<usize> = (0..self.probes.len()).collect();
        order.sort_unstable_by_key(|&index| self.probes[index].key);
        let mut offset = base_offset;
        for index in order {
            self.probes[index].hierarchical_offset = offset;
            offset += u64::from(self.probes[index].ray_count);
        }
        offset
    }

    #[must_use]
    pub fn exact_history_map(&self) -> BTreeMap<u64, usize> {
        self.probes
            .iter()
            .enumerate()
            .map(|(index, probe)| (probe.key, index))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum PublicationSlot {
        Empty,
        Claimed,
        Published {
            token: u32,
            key: u64,
            compact_index: Option<u32>,
        },
    }

    fn claim(slots: &mut [PublicationSlot], token: u32, key: u64) -> Option<usize> {
        let start = token as usize & (slots.len() - 1);
        for offset in 0..HASH_PROBE_LIMIT as usize {
            let slot = (start + offset) & (slots.len() - 1);
            match slots[slot] {
                PublicationSlot::Empty => {
                    slots[slot] = PublicationSlot::Claimed;
                    return Some(slot);
                }
                PublicationSlot::Published {
                    token: stored_token,
                    key: stored_key,
                    ..
                } if stored_token == token && stored_key == key => return None,
                PublicationSlot::Claimed => return None,
                PublicationSlot::Published { .. } => {}
            }
        }
        panic!("test publication table overflow");
    }

    fn publish(slots: &mut [PublicationSlot], slot: usize, token: u32, key: u64) {
        assert_eq!(slots[slot], PublicationSlot::Claimed);
        // This assignment models the shader's release publication: an
        // observer sees either Claimed or a complete token/full-key tuple.
        slots[slot] = PublicationSlot::Published {
            token,
            key,
            compact_index: None,
        };
    }

    fn canonicalize(slots: &mut [PublicationSlot]) {
        let mut next_index = 0;
        for slot in 0..slots.len() {
            let PublicationSlot::Published { token, key, .. } = slots[slot] else {
                continue;
            };
            let start = token as usize & (slots.len() - 1);
            let duplicate = (0..HASH_PROBE_LIMIT as usize)
                .map(|offset| (start + offset) & (slots.len() - 1))
                .take_while(|&earlier| earlier != slot)
                .any(|earlier| {
                    matches!(
                        slots[earlier],
                        PublicationSlot::Published {
                            token: earlier_token,
                            key: earlier_key,
                            compact_index: Some(_),
                        } if earlier_token == token && earlier_key == key
                    )
                });
            let compact_index = if duplicate {
                None
            } else {
                let result = next_index;
                next_index += 1;
                Some(result)
            };
            slots[slot] = PublicationSlot::Published {
                token,
                key,
                compact_index,
            };
        }
    }

    fn published_lookup(slots: &[PublicationSlot], token: u32, key: u64) -> Option<u32> {
        let start = token as usize & (slots.len() - 1);
        for offset in 0..HASH_PROBE_LIMIT as usize {
            match slots[(start + offset) & (slots.len() - 1)] {
                PublicationSlot::Empty => return None,
                PublicationSlot::Published {
                    token: stored_token,
                    key: stored_key,
                    compact_index: Some(index),
                } if stored_token == token && stored_key == key => return Some(index),
                PublicationSlot::Claimed | PublicationSlot::Published { .. } => {}
            }
        }
        None
    }

    #[test]
    fn duplicate_insertion_is_idempotent() {
        let mut table = SparseProbeTable::new(64, 16);
        let cell = IVec3::new(-3, 2, 7);
        let a = table.insert(cell, 0, 0.5).unwrap();
        let b = table.insert(cell, 0, 0.5).unwrap();
        assert_eq!(a, b);
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn canonical_offsets_ignore_insertion_order() {
        fn offsets(cells: impl IntoIterator<Item = IVec3>) -> BTreeMap<u64, u64> {
            let mut table = SparseProbeTable::new(64, 16);
            for cell in cells {
                let address = table.insert(cell, 0, 1.0).unwrap();
                table.add_ray(address);
                table.add_ray(address);
            }
            table.assign_offsets(7);
            table
                .probes()
                .iter()
                .map(|probe| (probe.key, probe.hierarchical_offset))
                .collect()
        }
        let cells = [IVec3::new(2, 0, 0), IVec3::new(-1, 1, 0), IVec3::new(7, 2, -3)];
        assert_eq!(offsets(cells), offsets(cells.into_iter().rev()));
    }

    #[test]
    fn publication_state_preserves_exact_keys_under_claim_races_and_token_collisions() {
        let mut slots = vec![PublicationSlot::Empty; 64];
        let token = 11;
        let keys: Vec<u64> = (0..24).map(|index| 0x0123_4567_0000_0000 | index).collect();

        // Every key deliberately shares one token, while 1,024 invocations
        // request one identical key. Each pass stages claims before publishing
        // them, modeling adversarial SIMD branch ordering.
        let requests: Vec<u64> = keys
            .iter()
            .copied()
            .chain(std::iter::repeat_n(keys[7], 1_024))
            .collect();
        for _ in 0..HASH_PROBE_LIMIT {
            let pending: Vec<(usize, u64)> = requests
                .iter()
                .filter_map(|&key| claim(&mut slots, token, key).map(|slot| (slot, key)))
                .collect();
            for (slot, key) in pending {
                publish(&mut slots, slot, token, key);
            }
        }
        canonicalize(&mut slots);

        let indices: Vec<u32> = keys
            .iter()
            .map(|&key| published_lookup(&slots, token, key).expect("exact key was dropped"))
            .collect();
        assert_eq!(indices.len(), 24);
        let mut unique = indices.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), keys.len());
        let published: Vec<_> = slots
            .iter()
            .filter_map(|slot| match slot {
                PublicationSlot::Published { compact_index, .. } => Some(compact_index),
                PublicationSlot::Empty | PublicationSlot::Claimed => None,
            })
            .collect();
        assert_eq!(published.len(), keys.len());
        assert!(published.into_iter().all(Option::is_some));
    }
}

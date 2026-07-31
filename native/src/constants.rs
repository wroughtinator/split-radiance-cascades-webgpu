//! Constants shared by CPU validation code and GPU shader generation.

/// Paper configuration: `K = 4`, `l = 4`, `Theta_0 = 4`.
pub const CASCADE_COUNT: usize = 4;
pub const BRANCHING_FACTOR: u32 = 4;
pub const SPATIAL_SCALE: u32 = 4;
pub const THETA_0: u32 = 4;
pub const BASE_DIRECTION_COUNT: u32 = 2 * THETA_0 * THETA_0;
pub const CASCADE_DIRECTIONS: [u32; CASCADE_COUNT] = [32, 128, 512, 2048];
pub const PROBE_CAPACITY: [u32; CASCADE_COUNT] = [4096, 1536, 512, 256];
pub const HASH_CAPACITY: [u32; CASCADE_COUNT] = [8192, 4096, 2048, 1024];
pub const PROBE_OFFSET: [u32; CASCADE_COUNT] = [
    0,
    PROBE_CAPACITY[0],
    PROBE_CAPACITY[0] + PROBE_CAPACITY[1],
    PROBE_CAPACITY[0] + PROBE_CAPACITY[1] + PROBE_CAPACITY[2],
];
pub const HASH_OFFSET: [u32; CASCADE_COUNT] = [
    0,
    HASH_CAPACITY[0],
    HASH_CAPACITY[0] + HASH_CAPACITY[1],
    HASH_CAPACITY[0] + HASH_CAPACITY[1] + HASH_CAPACITY[2],
];
pub const DATA_OFFSET: [u32; CASCADE_COUNT] = [
    0,
    PROBE_CAPACITY[0] * CASCADE_DIRECTIONS[0],
    PROBE_CAPACITY[0] * CASCADE_DIRECTIONS[0] + PROBE_CAPACITY[1] * CASCADE_DIRECTIONS[1],
    PROBE_CAPACITY[0] * CASCADE_DIRECTIONS[0]
        + PROBE_CAPACITY[1] * CASCADE_DIRECTIONS[1]
        + PROBE_CAPACITY[2] * CASCADE_DIRECTIONS[2],
];
pub const TOTAL_PROBES: u32 = PROBE_CAPACITY[0] + PROBE_CAPACITY[1] + PROBE_CAPACITY[2] + PROBE_CAPACITY[3];
pub const TOTAL_HASH_SLOTS: u32 = HASH_CAPACITY[0] + HASH_CAPACITY[1] + HASH_CAPACITY[2] + HASH_CAPACITY[3];
pub const TOTAL_DIRECTIONAL_VALUES: u32 = PROBE_CAPACITY[0] * CASCADE_DIRECTIONS[0]
    + PROBE_CAPACITY[1] * CASCADE_DIRECTIONS[1]
    + PROBE_CAPACITY[2] * CASCADE_DIRECTIONS[2]
    + PROBE_CAPACITY[3] * CASCADE_DIRECTIONS[3];

pub const MAX_LOD: u32 = 7;
pub const LOD_OVERLAP_START: f32 = 0.9;
pub const INTERVAL_SCALE: f32 = 1.6;
pub const TEMPORAL_WEIGHT_STATIC: f32 = 0.96;
pub const TEMPORAL_WEIGHT_ANIMATED: f32 = 0.84;
pub const TEMPORAL_ROTATION_PERIOD: u32 = 32;
pub const FIXED_POINT_SCALE: f32 = 4096.0;
pub const MAX_RADIANCE: f32 = 16.0;
pub const IRRADIANCE_EXTENT: u32 = 8;
pub const IRRADIANCE_INTERIOR: u32 = 6;
pub const IRRADIANCE_TEXELS: u32 = IRRADIANCE_EXTENT * IRRADIANCE_EXTENT;
pub const EMPTY_KEY: u32 = u32::MAX;
pub const HASH_PROBE_LIMIT: u32 = 32;
pub const BVH_STACK_SIZE: usize = 64;
pub const BVH_LEAF_TRIANGLES: usize = 4;
pub const BVH_SAH_BINS: usize = 16;

#[must_use]
pub const fn direction_count(cascade: u32) -> u32 {
    BASE_DIRECTION_COUNT << (2 * cascade)
}

#[must_use]
pub fn spacing(base_spacing: f32, cascade: u32, lod: u32) -> f32 {
    base_spacing * 2.0_f32.powi((cascade + lod) as i32)
}

#[must_use]
pub fn interval_cutoff(base_spacing: f32, cascade: u32, lod: u32) -> f32 {
    INTERVAL_SCALE * base_spacing * 2.0_f32.powi(lod as i32) * 4.0_f32.powi(cascade as i32)
}

#[must_use]
pub fn interval_bounds(base_spacing: f32, cascade: u32, lod: u32) -> (f32, f32) {
    let far = interval_cutoff(base_spacing, cascade, lod);
    let near = if cascade == 0 {
        0.0
    } else {
        interval_cutoff(base_spacing, cascade - 1, lod)
    };
    (near, far)
}

const _: () = assert!(BASE_DIRECTION_COUNT == 32);
const _: () = assert!(direction_count(3) == 2048);
const _: () = assert!(PROBE_OFFSET[3] + PROBE_CAPACITY[3] == TOTAL_PROBES);
const _: () = assert!(HASH_OFFSET[3] + HASH_CAPACITY[3] == TOTAL_HASH_SLOTS);

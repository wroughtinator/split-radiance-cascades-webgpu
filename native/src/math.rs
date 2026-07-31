//! Small deterministic math helpers mirrored in `shaders/split_rc.glsl`.

use std::f32::consts::{PI, TAU};

use glam::{IVec3, Vec2, Vec3};

use crate::constants::MAX_LOD;

const PLASTIC: f64 = 1.324_717_957_244_746;

#[must_use]
pub fn r2(index: u64, jitter: Vec2) -> Vec2 {
    let i = index as f64;
    Vec2::new(
        (0.5 + i / PLASTIC + f64::from(jitter.x)).fract() as f32,
        (0.5 + i / (PLASTIC * PLASTIC) + f64::from(jitter.y)).fract() as f32,
    )
}

/// Equal-area cylindrical map used by Algorithm 2.
#[must_use]
pub fn decode_equal_area(uv: Vec2) -> Vec3 {
    let phi = uv.x * TAU;
    let z = uv.y.mul_add(2.0, -1.0);
    let radius = (1.0 - z * z).max(0.0).sqrt();
    Vec3::new(radius * phi.cos(), radius * phi.sin(), z)
}

#[must_use]
pub fn encode_equal_area(direction: Vec3) -> Vec2 {
    let direction = direction.normalize_or_zero();
    Vec2::new(
        (direction.y.atan2(direction.x) / TAU).rem_euclid(1.0),
        direction.z.mul_add(0.5, 0.5),
    )
}

#[must_use]
pub fn oct_encode(direction: Vec3) -> Vec2 {
    let n = direction.normalize_or_zero();
    let denominator = n.x.abs() + n.y.abs() + n.z.abs();
    if denominator <= f32::EPSILON {
        return Vec2::splat(0.5);
    }
    let mut p = n.truncate() / denominator;
    if n.z < 0.0 {
        p = Vec2::new((1.0 - p.y.abs()).copysign(p.x), (1.0 - p.x.abs()).copysign(p.y));
    }
    p.mul_add(Vec2::splat(0.5), Vec2::splat(0.5))
}

#[must_use]
pub fn oct_decode(uv: Vec2) -> Vec3 {
    let p = uv.mul_add(Vec2::splat(2.0), Vec2::splat(-1.0));
    let mut n = Vec3::new(p.x, p.y, 1.0 - p.x.abs() - p.y.abs());
    if n.z < 0.0 {
        let x = n.x;
        n.x = (1.0 - n.y.abs()).copysign(x);
        n.y = (1.0 - x.abs()).copysign(n.y);
    }
    n.normalize_or_zero()
}

#[must_use]
pub fn probe_cell(position: Vec3, spacing: f32) -> IVec3 {
    (position / spacing).floor().as_ivec3()
}

#[must_use]
pub fn probe_center(cell: IVec3, spacing: f32) -> Vec3 {
    (cell.as_vec3() + Vec3::splat(0.5)) * spacing
}

/// Exact 61-bit logical key: 19 signed bits per axis, 3 LOD bits, and one
/// primary/secondary tag bit. GPU backends store the same value as two u32s.
#[must_use]
pub fn try_pack_probe_key(cell: IVec3, lod: u32) -> Option<u64> {
    try_pack_tagged_probe_key(cell, lod, false)
}

#[must_use]
pub fn try_pack_tagged_probe_key(cell: IVec3, lod: u32, secondary: bool) -> Option<u64> {
    const MINIMUM: i32 = -262_144;
    const MAXIMUM: i32 = 262_143;
    ((MINIMUM..=MAXIMUM).contains(&cell.x)
        && (MINIMUM..=MAXIMUM).contains(&cell.y)
        && (MINIMUM..=MAXIMUM).contains(&cell.z)
        && lod <= MAX_LOD)
        .then(|| {
            let encode = |value: i32| (value + 262_144) as u64;
            encode(cell.x)
                | (encode(cell.y) << 19)
                | (encode(cell.z) << 38)
                | (u64::from(lod & 7) << 57)
                | (u64::from(secondary) << 60)
        })
}

/// Packs an already validated probe key.
#[must_use]
pub fn pack_probe_key(cell: IVec3, lod: u32) -> u64 {
    try_pack_probe_key(cell, lod).expect("probe key is outside the signed 19-bit/3-bit LOD domain")
}

#[must_use]
pub fn unpack_probe_key(key: u64) -> (IVec3, u32) {
    let decode = |shift: u32| (((key >> shift) & 0x7_ffff) as i32) - 262_144;
    (
        IVec3::new(decode(0), decode(19), decode(38)),
        ((key >> 57) & 7) as u32,
    )
}

#[must_use]
pub fn hash32(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

#[must_use]
pub fn stable_rotation(frame: u32) -> Vec2 {
    let x = hash32(frame ^ 0x9e37_79b9);
    let y = hash32(frame ^ 0x243f_6a88);
    Vec2::new((x & 0xffff) as f32 / 65_536.0, (y & 0xffff) as f32 / 65_536.0)
}

#[must_use]
pub fn direction_index(direction: Vec3, cascade: u32) -> u32 {
    let theta = 4 << cascade;
    let uv = encode_equal_area(direction);
    let x = (uv.x * (theta * 2) as f32).floor() as u32 % (theta * 2);
    let y = (uv.y * theta as f32).floor().clamp(0.0, (theta - 1) as f32) as u32;
    morton_direction_index(x, y, cascade)
}

#[must_use]
pub fn direction_center(index: u32, cascade: u32) -> Vec3 {
    let theta = 4 << cascade;
    let width = theta * 2;
    let (x, y) = morton_direction_coordinates(index, cascade);
    decode_equal_area(Vec2::new(
        (x as f32 + 0.5) / width as f32,
        (y as f32 + 0.5) / theta as f32,
    ))
}

/// Solid angle represented by one sample in an equal-area sphere partition.
///
/// The C(-1) gather integrates incident radiance over the sphere before its
/// downstream Lambertian `/ PI`, so omitting PI here would under-light by PI.
#[must_use]
pub fn equal_area_solid_angle(sample_count: u32) -> f32 {
    assert!(sample_count > 0);
    4.0 * PI / sample_count as f32
}

#[must_use]
pub fn morton_direction_index(u: u32, v: u32, cascade: u32) -> u32 {
    let bits = 2 + cascade;
    let mut result = 0;
    for bit in 0..bits {
        result |= ((u >> bit) & 1) << (bit * 2);
        result |= ((v >> bit) & 1) << (bit * 2 + 1);
    }
    result | (((u >> bits) & 1) << (bits * 2))
}

#[must_use]
pub fn morton_direction_coordinates(index: u32, cascade: u32) -> (u32, u32) {
    let bits = 2 + cascade;
    let mut u = 0;
    let mut v = 0;
    for bit in 0..bits {
        u |= ((index >> (bit * 2)) & 1) << bit;
        v |= ((index >> (bit * 2 + 1)) & 1) << bit;
    }
    u |= ((index >> (bits * 2)) & 1) << bits;
    (u, v)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use glam::{IVec3, Vec2, Vec3};

    use super::*;

    #[test]
    fn equal_area_round_trip() {
        for i in 0..4096 {
            let direction = decode_equal_area(r2(i, Vec2::ZERO));
            let decoded = decode_equal_area(encode_equal_area(direction));
            assert_abs_diff_eq!(direction.x, decoded.x, epsilon = 2.0e-5);
            assert_abs_diff_eq!(direction.y, decoded.y, epsilon = 2.0e-5);
            assert_abs_diff_eq!(direction.z, decoded.z, epsilon = 2.0e-5);
        }
    }

    #[test]
    fn octahedral_round_trip() {
        for i in 0..4096 {
            let direction = decode_equal_area(r2(i, Vec2::ZERO));
            assert!(direction.dot(oct_decode(oct_encode(direction))) > 0.999_99);
        }
    }

    #[test]
    fn key_round_trip_at_bounds() {
        for cell in [IVec3::splat(-262_144), IVec3::ZERO, IVec3::splat(262_143)] {
            for lod in 0..=7 {
                assert_eq!(unpack_probe_key(pack_probe_key(cell, lod)), (cell, lod));
            }
        }
        assert!(try_pack_probe_key(IVec3::new(-262_145, 0, 0), 0).is_none());
        assert!(try_pack_probe_key(IVec3::new(262_144, 0, 0), 0).is_none());
        assert!(try_pack_probe_key(IVec3::ZERO, 8).is_none());
    }

    #[test]
    fn direction_count_and_parent_groups_are_exact() {
        for cascade in 0..4 {
            let count = crate::constants::direction_count(cascade);
            for index in 0..count {
                assert_eq!(direction_index(direction_center(index, cascade), cascade), index);
            }
            if cascade > 0 {
                assert_eq!(count / 4, crate::constants::direction_count(cascade - 1));
                for parent in 0..crate::constants::direction_count(cascade - 1) {
                    let (parent_u, parent_v) = morton_direction_coordinates(parent, cascade - 1);
                    for child in 0..4 {
                        let (child_u, child_v) = morton_direction_coordinates(parent * 4 + child, cascade);
                        assert_eq!(child_u >> 1, parent_u);
                        assert_eq!(child_v >> 1, parent_v);
                    }
                }
            }
        }
    }

    #[test]
    fn temporal_rotation_does_not_freeze_after_frame_31() {
        assert_ne!(stable_rotation(0), stable_rotation(1));
        assert_ne!(stable_rotation(31), stable_rotation(32));
        assert_ne!(stable_rotation(32), stable_rotation(33));
        assert_ne!(stable_rotation(31), stable_rotation(1_031));
        assert_ne!(stable_rotation(0), stable_rotation(65_536));
    }

    #[test]
    fn c_minus_one_equal_area_quadrature_preserves_constant_radiance() {
        let weight = equal_area_solid_angle(32);
        assert_abs_diff_eq!(weight, 4.0 * std::f32::consts::PI / 32.0, epsilon = 1.0e-7);

        // A unit-radiance environment must integrate to unit Lambertian
        // irradiance after the rendering equation's 1/PI factor.
        let cosine_integral = (0..32)
            .map(|sample| Vec3::Z.dot(direction_center(sample, 0)).max(0.0))
            .sum::<f32>()
            * weight;
        assert_abs_diff_eq!(cosine_integral / std::f32::consts::PI, 1.0, epsilon = 0.04);
    }
}

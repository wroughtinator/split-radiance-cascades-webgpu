//! Loader for the version-3 gzip-compressed `.rcb` Sponza package.

use std::{
    fs::File,
    io::{Read, Take},
    path::Path,
};

use flate2::read::GzDecoder;
use glam::{Vec2, Vec3};
use thiserror::Error;

use crate::{
    bvh::{Aabb, Bvh, GpuBvhNode, Triangle},
    constants::BVH_STACK_SIZE,
    math::oct_decode,
};

const MAGIC: u32 = 0x3142_4352;
const VERSION: u32 = 3;
const HEADER_BYTES: usize = 64;
const VERTEX_FLOATS: usize = 16;
const NODE_FLOATS: usize = 8;
const TRIANGLE_FLOATS: usize = 32;
const LEAF_BIT: u32 = 0x8000_0000;
const MAX_UNCOMPRESSED_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum RcbError {
    #[error("failed to read RCB asset: {0}")]
    Io(#[from] std::io::Error),
    #[error("RCB is truncated: expected at least {expected} bytes, got {actual}")]
    Truncated { expected: usize, actual: usize },
    #[error("RCB magic/version mismatch: magic={magic:#010x}, version={version}")]
    Unsupported { magic: u32, version: u32 },
    #[error("RCB count or byte length is invalid")]
    InvalidLength,
    #[error("RCB BVH node {node} contains an invalid child or leaf range")]
    InvalidBvh { node: usize },
    #[error("RCB BVH depth {depth} exceeds the GPU traversal stack")]
    BvhTooDeep { depth: usize },
}

#[derive(Clone, Debug)]
pub struct RcbScene {
    pub bvh: Bvh,
    pub vertex_count: u32,
    pub bounds: Aabb,
}

impl RcbScene {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, RcbError> {
        let file = File::open(path)?;
        let mut decoder: Take<GzDecoder<File>> = GzDecoder::new(file).take(MAX_UNCOMPRESSED_BYTES);
        let mut bytes = Vec::new();
        decoder.read_to_end(&mut bytes)?;
        Self::from_uncompressed(&bytes)
    }

    pub fn from_uncompressed(bytes: &[u8]) -> Result<Self, RcbError> {
        if bytes.len() < HEADER_BYTES {
            return Err(RcbError::Truncated {
                expected: HEADER_BYTES,
                actual: bytes.len(),
            });
        }
        let word = |offset| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        let float = |offset| f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        let magic = word(0);
        let version = word(4);
        if magic != MAGIC || version != VERSION {
            return Err(RcbError::Unsupported { magic, version });
        }
        let vertex_floats = word(8) as usize;
        let node_floats = word(12) as usize;
        let triangle_floats = word(16) as usize;
        let vertex_count = word(20);
        let node_count = word(24) as usize;
        let triangle_count = word(28) as usize;
        let expected = HEADER_BYTES
            .checked_add(
                vertex_floats
                    .checked_add(node_floats)
                    .and_then(|value| value.checked_add(triangle_floats))
                    .and_then(|value| value.checked_mul(4))
                    .ok_or(RcbError::InvalidLength)?,
            )
            .ok_or(RcbError::InvalidLength)?;
        if vertex_floats != vertex_count as usize * VERTEX_FLOATS
            || node_floats != node_count * NODE_FLOATS
            || triangle_floats != triangle_count * TRIANGLE_FLOATS
            || bytes.len() != expected
        {
            return Err(RcbError::InvalidLength);
        }
        let bounds = Aabb {
            min: Vec3::new(float(32), float(36), float(40)),
            max: Vec3::new(float(44), float(48), float(52)),
        };
        let mut offset = HEADER_BYTES + vertex_floats * 4;
        let mut nodes = Vec::with_capacity(node_count);
        for _ in 0..node_count {
            let min = [
                f32_at(bytes, offset),
                f32_at(bytes, offset + 4),
                f32_at(bytes, offset + 8),
            ];
            let web_left = u32_at(bytes, offset + 12);
            let max = [
                f32_at(bytes, offset + 16),
                f32_at(bytes, offset + 20),
                f32_at(bytes, offset + 24),
            ];
            let web_right = u32_at(bytes, offset + 28);
            let leaf = web_left & LEAF_BIT != 0;
            nodes.push(GpuBvhNode {
                bounds_min: min,
                left_or_first: web_left & !LEAF_BIT,
                bounds_max: max,
                right_or_count: if leaf { web_right | LEAF_BIT } else { web_right },
            });
            offset += NODE_FLOATS * 4;
        }
        let mut triangles = Vec::with_capacity(triangle_count);
        for _ in 0..triangle_count {
            let vector = |word_offset: usize| {
                Vec3::new(
                    f32_at(bytes, offset + word_offset * 4),
                    f32_at(bytes, offset + (word_offset + 1) * 4),
                    f32_at(bytes, offset + (word_offset + 2) * 4),
                )
            };
            let normals =
                [28, 29, 30].map(|normal_word| decode_packed_normal(u32_at(bytes, offset + normal_word * 4)));
            triangles.push(Triangle {
                positions: [vector(0), vector(4), vector(8)],
                normals,
                uvs: [
                    [f32_at(bytes, offset + 20 * 4), f32_at(bytes, offset + 21 * 4)],
                    [f32_at(bytes, offset + 22 * 4), f32_at(bytes, offset + 23 * 4)],
                    [f32_at(bytes, offset + 24 * 4), f32_at(bytes, offset + 25 * 4)],
                ],
                albedo: vector(12),
                emissive: vector(16),
                material: f32_at(bytes, offset + 26 * 4),
                alpha_cutoff: f32_at(bytes, offset + 27 * 4),
            });
            offset += TRIANGLE_FLOATS * 4;
        }
        let max_depth = validate_bvh(&nodes, triangle_count)?;
        if max_depth >= BVH_STACK_SIZE {
            return Err(RcbError::BvhTooDeep { depth: max_depth });
        }
        Ok(Self {
            bvh: Bvh::from_validated_parts(nodes, triangles, bounds, max_depth),
            vertex_count,
            bounds,
        })
    }
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn f32_at(bytes: &[u8], offset: usize) -> f32 {
    f32::from_bits(u32_at(bytes, offset))
}

fn decode_packed_normal(packed: u32) -> Vec3 {
    let x = (packed & 0xffff) as f32 / 65_535.0;
    let y = (packed >> 16) as f32 / 65_535.0;
    oct_decode(Vec2::new(x, y))
}

fn validate_bvh(nodes: &[GpuBvhNode], triangle_count: usize) -> Result<usize, RcbError> {
    if nodes.is_empty() {
        return Err(RcbError::InvalidBvh { node: 0 });
    }
    let mut stack = vec![(0_usize, 0_usize)];
    let mut max_depth = 0;
    while let Some((index, depth)) = stack.pop() {
        let Some(node) = nodes.get(index).copied() else {
            return Err(RcbError::InvalidBvh { node: index });
        };
        max_depth = max_depth.max(depth);
        if node.is_leaf() {
            let end = node.left_or_first as usize + node.count() as usize;
            if end > triangle_count || node.count() == 0 {
                return Err(RcbError::InvalidBvh { node: index });
            }
        } else {
            let left = node.left_or_first as usize;
            let right = node.right_or_count as usize;
            if left >= nodes.len() || right >= nodes.len() || left == index || right == index {
                return Err(RcbError::InvalidBvh { node: index });
            }
            stack.push((left, depth + 1));
            stack.push((right, depth + 1));
        }
    }
    Ok(max_depth)
}

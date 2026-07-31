//! 16-bin SAH software BVH shared by the native CPU validator and GPU layout.

use std::cmp::Ordering;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::constants::{BVH_LEAF_TRIANGLES, BVH_SAH_BINS, BVH_STACK_SIZE};

const LEAF_BIT: u32 = 0x8000_0000;

#[derive(Clone, Copy, Debug)]
pub struct Triangle {
    pub positions: [Vec3; 3],
    pub normals: [Vec3; 3],
    pub uvs: [[f32; 2]; 3],
    pub albedo: Vec3,
    pub emissive: Vec3,
    pub material: f32,
    pub alpha_cutoff: f32,
}

impl Triangle {
    #[must_use]
    pub fn new(a: Vec3, b: Vec3, c: Vec3, albedo: Vec3, emissive: Vec3) -> Self {
        let normal = (b - a).cross(c - a).normalize_or_zero();
        Self {
            positions: [a, b, c],
            normals: [normal; 3],
            uvs: [[0.0; 2]; 3],
            albedo,
            emissive,
            material: -1.0,
            alpha_cutoff: 0.0,
        }
    }

    #[must_use]
    pub fn bounds(self) -> Aabb {
        Aabb::from_points(&self.positions)
    }

    #[must_use]
    pub fn centroid(self) -> Vec3 {
        (self.positions[0] + self.positions[1] + self.positions[2]) / 3.0
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuTriangle {
    pub a: [f32; 4],
    pub edge_ab: [f32; 4],
    pub edge_ac: [f32; 4],
    pub albedo: [f32; 4],
    pub emissive: [f32; 4],
    pub uv_ab: [f32; 4],
    pub uv_c_material: [f32; 4],
    pub normal_oct: [u32; 4],
}

impl From<Triangle> for GpuTriangle {
    fn from(value: Triangle) -> Self {
        let a = value.positions[0];
        Self {
            a: a.extend(0.0).to_array(),
            edge_ab: (value.positions[1] - a).extend(0.0).to_array(),
            edge_ac: (value.positions[2] - a).extend(0.0).to_array(),
            albedo: value.albedo.extend(0.0).to_array(),
            emissive: value.emissive.extend(0.0).to_array(),
            uv_ab: [value.uvs[0][0], value.uvs[0][1], value.uvs[1][0], value.uvs[1][1]],
            uv_c_material: [
                value.uvs[2][0],
                value.uvs[2][1],
                value.material,
                value.alpha_cutoff,
            ],
            normal_oct: [
                pack_oct_normal(value.normals[0]),
                pack_oct_normal(value.normals[1]),
                pack_oct_normal(value.normals[2]),
                0,
            ],
        }
    }
}

fn pack_oct_normal(normal: Vec3) -> u32 {
    let normal = normal.normalize_or_zero();
    let denominator = (normal.x.abs() + normal.y.abs() + normal.z.abs()).max(1.0e-8);
    let mut p = normal.truncate() / denominator;
    if normal.z < 0.0 {
        p = glam::Vec2::new((1.0 - p.y.abs()).copysign(p.x), (1.0 - p.x.abs()).copysign(p.y));
    }
    let encoded = (p * 0.5 + glam::Vec2::splat(0.5)).clamp(glam::Vec2::ZERO, glam::Vec2::ONE);
    let x = (encoded.x * 65_535.0).round() as u32;
    let y = (encoded.y * 65_535.0).round() as u32;
    x | (y << 16)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub const EMPTY: Self = Self {
        min: Vec3::splat(f32::INFINITY),
        max: Vec3::splat(f32::NEG_INFINITY),
    };

    #[must_use]
    pub fn from_points(points: &[Vec3]) -> Self {
        points
            .iter()
            .fold(Self::EMPTY, |bounds, &point| bounds.extend_point(point))
    }

    #[must_use]
    pub fn extend_point(self, point: Vec3) -> Self {
        Self {
            min: self.min.min(point),
            max: self.max.max(point),
        }
    }

    #[must_use]
    pub fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    #[must_use]
    pub fn surface_area(self) -> f32 {
        let extent = (self.max - self.min).max(Vec3::ZERO);
        2.0 * (extent.x * extent.y + extent.x * extent.z + extent.y * extent.z)
    }

    #[must_use]
    pub fn hit(self, ray: Ray, maximum: f32) -> Option<f32> {
        let mut near = ray.minimum;
        let mut far = maximum;
        for axis in 0..3 {
            if ray.direction[axis].abs() < 1.0e-12 {
                if ray.origin[axis] < self.min[axis] || ray.origin[axis] > self.max[axis] {
                    return None;
                }
                continue;
            }
            let inverse = ray.direction[axis].recip();
            let t0 = (self.min[axis] - ray.origin[axis]) * inverse;
            let t1 = (self.max[axis] - ray.origin[axis]) * inverse;
            near = near.max(t0.min(t1));
            far = far.min(t0.max(t1));
            if near > far {
                return None;
            }
        }
        (near <= far).then_some(near)
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuBvhNode {
    pub bounds_min: [f32; 3],
    pub left_or_first: u32,
    pub bounds_max: [f32; 3],
    pub right_or_count: u32,
}

impl GpuBvhNode {
    #[must_use]
    pub const fn is_leaf(self) -> bool {
        self.right_or_count & LEAF_BIT != 0
    }

    #[must_use]
    pub const fn count(self) -> u32 {
        self.right_or_count & !LEAF_BIT
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
    pub minimum: f32,
    pub maximum: f32,
}

impl Ray {
    #[must_use]
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self {
            origin,
            direction: direction.normalize_or_zero(),
            minimum: 1.0e-3,
            maximum: f32::INFINITY,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub distance: f32,
    pub position: Vec3,
    pub normal: Vec3,
    pub albedo: Vec3,
    pub emissive: Vec3,
    pub triangle: u32,
}

#[derive(Clone, Debug)]
struct BuildReference {
    triangle: Triangle,
    bounds: Aabb,
    centroid: Vec3,
}

#[derive(Clone, Debug)]
pub struct Bvh {
    pub nodes: Vec<GpuBvhNode>,
    pub triangles: Vec<GpuTriangle>,
    cpu_triangles: Vec<Triangle>,
    pub bounds: Aabb,
    pub max_depth: usize,
}

impl Bvh {
    #[must_use]
    pub fn build(triangles: &[Triangle]) -> Self {
        if triangles.is_empty() {
            return Self {
                nodes: vec![GpuBvhNode::zeroed()],
                triangles: Vec::new(),
                cpu_triangles: Vec::new(),
                bounds: Aabb::EMPTY,
                max_depth: 0,
            };
        }
        let mut references: Vec<_> = triangles
            .iter()
            .copied()
            .map(|triangle| BuildReference {
                bounds: triangle.bounds(),
                centroid: triangle.centroid(),
                triangle,
            })
            .collect();
        let mut nodes = Vec::with_capacity(triangles.len() * 2);
        let mut ordered = Vec::with_capacity(triangles.len());
        let mut max_depth = 0;
        let length = references.len();
        build_node(
            &mut references,
            0,
            length,
            0,
            &mut max_depth,
            &mut nodes,
            &mut ordered,
        );
        let bounds = Aabb {
            min: Vec3::from_array(nodes[0].bounds_min),
            max: Vec3::from_array(nodes[0].bounds_max),
        };
        Self {
            nodes,
            triangles: ordered.iter().copied().map(Into::into).collect(),
            cpu_triangles: ordered,
            bounds,
            max_depth,
        }
    }

    /// Constructs a BVH from trusted, already SAH-packed data.
    ///
    /// The RCB loader validates every node and leaf range before calling this.
    #[must_use]
    pub(crate) fn from_validated_parts(
        nodes: Vec<GpuBvhNode>,
        cpu_triangles: Vec<Triangle>,
        bounds: Aabb,
        max_depth: usize,
    ) -> Self {
        Self {
            triangles: cpu_triangles.iter().copied().map(Into::into).collect(),
            nodes,
            cpu_triangles,
            bounds,
            max_depth,
        }
    }

    #[must_use]
    pub fn trace(&self, ray: Ray) -> Option<Hit> {
        if self.cpu_triangles.is_empty() {
            return None;
        }
        let mut stack = [0_u32; BVH_STACK_SIZE];
        let mut stack_size = 1;
        let mut closest = ray.maximum;
        let mut result = None;
        while stack_size > 0 {
            stack_size -= 1;
            let node_index = stack[stack_size] as usize;
            let node = self.nodes[node_index];
            let bounds = Aabb {
                min: Vec3::from_array(node.bounds_min),
                max: Vec3::from_array(node.bounds_max),
            };
            if bounds.hit(ray, closest).is_none() {
                continue;
            }
            if node.is_leaf() {
                for triangle_index in node.left_or_first..node.left_or_first + node.count() {
                    if let Some(hit) =
                        intersect_triangle(ray, self.cpu_triangles[triangle_index as usize], triangle_index)
                    {
                        if hit.distance < closest {
                            closest = hit.distance;
                            result = Some(hit);
                        }
                    }
                }
                continue;
            }
            let left = node.left_or_first;
            let right = node.right_or_count;
            let left_near = node_near(&self.nodes[left as usize], ray, closest);
            let right_near = node_near(&self.nodes[right as usize], ray, closest);
            let (near_index, near_t, far_index, far_t) = if left_near <= right_near {
                (left, left_near, right, right_near)
            } else {
                (right, right_near, left, left_near)
            };
            if far_t.is_finite() && stack_size < BVH_STACK_SIZE {
                stack[stack_size] = far_index;
                stack_size += 1;
            }
            if near_t.is_finite() && stack_size < BVH_STACK_SIZE {
                stack[stack_size] = near_index;
                stack_size += 1;
            }
        }
        result
    }

    /// Reference traversal used by validation to catch hierarchy, packing, and
    /// near-first stack errors. This intentionally tests every triangle.
    #[must_use]
    pub fn trace_brute_force(&self, ray: Ray) -> Option<Hit> {
        self.cpu_triangles
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, triangle)| intersect_triangle(ray, triangle, index as u32))
            .min_by(|left, right| left.distance.total_cmp(&right.distance))
    }
}

fn build_node(
    references: &mut [BuildReference],
    start: usize,
    end: usize,
    depth: usize,
    max_depth: &mut usize,
    nodes: &mut Vec<GpuBvhNode>,
    ordered: &mut Vec<Triangle>,
) -> u32 {
    *max_depth = (*max_depth).max(depth);
    let node_index = nodes.len() as u32;
    nodes.push(GpuBvhNode::zeroed());
    let mut bounds = Aabb::EMPTY;
    let mut centroid_bounds = Aabb::EMPTY;
    for reference in &references[start..end] {
        bounds = bounds.union(reference.bounds);
        centroid_bounds = centroid_bounds.extend_point(reference.centroid);
    }
    let count = end - start;
    if count <= BVH_LEAF_TRIANGLES {
        let first = ordered.len() as u32;
        ordered.extend(references[start..end].iter().map(|reference| reference.triangle));
        nodes[node_index as usize] = GpuBvhNode {
            bounds_min: bounds.min.to_array(),
            left_or_first: first,
            bounds_max: bounds.max.to_array(),
            right_or_count: LEAF_BIT | count as u32,
        };
        return node_index;
    }

    let mut best_axis = 0;
    let mut best_split = 0;
    let mut best_cost = f32::INFINITY;
    for axis in 0..3 {
        let extent = centroid_bounds.max[axis] - centroid_bounds.min[axis];
        if extent <= 1.0e-7 {
            continue;
        }
        let mut bin_counts = [0_usize; BVH_SAH_BINS];
        let mut bin_bounds = [Aabb::EMPTY; BVH_SAH_BINS];
        for reference in &references[start..end] {
            let bin = (((reference.centroid[axis] - centroid_bounds.min[axis]) / extent)
                * BVH_SAH_BINS as f32)
                .floor()
                .clamp(0.0, (BVH_SAH_BINS - 1) as f32) as usize;
            bin_counts[bin] += 1;
            bin_bounds[bin] = bin_bounds[bin].union(reference.bounds);
        }
        let mut left_counts = [0_usize; BVH_SAH_BINS - 1];
        let mut right_counts = [0_usize; BVH_SAH_BINS - 1];
        let mut left_areas = [0.0; BVH_SAH_BINS - 1];
        let mut right_areas = [0.0; BVH_SAH_BINS - 1];
        let mut running_count = 0;
        let mut running_bounds = Aabb::EMPTY;
        for split in 0..BVH_SAH_BINS - 1 {
            running_count += bin_counts[split];
            running_bounds = running_bounds.union(bin_bounds[split]);
            left_counts[split] = running_count;
            left_areas[split] = running_bounds.surface_area();
        }
        running_count = 0;
        running_bounds = Aabb::EMPTY;
        for bin in (1..BVH_SAH_BINS).rev() {
            running_count += bin_counts[bin];
            running_bounds = running_bounds.union(bin_bounds[bin]);
            right_counts[bin - 1] = running_count;
            right_areas[bin - 1] = running_bounds.surface_area();
        }
        for split in 0..BVH_SAH_BINS - 1 {
            if left_counts[split] == 0 || right_counts[split] == 0 {
                continue;
            }
            let cost = left_areas[split] * left_counts[split] as f32
                + right_areas[split] * right_counts[split] as f32;
            if cost < best_cost {
                best_cost = cost;
                best_axis = axis;
                best_split = split;
            }
        }
    }

    let extent = centroid_bounds.max[best_axis] - centroid_bounds.min[best_axis];
    references[start..end].sort_unstable_by(|a, b| {
        a.centroid[best_axis]
            .partial_cmp(&b.centroid[best_axis])
            .unwrap_or(Ordering::Equal)
    });
    let mut middle = if best_cost.is_finite() && extent > 1.0e-7 {
        let split_position =
            centroid_bounds.min[best_axis] + extent * (best_split + 1) as f32 / BVH_SAH_BINS as f32;
        start
            + references[start..end]
                .partition_point(|reference| reference.centroid[best_axis] < split_position)
    } else {
        start + count / 2
    };
    if middle == start || middle == end {
        middle = start + count / 2;
    }
    let left = build_node(references, start, middle, depth + 1, max_depth, nodes, ordered);
    let right = build_node(references, middle, end, depth + 1, max_depth, nodes, ordered);
    nodes[node_index as usize] = GpuBvhNode {
        bounds_min: bounds.min.to_array(),
        left_or_first: left,
        bounds_max: bounds.max.to_array(),
        right_or_count: right,
    };
    node_index
}

fn node_near(node: &GpuBvhNode, ray: Ray, maximum: f32) -> f32 {
    Aabb {
        min: Vec3::from_array(node.bounds_min),
        max: Vec3::from_array(node.bounds_max),
    }
    .hit(ray, maximum)
    .unwrap_or(f32::INFINITY)
}

#[must_use]
pub fn intersect_triangle(ray: Ray, triangle: Triangle, triangle_index: u32) -> Option<Hit> {
    let edge_ab = triangle.positions[1] - triangle.positions[0];
    let edge_ac = triangle.positions[2] - triangle.positions[0];
    let p = ray.direction.cross(edge_ac);
    let determinant = edge_ab.dot(p);
    if determinant.abs() < 1.0e-8 {
        return None;
    }
    let inverse = determinant.recip();
    let t = ray.origin - triangle.positions[0];
    let u = t.dot(p) * inverse;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = t.cross(edge_ab);
    let v = ray.direction.dot(q) * inverse;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let distance = edge_ac.dot(q) * inverse;
    if distance < ray.minimum || distance > ray.maximum {
        return None;
    }
    let w = 1.0 - u - v;
    let mut normal =
        (triangle.normals[0] * w + triangle.normals[1] * u + triangle.normals[2] * v).normalize_or_zero();
    if normal.dot(ray.direction) > 0.0 {
        normal = -normal;
    }
    Some(Hit {
        distance,
        position: ray.origin + ray.direction * distance,
        normal,
        albedo: triangle.albedo,
        emissive: triangle.emissive,
        triangle: triangle_index,
    })
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    fn brute_force(triangles: &[Triangle], ray: Ray) -> Option<Hit> {
        triangles
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, triangle)| intersect_triangle(ray, triangle, index as u32))
            .min_by(|a, b| a.distance.total_cmp(&b.distance))
    }

    #[test]
    fn bvh_matches_brute_force() {
        let mut triangles = Vec::new();
        for z in 0..8 {
            for x in 0..8 {
                let p = Vec3::new(x as f32 - 4.0, ((x * 7 + z * 3) % 5) as f32 * 0.1, z as f32);
                triangles.push(Triangle::new(
                    p,
                    p + Vec3::X,
                    p + Vec3::new(1.0, 0.0, 1.0),
                    Vec3::ONE,
                    Vec3::ZERO,
                ));
                triangles.push(Triangle::new(
                    p,
                    p + Vec3::new(1.0, 0.0, 1.0),
                    p + Vec3::Z,
                    Vec3::ONE,
                    Vec3::ZERO,
                ));
            }
        }
        let bvh = Bvh::build(&triangles);
        assert!(bvh.max_depth < BVH_STACK_SIZE);
        for sample in 0..512 {
            let x = (sample % 32) as f32 * 0.25 - 4.0;
            let z = (sample / 32) as f32 * 0.5;
            let ray = Ray::new(Vec3::new(x, 8.0, z), -Vec3::Y);
            let expected = brute_force(&triangles, ray);
            let actual = bvh.trace(ray);
            assert_eq!(actual.is_some(), expected.is_some());
            if let (Some(actual), Some(expected)) = (actual, expected) {
                assert_abs_diff_eq!(actual.distance, expected.distance, epsilon = 1.0e-5);
            }
        }
    }
}

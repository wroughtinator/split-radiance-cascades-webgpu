//! Native scene registry. IDs and test intent match the twelve browser scenes.

use std::{f32::consts::TAU, path::Path};

use glam::Vec3;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    bvh::{Bvh, Triangle},
    rcb::{RcbError, RcbScene},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[repr(u8)]
pub enum SceneId {
    Laboratory,
    Sponza,
    Canyon,
    Forest,
    Atrium,
    Pipes,
    Temple,
    Orbit,
    Market,
    Stress,
    Cornell,
    Heightmap,
}

impl SceneId {
    pub const ALL: [Self; 12] = [
        Self::Laboratory,
        Self::Sponza,
        Self::Canyon,
        Self::Forest,
        Self::Atrium,
        Self::Pipes,
        Self::Temple,
        Self::Orbit,
        Self::Market,
        Self::Stress,
        Self::Cornell,
        Self::Heightmap,
    ];

    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    #[must_use]
    pub const fn info(self) -> SceneInfo {
        SCENE_INFO[self.index()]
    }

    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|id| {
            id.info().short.eq_ignore_ascii_case(name) || id.info().name.eq_ignore_ascii_case(name)
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SceneInfo {
    pub name: &'static str,
    pub short: &'static str,
    pub description: &'static str,
}

pub const SCENE_INFO: [SceneInfo; 12] = [
    SceneInfo {
        name: "Color bleed laboratory",
        short: "Lab",
        description: "Near-field red/green transfer, hard occluders, emissive geometry, and moving dual lights.",
    },
    SceneInfo {
        name: "Sponza atrium (paper scene)",
        short: "Sponza",
        description: "Official 262k-triangle Crytek Sponza geometry and the paper's red area emitter.",
    },
    SceneInfo {
        name: "Concave canyon heightfield",
        short: "Canyon",
        description: "72x72 terrain with nested craters, ravines, overhangs, and a moving low sun.",
    },
    SceneInfo {
        name: "Dense lantern forest",
        short: "Forest",
        description: "Thin geometry, deep occlusion, high-frequency foliage, and moving colored lanterns.",
    },
    SceneInfo {
        name: "Multi-level atrium",
        short: "Atrium",
        description: "Stairs, balconies, skylight transfer, and cross-floor indirect illumination.",
    },
    SceneInfo {
        name: "Industrial pipe maze",
        short: "Pipes",
        description: "Dense curved pipework, narrow gaps, emissive furnaces, and complex self-occlusion.",
    },
    SceneInfo {
        name: "Sun temple",
        short: "Temple",
        description: "Columns, layered portals, moving sunlight, and warm/cool material transfer.",
    },
    SceneInfo {
        name: "Orbital sculpture field",
        short: "Orbit",
        description: "High-curvature meshes test open-sky misses, distance intervals, and moving lights.",
    },
    SceneInfo {
        name: "Night market",
        short: "Market",
        description: "Colored area emitters, stalls, canopies, and dark-region temporal stability.",
    },
    SceneInfo {
        name: "Megacity stress grid",
        short: "Stress",
        description: "Large terrain, 150 structures, complex monuments, and maximum probe pressure.",
    },
    SceneInfo {
        name: "Cornell box reference",
        short: "Cornell",
        description: "Canonical red/green enclosure, two occluding boxes, ceiling emitter, and animated comparison light.",
    },
    SceneInfo {
        name: "Grand concave heightmap",
        short: "Heightmap",
        description: "128x128 heightmap with nested bowls, ravine, cliff shelves, moving sun, and orbiting fill light.",
    },
];

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct CameraPreset {
    pub position: Vec3,
    pub target: Vec3,
    pub vertical_fov_degrees: f32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SceneSettings {
    pub camera: CameraPreset,
    pub base_spacing: f32,
    pub environment: Vec3,
    pub sun_intensity: f32,
    pub exposure: f32,
    pub point_intensity: f32,
    pub point_orbit: f32,
    pub point_base_height: f32,
    pub point_height: f32,
    pub point_color: Vec3,
    pub sun_height: f32,
    pub sun_horizontal: f32,
}

impl SceneSettings {
    #[must_use]
    pub fn animated_lights(self, time: f32) -> AnimatedLights {
        let phase = time * 0.19;
        let horizontal = Vec3::new(phase.cos(), 0.0, phase.sin()) * self.sun_horizontal.max(1.0e-3);
        let sun_direction = (horizontal + Vec3::Y * self.sun_height.abs().max(0.15)).normalize_or_zero();
        let point_position = Vec3::new(
            (time * 0.43).cos() * self.point_orbit,
            self.point_base_height + (time * 0.31).sin() * self.point_height,
            (time * 0.43).sin() * self.point_orbit,
        );
        AnimatedLights {
            sun_direction,
            sun_radiance: Vec3::new(1.0, 0.92, 0.78) * self.sun_intensity,
            point_position,
            point_radiance: self.point_color * self.point_intensity,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct AnimatedLights {
    pub sun_direction: Vec3,
    pub sun_radiance: Vec3,
    pub point_position: Vec3,
    pub point_radiance: Vec3,
}

#[derive(Clone, Debug)]
pub struct Scene {
    pub id: SceneId,
    pub info: SceneInfo,
    pub settings: SceneSettings,
    pub bvh: Bvh,
    pub source_triangle_count: usize,
}

#[derive(Debug, Error)]
pub enum SceneError {
    #[error("Sponza asset error: {0}")]
    Rcb(#[from] RcbError),
    #[error("scene geometry is empty")]
    Empty,
}

impl Scene {
    pub fn load(id: SceneId, asset_root: impl AsRef<Path>) -> Result<Self, SceneError> {
        if id == SceneId::Sponza {
            let packed = RcbScene::load(asset_root.as_ref().join("sponza.rcb"))?;
            let count = packed.bvh.triangles.len();
            return Ok(Self {
                id,
                info: id.info(),
                settings: settings(id),
                bvh: packed.bvh,
                source_triangle_count: count,
            });
        }
        let mut geometry = Geometry::default();
        build_scene(id, &mut geometry);
        if geometry.triangles.is_empty() {
            return Err(SceneError::Empty);
        }
        let source_triangle_count = geometry.triangles.len();
        Ok(Self {
            id,
            info: id.info(),
            settings: settings(id),
            bvh: Bvh::build(&geometry.triangles),
            source_triangle_count,
        })
    }
}

fn settings(id: SceneId) -> SceneSettings {
    let (position, target, base_spacing, environment, sun_intensity) = match id {
        SceneId::Laboratory => (
            Vec3::new(6.5, 3.7, 13.0),
            Vec3::new(0.0, 1.8, 0.0),
            0.72,
            Vec3::new(0.02, 0.025, 0.04),
            2.2,
        ),
        SceneId::Sponza => (
            Vec3::new(-8.0, 8.0, -0.5),
            Vec3::new(5.0, 2.0, -0.5),
            0.55,
            Vec3::new(0.55, 0.65, 0.82),
            5.2,
        ),
        SceneId::Canyon => (
            Vec3::new(31.0, 18.0, 30.0),
            Vec3::ZERO,
            1.55,
            Vec3::new(0.08, 0.11, 0.16),
            4.2,
        ),
        SceneId::Forest => (
            Vec3::new(25.0, 11.0, 26.0),
            Vec3::new(0.0, 3.0, 0.0),
            1.0,
            Vec3::new(0.008, 0.016, 0.028),
            1.4,
        ),
        SceneId::Atrium => (
            Vec3::new(25.0, 12.0, 25.0),
            Vec3::new(0.0, 3.0, 0.0),
            1.05,
            Vec3::new(0.025, 0.04, 0.065),
            2.8,
        ),
        SceneId::Pipes => (
            Vec3::new(29.0, 13.0, 30.0),
            Vec3::new(0.0, 2.2, 0.0),
            1.2,
            Vec3::new(0.012, 0.018, 0.022),
            1.7,
        ),
        SceneId::Temple => (
            Vec3::new(31.0, 16.0, 33.0),
            Vec3::new(0.0, 3.0, 0.0),
            1.25,
            Vec3::new(0.07, 0.08, 0.12),
            4.8,
        ),
        SceneId::Orbit => (
            Vec3::new(30.0, 14.0, 32.0),
            Vec3::ZERO,
            1.35,
            Vec3::new(0.025, 0.035, 0.09),
            2.0,
        ),
        SceneId::Market => (
            Vec3::new(28.0, 12.0, 28.0),
            Vec3::new(0.0, 1.2, 0.0),
            1.1,
            Vec3::new(0.003, 0.006, 0.018),
            0.35,
        ),
        SceneId::Stress => (
            Vec3::new(50.0, 30.0, 53.0),
            Vec3::new(0.0, 3.0, 0.0),
            2.05,
            Vec3::new(0.02, 0.035, 0.06),
            3.6,
        ),
        SceneId::Cornell => (
            Vec3::new(0.0, 2.65, 8.6),
            Vec3::new(0.0, 2.5, -2.75),
            0.32,
            Vec3::splat(0.0015),
            0.05,
        ),
        SceneId::Heightmap => (
            Vec3::new(67.0, 38.0, 72.0),
            Vec3::new(0.0, -1.0, 0.0),
            2.2,
            Vec3::new(0.055, 0.075, 0.12),
            5.8,
        ),
    };
    let (point_intensity, point_orbit, point_base_height, point_height, point_color, exposure) = match id {
        SceneId::Cornell => (3.5, 1.45, 2.6, 0.75, Vec3::new(1.0, 0.82, 0.62), 1.15),
        SceneId::Heightmap => (15.0, 28.0, 9.0, 6.0, Vec3::new(0.12, 0.45, 1.0), 1.0),
        _ => (2.0, 6.0, 4.0, 2.0, Vec3::new(0.45, 0.65, 1.0), 1.0),
    };
    SceneSettings {
        camera: CameraPreset {
            position,
            target,
            vertical_fov_degrees: 60.0,
        },
        base_spacing,
        environment,
        sun_intensity,
        exposure,
        point_intensity,
        point_orbit,
        point_base_height,
        point_height,
        point_color,
        sun_height: if id == SceneId::Heightmap { -0.52 } else { -0.72 },
        sun_horizontal: if id == SceneId::Heightmap { 0.86 } else { 0.69 },
    }
}

#[derive(Default)]
struct Geometry {
    triangles: Vec<Triangle>,
}

impl Geometry {
    fn triangle(&mut self, a: Vec3, b: Vec3, c: Vec3, albedo: Vec3, emissive: Vec3) {
        self.triangles.push(Triangle::new(a, b, c, albedo, emissive));
    }

    fn quad(&mut self, a: Vec3, b: Vec3, c: Vec3, d: Vec3, albedo: Vec3, emissive: Vec3) {
        self.triangle(a, b, c, albedo, emissive);
        self.triangle(a, c, d, albedo, emissive);
    }

    fn cube(&mut self, center: Vec3, size: Vec3, color: Vec3, emissive: Vec3) {
        let h = size * 0.5;
        let p = [
            center + Vec3::new(-h.x, -h.y, -h.z),
            center + Vec3::new(h.x, -h.y, -h.z),
            center + Vec3::new(h.x, h.y, -h.z),
            center + Vec3::new(-h.x, h.y, -h.z),
            center + Vec3::new(-h.x, -h.y, h.z),
            center + Vec3::new(h.x, -h.y, h.z),
            center + Vec3::new(h.x, h.y, h.z),
            center + Vec3::new(-h.x, h.y, h.z),
        ];
        self.quad(p[1], p[0], p[3], p[2], color, emissive);
        self.quad(p[4], p[5], p[6], p[7], color, emissive);
        self.quad(p[0], p[4], p[7], p[3], color, emissive);
        self.quad(p[5], p[1], p[2], p[6], color, emissive);
        self.quad(p[3], p[7], p[6], p[2], color, emissive);
        self.quad(p[0], p[1], p[5], p[4], color, emissive);
    }

    fn cube_rotated_y(&mut self, center: Vec3, size: Vec3, angle: f32, color: Vec3, emissive: Vec3) {
        let h = size * 0.5;
        let (sine, cosine) = angle.sin_cos();
        let point =
            |x: f32, y: f32, z: f32| center + Vec3::new(x * cosine - z * sine, y, x * sine + z * cosine);
        let p = [
            point(-h.x, -h.y, -h.z),
            point(h.x, -h.y, -h.z),
            point(h.x, h.y, -h.z),
            point(-h.x, h.y, -h.z),
            point(-h.x, -h.y, h.z),
            point(h.x, -h.y, h.z),
            point(h.x, h.y, h.z),
            point(-h.x, h.y, h.z),
        ];
        self.quad(p[1], p[0], p[3], p[2], color, emissive);
        self.quad(p[4], p[5], p[6], p[7], color, emissive);
        self.quad(p[0], p[4], p[7], p[3], color, emissive);
        self.quad(p[5], p[1], p[2], p[6], color, emissive);
        self.quad(p[3], p[7], p[6], p[2], color, emissive);
        self.quad(p[0], p[1], p[5], p[4], color, emissive);
    }

    fn sphere(
        &mut self,
        center: Vec3,
        radius: f32,
        color: Vec3,
        emissive: Vec3,
        rings: usize,
        segments: usize,
    ) {
        let point = |v: usize, u: usize| {
            let theta = v as f32 / rings as f32 * std::f32::consts::PI;
            let phi = u as f32 / segments as f32 * TAU;
            center + radius * Vec3::new(theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin())
        };
        for v in 0..rings {
            for u in 0..segments {
                let p00 = point(v, u);
                let p10 = point(v, u + 1);
                let p01 = point(v + 1, u);
                let p11 = point(v + 1, u + 1);
                if v > 0 {
                    self.triangle(p00, p01, p10, color, emissive);
                }
                if v + 1 < rings {
                    self.triangle(p10, p01, p11, color, emissive);
                }
            }
        }
    }

    fn cylinder(
        &mut self,
        center: Vec3,
        radius: f32,
        height: f32,
        color: Vec3,
        emissive: Vec3,
        segments: usize,
        axis: usize,
    ) {
        let basis = match axis {
            0 => (Vec3::Y, Vec3::X, Vec3::Z),
            2 => (Vec3::X, Vec3::Z, Vec3::Y),
            _ => (Vec3::X, Vec3::Y, Vec3::Z),
        };
        let point = |h: f32, angle: f32| {
            center + basis.1 * h + radius * (basis.0 * angle.cos() + basis.2 * angle.sin())
        };
        let top = center + basis.1 * height * 0.5;
        let bottom = center - basis.1 * height * 0.5;
        for i in 0..segments {
            let a = i as f32 / segments as f32 * TAU;
            let b = (i + 1) as f32 / segments as f32 * TAU;
            let p00 = point(-height * 0.5, a);
            let p10 = point(-height * 0.5, b);
            let p01 = point(height * 0.5, a);
            let p11 = point(height * 0.5, b);
            self.quad(p00, p01, p11, p10, color, emissive);
            self.triangle(top, p11, p01, color, emissive);
            self.triangle(bottom, p00, p10, color, emissive);
        }
    }

    fn torus(
        &mut self,
        center: Vec3,
        major: f32,
        minor: f32,
        color: Vec3,
        major_segments: usize,
        minor_segments: usize,
    ) {
        let sample = |u: usize, v: usize| {
            let u = u as f32 / major_segments as f32 * TAU;
            let v = v as f32 / minor_segments as f32 * TAU;
            center
                + Vec3::new(
                    (major + minor * v.cos()) * u.cos(),
                    minor * v.sin(),
                    (major + minor * v.cos()) * u.sin(),
                )
        };
        for u in 0..major_segments {
            for v in 0..minor_segments {
                self.quad(
                    sample(u, v),
                    sample(u, v + 1),
                    sample(u + 1, v + 1),
                    sample(u + 1, v),
                    color,
                    Vec3::ZERO,
                );
            }
        }
    }

    fn terrain(
        &mut self,
        size: f32,
        resolution: usize,
        height: impl Fn(f32, f32) -> f32,
        color: impl Fn(f32) -> Vec3,
    ) {
        let sample = |x: usize, z: usize| {
            let px = (x as f32 / resolution as f32 - 0.5) * size;
            let pz = (z as f32 / resolution as f32 - 0.5) * size;
            Vec3::new(px, height(px, pz), pz)
        };
        for z in 0..resolution {
            for x in 0..resolution {
                let a = sample(x, z);
                let b = sample(x + 1, z);
                let c = sample(x + 1, z + 1);
                let d = sample(x, z + 1);
                let material = color((a.y + b.y + c.y + d.y) * 0.25);
                self.triangle(a, d, b, material, Vec3::ZERO);
                self.triangle(b, d, c, material, Vec3::ZERO);
            }
        }
    }
}

mod color {
    use glam::Vec3;
    pub const WHITE: Vec3 = Vec3::new(0.73, 0.73, 0.73);
    pub const CHALK: Vec3 = Vec3::new(0.82, 0.80, 0.73);
    pub const RED: Vec3 = Vec3::new(0.63, 0.045, 0.035);
    pub const GREEN: Vec3 = Vec3::new(0.055, 0.42, 0.085);
    pub const BLUE: Vec3 = Vec3::new(0.04, 0.13, 0.52);
    pub const CYAN: Vec3 = Vec3::new(0.04, 0.55, 0.64);
    pub const ORANGE: Vec3 = Vec3::new(0.9, 0.28, 0.045);
    pub const VIOLET: Vec3 = Vec3::new(0.42, 0.08, 0.65);
    pub const YELLOW: Vec3 = Vec3::new(0.82, 0.63, 0.05);
    pub const STONE: Vec3 = Vec3::new(0.38, 0.34, 0.28);
    pub const SAND: Vec3 = Vec3::new(0.52, 0.37, 0.19);
    pub const DARK: Vec3 = Vec3::new(0.035, 0.045, 0.055);
    pub const METAL: Vec3 = Vec3::new(0.24, 0.27, 0.30);
    pub const BARK: Vec3 = Vec3::new(0.16, 0.075, 0.025);
    pub const LEAF: Vec3 = Vec3::new(0.035, 0.18, 0.045);
    pub const CONCRETE: Vec3 = Vec3::new(0.22, 0.23, 0.24);
}

fn deterministic(index: usize) -> f32 {
    (index as f32)
        .mul_add(91.345, 17.23)
        .sin()
        .mul_add(43_758.547, 0.0)
        .fract()
        .abs()
}

fn room(g: &mut Geometry, half: Vec3) {
    use color::*;
    g.cube(
        Vec3::new(0.0, -0.2, 0.0),
        Vec3::new(half.x * 2.0, 0.4, half.z * 2.0),
        CHALK,
        Vec3::ZERO,
    );
    g.cube(
        Vec3::new(-half.x, half.y * 0.5, 0.0),
        Vec3::new(0.35, half.y, half.z * 2.0),
        RED,
        Vec3::ZERO,
    );
    g.cube(
        Vec3::new(half.x, half.y * 0.5, 0.0),
        Vec3::new(0.35, half.y, half.z * 2.0),
        GREEN,
        Vec3::ZERO,
    );
    g.cube(
        Vec3::new(0.0, half.y * 0.5, -half.z),
        Vec3::new(half.x * 2.0, half.y, 0.35),
        WHITE,
        Vec3::ZERO,
    );
    g.cube(
        Vec3::new(0.0, half.y, 0.0),
        Vec3::new(half.x * 2.0, 0.3, half.z * 2.0),
        WHITE,
        Vec3::ZERO,
    );
}

fn build_scene(id: SceneId, g: &mut Geometry) {
    use color::*;
    match id {
        SceneId::Laboratory => {
            room(g, Vec3::new(8.0, 5.0, 8.0));
            g.cube(
                Vec3::new(-2.2, 1.25, 0.5),
                Vec3::new(2.2, 2.5, 2.2),
                WHITE,
                Vec3::ZERO,
            );
            g.cube(
                Vec3::new(2.1, 0.8, -2.1),
                Vec3::new(2.6, 1.6, 2.6),
                BLUE,
                Vec3::ZERO,
            );
            g.sphere(Vec3::new(0.0, 1.05, 2.4), 1.05, YELLOW, Vec3::ZERO, 10, 18);
            g.torus(Vec3::new(0.0, 3.2, -2.6), 1.25, 0.25, CYAN, 20, 10);
            g.cube(
                Vec3::new(0.0, 4.75, 0.0),
                Vec3::new(3.3, 0.08, 3.3),
                WHITE,
                Vec3::new(5.2, 4.6, 3.5),
            );
        }
        SceneId::Sponza => unreachable!("Sponza is loaded from the official RCB asset"),
        SceneId::Canyon => {
            let height = |x: f32, z: f32| {
                let r = Vec3::new(x + 4.0, 0.0, z - 2.0).length();
                let r2 = Vec3::new(x - 7.0, 0.0, z + 7.0).length();
                2.4 * (x * 0.19).sin() * (z * 0.16).cos()
                    - 6.0 * (-r * r / 38.0).exp()
                    - 3.7 * (-r2 * r2 / 19.0).exp()
                    + 0.012 * (x * x + z * z)
            };
            g.terrain(44.0, 72, height, |y| {
                if y < 0.0 {
                    Vec3::new(0.31, 0.18, 0.075)
                } else {
                    SAND
                }
            });
            g.cube(
                Vec3::new(-4.0, -0.7, 2.0),
                Vec3::new(7.0, 0.65, 3.2),
                DARK,
                Vec3::ZERO,
            );
            g.cube(
                Vec3::new(9.0, 3.2, -4.0),
                Vec3::new(8.0, 0.45, 4.0),
                STONE,
                Vec3::ZERO,
            );
        }
        SceneId::Forest => {
            g.terrain(
                34.0,
                32,
                |x, z| 0.35 * (x * 0.4).sin() + 0.28 * (z * 0.5).cos(),
                |_| Vec3::new(0.08, 0.14, 0.055),
            );
            for i in 0..70 {
                let x = (deterministic(i * 3) - 0.5) * 31.0;
                let z = (deterministic(i * 3 + 1) - 0.5) * 31.0;
                let height = 2.5 + deterministic(i * 3 + 2) * 3.0;
                g.cylinder(
                    Vec3::new(x, height * 0.48, z),
                    0.13 + deterministic(i + 99) * 0.12,
                    height,
                    BARK,
                    Vec3::ZERO,
                    9,
                    1,
                );
                g.sphere(
                    Vec3::new(x, height + 0.8, z),
                    1.0 + deterministic(i + 7) * 0.7,
                    if i % 8 == 0 { YELLOW } else { LEAF },
                    Vec3::ZERO,
                    5,
                    9,
                );
            }
            for i in 0..8 {
                g.sphere(
                    Vec3::new(-12.0 + i as f32 * 3.4, 2.0 + (i as f32).sin(), -2.0),
                    0.38,
                    ORANGE,
                    Vec3::new(5.0, 1.6, 0.15),
                    6,
                    10,
                );
            }
        }
        SceneId::Atrium => {
            g.cube(
                Vec3::new(0.0, -0.3, 0.0),
                Vec3::new(28.0, 0.6, 22.0),
                CONCRETE,
                Vec3::ZERO,
            );
            for y in [2.8, 6.0] {
                g.cube(
                    Vec3::new(-11.0, y, 0.0),
                    Vec3::new(1.2, 0.4, 20.0),
                    WHITE,
                    Vec3::ZERO,
                );
                g.cube(
                    Vec3::new(11.0, y, 0.0),
                    Vec3::new(1.2, 0.4, 20.0),
                    WHITE,
                    Vec3::ZERO,
                );
                g.cube(
                    Vec3::new(0.0, y, -9.0),
                    Vec3::new(23.0, 0.4, 1.2),
                    WHITE,
                    Vec3::ZERO,
                );
            }
            for x in [-9.0, -5.0, -1.0, 3.0, 7.0] {
                g.cylinder(Vec3::new(x, 3.1, -7.8), 0.35, 6.2, STONE, Vec3::ZERO, 16, 1);
            }
            for i in 0..10 {
                g.cube(
                    Vec3::new(-7.0 + i as f32 * 0.8, 0.25 + i as f32 * 0.28, 3.0),
                    Vec3::new(0.8, 0.56, 5.5),
                    if i % 2 == 0 { STONE } else { CHALK },
                    Vec3::ZERO,
                );
            }
            g.torus(Vec3::new(3.5, 2.7, 2.0), 2.1, 0.48, VIOLET, 24, 12);
            g.sphere(Vec3::new(-3.8, 1.6, 2.3), 1.55, CYAN, Vec3::ZERO, 12, 20);
        }
        SceneId::Pipes => {
            g.cube(
                Vec3::new(0.0, -0.25, 0.0),
                Vec3::new(34.0, 0.5, 24.0),
                CONCRETE,
                Vec3::ZERO,
            );
            g.cube(
                Vec3::new(0.0, 5.0, -11.5),
                Vec3::new(34.0, 10.0, 0.7),
                DARK,
                Vec3::ZERO,
            );
            for i in 0..15 {
                let x = -14.0 + (i % 5) as f32 * 7.0;
                let z = -8.0 + (i / 5) as f32 * 7.0;
                let y = 1.2 + (i % 3) as f32 * 1.6;
                g.cylinder(
                    Vec3::new(x, y, z),
                    0.38 + (i % 2) as f32 * 0.18,
                    5.5,
                    METAL,
                    Vec3::ZERO,
                    14,
                    i % 3,
                );
                g.torus(
                    Vec3::new(x, y + 1.4, z),
                    1.0 + (i % 3) as f32 * 0.2,
                    0.24,
                    if i % 4 == 0 { ORANGE } else { METAL },
                    18,
                    8,
                );
            }
            for i in 0..7 {
                g.cube(
                    Vec3::new(-13.0 + i as f32 * 4.3, 1.2, -10.7),
                    Vec3::new(2.8, 2.4, 0.5),
                    DARK,
                    Vec3::new(4.5, 0.35 + 0.15 * i as f32, 0.05),
                );
            }
        }
        SceneId::Temple => {
            g.cube(
                Vec3::new(0.0, -0.25, 0.0),
                Vec3::new(32.0, 0.5, 32.0),
                SAND,
                Vec3::ZERO,
            );
            for z in [-10.0, -4.0, 4.0, 10.0] {
                for x in [-11.0, -5.0, 5.0, 11.0] {
                    g.cylinder(Vec3::new(x, 3.8, z), 0.65, 7.6, STONE, Vec3::ZERO, 18, 1);
                    g.cube(Vec3::new(x, 7.75, z), Vec3::splat(2.1), SAND, Vec3::ZERO);
                }
            }
            g.cube(
                Vec3::new(0.0, 0.7, -13.0),
                Vec3::new(9.0, 1.4, 5.0),
                RED,
                Vec3::ZERO,
            );
            g.sphere(Vec3::new(0.0, 2.7, -13.0), 1.75, YELLOW, Vec3::ZERO, 12, 20);
        }
        SceneId::Orbit => {
            g.cube(
                Vec3::new(0.0, -3.2, 0.0),
                Vec3::new(38.0, 0.45, 38.0),
                DARK,
                Vec3::ZERO,
            );
            for i in 0..22 {
                let a = i as f32 / 22.0 * TAU;
                let radius = 5.0 + (i % 5) as f32 * 2.4;
                let position = Vec3::new(
                    a.cos() * radius,
                    -0.5 + (i as f32 * 1.7).sin() * 3.2,
                    a.sin() * radius,
                );
                match i % 3 {
                    0 => g.torus(
                        position,
                        1.4 + (i % 4) as f32 * 0.2,
                        0.3,
                        if i % 2 == 0 { CYAN } else { VIOLET },
                        20,
                        10,
                    ),
                    1 => g.sphere(
                        position,
                        1.0 + (i % 4) as f32 * 0.3,
                        if i % 2 == 0 { ORANGE } else { BLUE },
                        Vec3::ZERO,
                        10,
                        18,
                    ),
                    _ => g.cylinder(position, 0.65, 3.3, METAL, Vec3::ZERO, 16, i % 3),
                }
            }
            g.sphere(
                Vec3::new(0.0, 1.0, 0.0),
                2.2,
                WHITE,
                Vec3::new(1.8, 0.45, 3.5),
                12,
                22,
            );
        }
        SceneId::Market => {
            g.cube(
                Vec3::new(0.0, -0.2, 0.0),
                Vec3::new(34.0, 0.4, 24.0),
                DARK,
                Vec3::ZERO,
            );
            for row in 0..3 {
                for i in 0..8 {
                    let x = -14.0 + i as f32 * 4.0;
                    let z = -8.0 + row as f32 * 8.0;
                    let color = [RED, BLUE, GREEN][i % 3];
                    g.cube(Vec3::new(x, 1.0, z), Vec3::new(3.2, 2.0, 3.2), color, Vec3::ZERO);
                    g.sphere(
                        Vec3::new(x, 2.5, z),
                        0.45,
                        WHITE,
                        [
                            Vec3::new(4.5, 0.3, 0.12),
                            Vec3::new(0.15, 1.3, 4.8),
                            Vec3::new(0.15, 4.0, 0.7),
                        ][i % 3],
                        7,
                        12,
                    );
                }
            }
        }
        SceneId::Stress => {
            g.terrain(
                60.0,
                56,
                |x, z| {
                    0.7 * (x * 0.17).sin() * (z * 0.13).cos()
                        - 2.8 * (-((x - 8.0).powi(2) + (z + 6.0).powi(2)) / 70.0).exp()
                },
                |_| CONCRETE,
            );
            for i in 0..150 {
                let x = (deterministic(i * 4) - 0.5) * 54.0;
                let z = (deterministic(i * 4 + 1) - 0.5) * 54.0;
                let height = 1.5 + deterministic(i * 4 + 2) * 9.0;
                let width = 1.1 + deterministic(i * 4 + 3) * 2.2;
                g.cube(
                    Vec3::new(x, height * 0.5, z),
                    Vec3::new(width, height, width * (0.75 + deterministic(i + 55) * 0.7)),
                    if i % 7 == 0 {
                        RED
                    } else if i % 5 == 0 {
                        BLUE
                    } else {
                        DARK
                    },
                    Vec3::ZERO,
                );
            }
            g.sphere(
                Vec3::new(0.0, 8.0, 0.0),
                3.0,
                WHITE,
                Vec3::new(2.5, 0.25, 0.08),
                14,
                24,
            );
        }
        SceneId::Cornell => {
            let left = -2.78;
            let right = 2.78;
            let floor = 0.0;
            let ceiling = 5.49;
            let back = -5.59;
            let front = 0.15;
            g.quad(
                Vec3::new(left, floor, front),
                Vec3::new(right, floor, front),
                Vec3::new(right, floor, back),
                Vec3::new(left, floor, back),
                WHITE,
                Vec3::ZERO,
            );
            g.quad(
                Vec3::new(left, ceiling, back),
                Vec3::new(right, ceiling, back),
                Vec3::new(right, ceiling, front),
                Vec3::new(left, ceiling, front),
                WHITE,
                Vec3::ZERO,
            );
            g.quad(
                Vec3::new(left, floor, back),
                Vec3::new(right, floor, back),
                Vec3::new(right, ceiling, back),
                Vec3::new(left, ceiling, back),
                WHITE,
                Vec3::ZERO,
            );
            g.quad(
                Vec3::new(left, floor, front),
                Vec3::new(left, floor, back),
                Vec3::new(left, ceiling, back),
                Vec3::new(left, ceiling, front),
                RED,
                Vec3::ZERO,
            );
            g.quad(
                Vec3::new(right, floor, back),
                Vec3::new(right, floor, front),
                Vec3::new(right, ceiling, front),
                Vec3::new(right, ceiling, back),
                GREEN,
                Vec3::ZERO,
            );
            g.cube_rotated_y(
                Vec3::new(-1.05, 0.82, -3.8),
                Vec3::new(1.65, 1.64, 1.65),
                -0.30,
                WHITE,
                Vec3::ZERO,
            );
            g.cube_rotated_y(
                Vec3::new(1.0, 1.55, -2.05),
                Vec3::new(1.75, 3.1, 1.75),
                0.28,
                WHITE,
                Vec3::ZERO,
            );
            g.quad(
                Vec3::new(-0.65, ceiling - 0.025, -3.9),
                Vec3::new(0.65, ceiling - 0.025, -3.9),
                Vec3::new(0.65, ceiling - 0.025, -2.85),
                Vec3::new(-0.65, ceiling - 0.025, -2.85),
                Vec3::new(0.9, 0.88, 0.72),
                Vec3::new(8.5, 7.4, 5.8),
            );
        }
        SceneId::Heightmap => {
            let height = |x: f32, z: f32| {
                let bowl = Vec3::new(x + 12.0, 0.0, z - 7.0).length();
                let crater = Vec3::new(x - 16.0, 0.0, z + 13.0).length();
                let ravine = (z - 7.0 * (x * 0.075).sin() - 2.0 * (x * 0.22).sin()).abs();
                let terracing = (((x * 0.055).sin() + (z * 0.061).cos() + 2.0) * 2.2).floor() / 2.2;
                2.8 * (x * 0.075).sin() * (z * 0.068).cos()
                    - 10.5 * (-bowl * bowl / 155.0).exp()
                    - 7.0 * (-crater * crater / 88.0).exp()
                    - 5.8 * (-ravine * ravine / 5.5).exp()
                    + 0.7 * terracing
                    + 0.0022 * (x * x + z * z)
            };
            g.terrain(92.0, 128, height, |y| {
                if y < -4.0 {
                    Vec3::new(0.16, 0.12, 0.075)
                } else if y > 7.0 {
                    Vec3::new(0.58, 0.59, 0.56)
                } else {
                    Vec3::new(0.38, 0.29, 0.17)
                }
            });
            for i in 0..9 {
                let x = -30.0 + i as f32 * 7.5;
                let z = -15.0 + 5.0 * (i as f32 * 0.9).sin();
                let y = height(x, z) + 2.4;
                g.cube(
                    Vec3::new(x, y, z),
                    Vec3::new(7.8, 0.55, 5.2),
                    if i % 2 == 0 { STONE } else { SAND },
                    Vec3::ZERO,
                );
                g.torus(Vec3::new(x, y - 0.8, z), 1.55, 0.28, DARK, 18, 8);
            }
            for i in 0..18 {
                let angle = i as f32 / 18.0 * TAU;
                let radius = 18.0 + 4.0 * (i as f32 * 1.7).sin();
                let x = angle.cos() * radius;
                let z = angle.sin() * radius;
                g.sphere(
                    Vec3::new(x, height(x, z) + 1.2, z),
                    0.65 + (i % 3) as f32 * 0.18,
                    if i % 2 == 0 { ORANGE } else { CYAN },
                    Vec3::ZERO,
                    8,
                    14,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_exact_web_scene_order() {
        assert_eq!(SceneId::ALL.len(), 12);
        assert_eq!(SceneId::ALL[1], SceneId::Sponza);
        assert_eq!(SceneId::ALL[10], SceneId::Cornell);
        assert_eq!(SceneId::ALL[11], SceneId::Heightmap);
    }

    #[test]
    fn cornell_and_heightmap_have_required_geometry() {
        let cornell = Scene::load(SceneId::Cornell, ".").unwrap();
        let heightmap = Scene::load(SceneId::Heightmap, ".").unwrap();
        assert!(cornell.source_triangle_count >= 36);
        assert!(heightmap.source_triangle_count >= 128 * 128 * 2);
        assert!(heightmap.bvh.max_depth < crate::constants::BVH_STACK_SIZE);
    }

    #[test]
    fn packed_sponza_preserves_uv_material_and_cutout_data() {
        let assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("../public/models");
        let sponza = Scene::load(SceneId::Sponza, assets).unwrap();
        let triangles = &sponza.bvh.triangles;
        assert!(triangles.iter().any(|triangle| triangle.uv_c_material[2] >= 0.0));
        assert!(triangles.iter().any(|triangle| triangle.uv_c_material[3] > 0.0));
        assert!(triangles.iter().any(|triangle| {
            triangle.uv_ab.iter().any(|coordinate| coordinate.abs() > 1.0e-6)
                || triangle.uv_c_material[..2]
                    .iter()
                    .any(|coordinate| coordinate.abs() > 1.0e-6)
        }));
    }

    #[test]
    fn every_procedural_scene_has_finite_bounds_and_moving_lights() {
        for id in SceneId::ALL {
            if id == SceneId::Sponza {
                continue;
            }
            let scene = Scene::load(id, ".").unwrap();
            assert!(scene.bvh.bounds.min.is_finite());
            assert!(scene.bvh.bounds.max.is_finite());
            assert_ne!(
                scene.settings.animated_lights(0.0).sun_direction,
                scene.settings.animated_lights(1.0).sun_direction
            );
        }
    }
}

//! Sokol GPU application and explicit Split RC frame graph.

use std::{
    collections::{BTreeMap, BTreeSet},
    env, ffi,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use glam::{Mat4, Vec3};
use sokol::{app as sapp, gfx as sg, glue as sglue, log as slog};
use thiserror::Error;

use crate::{
    constants::{
        CASCADE_COUNT, CASCADE_DIRECTIONS, HASH_CAPACITY, PROBE_CAPACITY, TOTAL_DIRECTIONAL_VALUES,
        TOTAL_HASH_SLOTS, TOTAL_PROBES,
    },
    profiling::{FrameDiagnostics, FrameProfiler},
    scene::{Scene, SceneId},
};

#[rustfmt::skip]
#[path = "shader.rs"]
mod shader;

const INITIAL_WIDTH: i32 = 1280;
const INITIAL_HEIGHT: i32 = 720;
/// Algorithm 3 assigns one canonical ray owner to every internal-resolution
/// visible pixel. Performance scaling changes the internal resolution; it must
/// never decimate the ray-owner grid because that makes probe populations
/// camera dependent.
const RAY_STRIDE: u32 = 1;
const MAX_INTERNAL_WIDTH: u32 = 960;
const MAX_INTERNAL_HEIGHT: u32 = 540;
const RANK_BLOCK_SIZE: u32 = 512;
const COUNTER_WORDS: u32 = 4 + TOTAL_PROBES * 2 + 16;
// Atomic publication state, hash token, two collision-verified logical-key
// words, and compact probe index per slot.
const HASH_FRAME_STRIDE: u32 = TOTAL_HASH_SLOTS * 5;
const META_FRAME_STRIDE: u32 = TOTAL_PROBES * 4;
const ACCUM_FRAME_STRIDE: u32 = TOTAL_DIRECTIONAL_VALUES * 5;
const CONE_FRAME_STRIDE: u32 = TOTAL_DIRECTIONAL_VALUES * 4;
const IRRADIANCE_FRAME_STRIDE: u32 = PROBE_CAPACITY[0] * 64 * 4;
const GPU_STAGE_NAMES: [&str; 7] = [
    "clear",
    "primary_and_direct_shadows",
    "sparse_probe_construction",
    "split_trace_and_indirect_shadows",
    "merge_and_irradiance",
    "shade",
    "present",
];
const GPU_PROFILE_SAMPLE_INTERVAL: usize = 10;
const BENCHMARK_WARMUP_FRAMES: u32 = 60;

#[derive(Clone, Debug, Default, serde::Serialize)]
struct GpuStageTimingSummary {
    sample_count: usize,
    minimum_ms: f64,
    median_ms: f64,
    p95_ms: f64,
    maximum_ms: f64,
}

fn summarize_gpu_samples(samples: &[f64]) -> GpuStageTimingSummary {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.is_empty() {
        return GpuStageTimingSummary::default();
    }
    let percentile = |fraction: f64| sorted[((sorted.len() - 1) as f64 * fraction).round() as usize];
    GpuStageTimingSummary {
        sample_count: sorted.len(),
        minimum_ms: sorted[0],
        median_ms: percentile(0.5),
        p95_ms: percentile(0.95),
        maximum_ms: sorted[sorted.len() - 1],
    }
}

fn adaptive_scale_step(current_scale: f32, callback_seconds: f32, recovery_probe_ready: bool) -> f32 {
    if callback_seconds > 1.0 / 48.0 {
        (current_scale * 0.9).max(0.5)
    } else if callback_seconds < 1.0 / 75.0 || recovery_probe_ready {
        (current_scale * 1.05).min(1.0)
    } else {
        current_scale
    }
}

fn feature_value_enabled(value: Option<&str>) -> bool {
    value.is_some_and(|value| matches!(value, "1" | "true" | "on"))
}

#[derive(Clone, Copy, Debug)]
struct StateLayout {
    hash_offset: u32,
    hash_frame_stride: u32,
    meta_offset: u32,
    meta_frame_stride: u32,
    counters_offset: u32,
    ray_map_offset: u32,
    blocks_offset: u32,
    block_count: u32,
    sample_count: u32,
    ray_slot_count: u32,
    total_words: u32,
}

impl StateLayout {
    fn new(width: u32, height: u32) -> Self {
        let sample_width = width.div_ceil(RAY_STRIDE);
        let sample_height = height.div_ceil(RAY_STRIDE);
        let sample_count = sample_width * sample_height;
        // Fine/coarse primary plus one previous-frame secondary stream. The
        // secondary cache feeds itself through its previous irradiance atlas.
        let ray_slot_count = sample_count * 3;
        let block_count = ray_slot_count.div_ceil(RANK_BLOCK_SIZE);
        let hash_offset = 0;
        let meta_offset = hash_offset + HASH_FRAME_STRIDE * 2;
        let counters_offset = meta_offset + META_FRAME_STRIDE * 2;
        let ray_map_offset = counters_offset + COUNTER_WORDS;
        let ray_map_words = ray_slot_count * 2;
        let hit_record_words = sample_count * 8 * 2;
        let blocks_offset = ray_map_offset + ray_map_words + hit_record_words;
        let total_words = blocks_offset + PROBE_CAPACITY[0] * block_count;
        Self {
            hash_offset,
            hash_frame_stride: HASH_FRAME_STRIDE,
            meta_offset,
            meta_frame_stride: META_FRAME_STRIDE,
            counters_offset,
            ray_map_offset,
            blocks_offset,
            block_count,
            sample_count,
            ray_slot_count,
            total_words,
        }
    }

    fn hit_frame_offset(self, frame: u32) -> u32 {
        self.ray_map_offset + self.ray_slot_count * 2 + frame * self.sample_count * 8
    }
}

#[derive(Clone, Copy, Debug)]
struct RadianceLayout {
    accum_offset: u32,
    cone_offset: u32,
    cone_frame_stride: u32,
    irradiance_offset: u32,
    irradiance_frame_stride: u32,
    total_words: u32,
}

impl RadianceLayout {
    const fn new() -> Self {
        let accum_offset = 0;
        let cone_offset = accum_offset + ACCUM_FRAME_STRIDE * 2;
        let irradiance_offset = cone_offset + CONE_FRAME_STRIDE * 2;
        let total_words = irradiance_offset + IRRADIANCE_FRAME_STRIDE * 2;
        Self {
            accum_offset,
            cone_offset,
            cone_frame_stride: CONE_FRAME_STRIDE,
            irradiance_offset,
            irradiance_frame_stride: IRRADIANCE_FRAME_STRIDE,
            total_words,
        }
    }
}

#[derive(Debug, Error)]
enum GpuError {
    #[error("this Sokol backend does not expose compute shaders")]
    ComputeUnavailable,
    #[error(
        "backend exposes only {actual} storage buffers per stage; Split RC requires at least {required}"
    )]
    StorageBindingLimit { actual: i32, required: i32 },
    #[error("required RGBA8/RGBA16F storage images are unavailable")]
    StorageImageUnavailable,
    #[error("Sokol resource creation failed for {0}")]
    Resource(&'static str),
    #[error("failed to load scene: {0}")]
    Scene(#[from] crate::scene::SceneError),
    #[error("failed to decode material atlas: {0}")]
    Image(#[from] image::ImageError),
}

#[derive(Clone, Copy)]
struct Pipelines {
    clear_state: sg::Pipeline,
    clear_radiance: sg::Pipeline,
    primary: sg::Pipeline,
    insert_primary: sg::Pipeline,
    insert_secondary: sg::Pipeline,
    insert_retry: sg::Pipeline,
    canonicalize: sg::Pipeline,
    map_primary: sg::Pipeline,
    map_secondary: sg::Pipeline,
    prefix_blocks: sg::Pipeline,
    insert_parent: sg::Pipeline,
    insert_parent_retry: sg::Pipeline,
    propagate_parent: sg::Pipeline,
    assign_offsets: sg::Pipeline,
    trace_split: sg::Pipeline,
    merge: sg::Pipeline,
    irradiance: sg::Pipeline,
    shade: sg::Pipeline,
    present: sg::Pipeline,
}

impl Pipelines {
    fn create(shaders: &mut Vec<sg::Shader>) -> Result<Self, GpuError> {
        let backend = sg::query_backend();
        let mut compute = |name, description: sg::ShaderDesc| -> Result<sg::Pipeline, GpuError> {
            let shader = sg::make_shader(&description);
            valid_shader(shader, name)?;
            shaders.push(shader);
            let pipeline = sg::make_pipeline(&sg::PipelineDesc {
                compute: true,
                shader,
                ..Default::default()
            });
            valid_pipeline(pipeline, name)?;
            Ok(pipeline)
        };
        let clear_state = compute("clear state", shader::clear_state_shader_desc(backend))?;
        let clear_radiance = compute("clear radiance", shader::clear_radiance_shader_desc(backend))?;
        let primary = compute("primary visibility", shader::primary_shader_desc(backend))?;
        let insert_primary = compute(
            "insert primary probes",
            shader::insert_primary_shader_desc(backend),
        )?;
        let insert_secondary = compute(
            "insert secondary probes",
            shader::insert_secondary_shader_desc(backend),
        )?;
        let insert_retry = compute("retry deferred probes", shader::insert_retry_shader_desc(backend))?;
        let canonicalize = compute(
            "canonicalize sparse probes",
            shader::canonicalize_shader_desc(backend),
        )?;
        let map_primary = compute("map primary owners", shader::map_primary_shader_desc(backend))?;
        let map_secondary = compute("map secondary owners", shader::map_secondary_shader_desc(backend))?;
        let prefix_blocks = compute(
            "stable block prefixes",
            shader::prefix_blocks_shader_desc(backend),
        )?;
        let insert_parent = compute("insert parent probes", shader::insert_parent_shader_desc(backend))?;
        let insert_parent_retry = compute(
            "retry deferred parent probes",
            shader::insert_parent_retry_shader_desc(backend),
        )?;
        let propagate_parent = compute(
            "propagate parent counts",
            shader::propagate_parent_shader_desc(backend),
        )?;
        let assign_offsets = compute(
            "hierarchical R2 offsets",
            shader::assign_offsets_shader_desc(backend),
        )?;
        let trace_split = compute("split rays", shader::trace_split_shader_desc(backend))?;
        let merge = compute("merge cascades", shader::merge_shader_desc(backend))?;
        let irradiance = compute("prefilter irradiance", shader::irradiance_shader_desc(backend))?;
        let shade = compute("shade", shader::shade_shader_desc(backend))?;

        let present_shader = sg::make_shader(&shader::present_shader_desc(backend));
        valid_shader(present_shader, "presentation")?;
        shaders.push(present_shader);
        let present = sg::make_pipeline(&sg::PipelineDesc {
            shader: present_shader,
            primitive_type: sg::PrimitiveType::Triangles,
            ..Default::default()
        });
        valid_pipeline(present, "presentation")?;
        Ok(Self {
            clear_state,
            clear_radiance,
            primary,
            insert_primary,
            insert_secondary,
            insert_retry,
            canonicalize,
            map_primary,
            map_secondary,
            prefix_blocks,
            insert_parent,
            insert_parent_retry,
            propagate_parent,
            assign_offsets,
            trace_split,
            merge,
            irradiance,
            shade,
            present,
        })
    }

    fn destroy(self) {
        for pipeline in [
            self.clear_state,
            self.clear_radiance,
            self.primary,
            self.insert_primary,
            self.insert_secondary,
            self.insert_retry,
            self.canonicalize,
            self.map_primary,
            self.map_secondary,
            self.prefix_blocks,
            self.insert_parent,
            self.insert_parent_retry,
            self.propagate_parent,
            self.assign_offsets,
            self.trace_split,
            self.merge,
            self.irradiance,
            self.shade,
            self.present,
        ] {
            sg::destroy_pipeline(pipeline);
        }
    }
}

struct SceneResources {
    nodes: sg::Buffer,
    triangles: sg::Buffer,
    node_view: sg::View,
    triangle_view: sg::View,
    material_atlas: sg::Image,
    material_view: sg::View,
    material_sampler: sg::Sampler,
}

impl SceneResources {
    fn create(scene: &Scene, asset_root: &Path) -> Result<Self, GpuError> {
        let nodes = immutable_storage_buffer(&scene.bvh.nodes, "BVH nodes")?;
        let triangles = immutable_storage_buffer(&scene.bvh.triangles, "BVH triangles")?;
        let node_view = storage_view(nodes, "BVH node view")?;
        let triangle_view = storage_view(triangles, "BVH triangle view")?;
        let (width, height, pixels) = if scene.id == SceneId::Sponza {
            let decoded = image::open(asset_root.join("sponza-atlas.webp"))?.into_rgba8();
            (
                decoded.width() as i32,
                decoded.height() as i32,
                decoded.into_raw(),
            )
        } else {
            (1, 1, vec![255_u8; 4])
        };
        let mut data = sg::ImageData::new();
        data.mip_levels[0] = sg::slice_as_range(&pixels);
        let material_atlas = sg::make_image(&sg::ImageDesc {
            width,
            height,
            pixel_format: sg::PixelFormat::Rgba8,
            data,
            ..Default::default()
        });
        if sg::query_image_state(material_atlas) != sg::ResourceState::Valid {
            return Err(GpuError::Resource("material atlas"));
        }
        let material_view = sg::make_view(&sg::ViewDesc {
            texture: sg::TextureViewDesc {
                image: material_atlas,
                ..Default::default()
            },
            ..Default::default()
        });
        valid_view(material_view, "material atlas texture view")?;
        let material_sampler = sg::make_sampler(&sg::SamplerDesc {
            min_filter: sg::Filter::Linear,
            mag_filter: sg::Filter::Linear,
            wrap_u: sg::Wrap::Repeat,
            wrap_v: sg::Wrap::Repeat,
            ..Default::default()
        });
        if sg::query_sampler_state(material_sampler) != sg::ResourceState::Valid {
            return Err(GpuError::Resource("material sampler"));
        }
        Ok(Self {
            nodes,
            triangles,
            node_view,
            triangle_view,
            material_atlas,
            material_view,
            material_sampler,
        })
    }

    fn destroy(self) {
        sg::destroy_view(self.node_view);
        sg::destroy_view(self.triangle_view);
        sg::destroy_view(self.material_view);
        sg::destroy_sampler(self.material_sampler);
        sg::destroy_image(self.material_atlas);
        sg::destroy_buffer(self.nodes);
        sg::destroy_buffer(self.triangles);
    }
}

struct DynamicResources {
    width: u32,
    height: u32,
    state_layout: StateLayout,
    radiance_layout: RadianceLayout,
    state: sg::Buffer,
    radiance: sg::Buffer,
    gbuffer: sg::Buffer,
    state_view: sg::View,
    radiance_view: sg::View,
    gbuffer_view: sg::View,
    output: sg::Image,
    output_storage_view: sg::View,
    output_texture_view: sg::View,
    irradiance_atlas: sg::Image,
    irradiance_atlas_storage_view: sg::View,
    irradiance_atlas_texture_view: sg::View,
    sampler: sg::Sampler,
}

impl DynamicResources {
    fn create(width: u32, height: u32) -> Result<Self, GpuError> {
        let state_layout = StateLayout::new(width, height);
        let radiance_layout = RadianceLayout::new();
        let state = storage_buffer(state_layout.total_words as usize * 4, "Split RC state")?;
        let radiance = storage_buffer(
            radiance_layout.total_words as usize * 4,
            "Split RC interval/cone/irradiance",
        )?;
        let gbuffer = storage_buffer(
            width as usize * height as usize * std::mem::size_of::<shader::Surface>(),
            "primary surface buffer",
        )?;
        let state_view = storage_view(state, "state view")?;
        let radiance_view = storage_view(radiance, "radiance view")?;
        let gbuffer_view = storage_view(gbuffer, "surface view")?;
        let output = sg::make_image(&sg::ImageDesc {
            usage: sg::ImageUsage {
                storage_image: true,
                ..Default::default()
            },
            width: width as i32,
            height: height as i32,
            pixel_format: sg::PixelFormat::Rgba8,
            ..Default::default()
        });
        if sg::query_image_state(output) != sg::ResourceState::Valid {
            return Err(GpuError::Resource("output image"));
        }
        let output_storage_view = sg::make_view(&sg::ViewDesc {
            storage_image: sg::ImageViewDesc {
                image: output,
                ..Default::default()
            },
            ..Default::default()
        });
        valid_view(output_storage_view, "output storage view")?;
        let output_texture_view = sg::make_view(&sg::ViewDesc {
            texture: sg::TextureViewDesc {
                image: output,
                ..Default::default()
            },
            ..Default::default()
        });
        valid_view(output_texture_view, "output texture view")?;
        let irradiance_atlas = sg::make_image(&sg::ImageDesc {
            usage: sg::ImageUsage {
                storage_image: true,
                ..Default::default()
            },
            width: 512,
            height: 1024,
            pixel_format: sg::PixelFormat::Rgba16f,
            ..Default::default()
        });
        if sg::query_image_state(irradiance_atlas) != sg::ResourceState::Valid {
            return Err(GpuError::Resource("irradiance atlas"));
        }
        let irradiance_atlas_storage_view = sg::make_view(&sg::ViewDesc {
            storage_image: sg::ImageViewDesc {
                image: irradiance_atlas,
                ..Default::default()
            },
            ..Default::default()
        });
        valid_view(irradiance_atlas_storage_view, "irradiance atlas storage view")?;
        let irradiance_atlas_texture_view = sg::make_view(&sg::ViewDesc {
            texture: sg::TextureViewDesc {
                image: irradiance_atlas,
                ..Default::default()
            },
            ..Default::default()
        });
        valid_view(irradiance_atlas_texture_view, "irradiance atlas texture view")?;
        let sampler = sg::make_sampler(&sg::SamplerDesc {
            min_filter: sg::Filter::Linear,
            mag_filter: sg::Filter::Linear,
            wrap_u: sg::Wrap::ClampToEdge,
            wrap_v: sg::Wrap::ClampToEdge,
            ..Default::default()
        });
        if sg::query_sampler_state(sampler) != sg::ResourceState::Valid {
            return Err(GpuError::Resource("presentation sampler"));
        }
        Ok(Self {
            width,
            height,
            state_layout,
            radiance_layout,
            state,
            radiance,
            gbuffer,
            state_view,
            radiance_view,
            gbuffer_view,
            output,
            output_storage_view,
            output_texture_view,
            irradiance_atlas,
            irradiance_atlas_storage_view,
            irradiance_atlas_texture_view,
            sampler,
        })
    }

    fn destroy(self) {
        for view in [
            self.state_view,
            self.radiance_view,
            self.gbuffer_view,
            self.output_storage_view,
            self.output_texture_view,
            self.irradiance_atlas_storage_view,
            self.irradiance_atlas_texture_view,
        ] {
            sg::destroy_view(view);
        }
        sg::destroy_sampler(self.sampler);
        sg::destroy_image(self.output);
        sg::destroy_image(self.irradiance_atlas);
        sg::destroy_buffer(self.state);
        sg::destroy_buffer(self.radiance);
        sg::destroy_buffer(self.gbuffer);
    }
}

struct Renderer {
    scene: Scene,
    scene_resources: SceneResources,
    dynamic: Option<DynamicResources>,
    pipelines: Pipelines,
    shaders: Vec<sg::Shader>,
    compute_bindings: sg::Bindings,
    present_bindings: sg::Bindings,
    frame: u32,
    animation_time: f32,
    animate: bool,
    c_minus_one: bool,
    temporal_jitter: bool,
    rough_specular: bool,
    indirect_only: bool,
    previous_view_projection: Mat4,
    resolution_scale: f32,
    adaptive_resolution: bool,
    adaptation_frames: u32,
    adaptation_recovery_seconds: f32,
    gpu_profile_enabled: bool,
    gpu_stage_samples_ms: BTreeMap<String, Vec<f64>>,
    profiler: FrameProfiler,
    last_diagnostics: FrameDiagnostics,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
struct HashPublicationAudit {
    occupied_slots: u32,
    tombstone_slots: u32,
    unpublished_slots: u32,
    reserved_slots: u32,
    canonical_slots: u32,
    duplicate_slots: u32,
    repeated_exact_keys: u32,
}

impl Renderer {
    fn create(scene: Scene, asset_root: &Path) -> Result<Self, GpuError> {
        let features = sg::query_features();
        if !features.compute {
            return Err(GpuError::ComputeUnavailable);
        }
        let limits = sg::query_limits();
        if limits.max_storage_buffer_bindings_per_stage < 5 {
            return Err(GpuError::StorageBindingLimit {
                actual: limits.max_storage_buffer_bindings_per_stage,
                required: 5,
            });
        }
        let output_format = sg::query_pixelformat(sg::PixelFormat::Rgba8);
        let atlas_format = sg::query_pixelformat(sg::PixelFormat::Rgba16f);
        if !output_format.write
            || !output_format.sample
            || !atlas_format.write
            || !atlas_format.sample
            || !atlas_format.filter
        {
            return Err(GpuError::StorageImageUnavailable);
        }
        let mut shaders = Vec::new();
        let pipelines = Pipelines::create(&mut shaders)?;
        let scene_resources = SceneResources::create(&scene, asset_root)?;
        let mut renderer = Self {
            scene,
            scene_resources,
            dynamic: None,
            pipelines,
            shaders,
            compute_bindings: sg::Bindings::new(),
            present_bindings: sg::Bindings::new(),
            frame: 0,
            animation_time: 0.0,
            animate: env::var_os("SPLIT_RC_FREEZE_LIGHTS").is_none(),
            c_minus_one: env::var("SPLIT_RC_C_MINUS_ONE")
                .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "on")),
            temporal_jitter: !env::var("SPLIT_RC_TEMPORAL_JITTER")
                .is_ok_and(|value| matches!(value.as_str(), "0" | "false" | "off")),
            rough_specular: feature_value_enabled(env::var("SPLIT_RC_ROUGH_SPECULAR").ok().as_deref()),
            indirect_only: feature_value_enabled(env::var("SPLIT_RC_INDIRECT_ONLY").ok().as_deref()),
            previous_view_projection: Mat4::IDENTITY,
            resolution_scale: env::var("SPLIT_RC_RESOLUTION_SCALE")
                .ok()
                .and_then(|value| value.parse::<f32>().ok())
                .unwrap_or(1.0)
                .clamp(0.5, 1.0),
            adaptive_resolution: env::var_os("SPLIT_RC_FIXED_SCALE").is_none(),
            adaptation_frames: 0,
            adaptation_recovery_seconds: 0.0,
            gpu_profile_enabled: env::var_os("SPLIT_RC_GPU_PROFILE").is_some(),
            gpu_stage_samples_ms: BTreeMap::new(),
            profiler: FrameProfiler::default(),
            last_diagnostics: FrameDiagnostics::default(),
        };
        renderer.resize_if_needed()?;
        renderer.compute_bindings.views[shader::VIEW_MATERIAL_ATLAS] = renderer.scene_resources.material_view;
        renderer.compute_bindings.samplers[shader::SMP_MATERIAL_SAMPLER] =
            renderer.scene_resources.material_sampler;
        Ok(renderer)
    }

    fn resize_if_needed(&mut self) -> Result<(), GpuError> {
        let window_width = sapp::width().max(1) as u32;
        let window_height = sapp::height().max(1) as u32;
        let cap_scale = (MAX_INTERNAL_WIDTH as f32 / window_width as f32)
            .min(MAX_INTERNAL_HEIGHT as f32 / window_height as f32)
            .min(1.0);
        let scale = self.resolution_scale * cap_scale;
        let width = ((window_width as f32 * scale).round() as u32).max(64) & !7;
        let height = ((window_height as f32 * scale).round() as u32).max(64) & !7;
        if self
            .dynamic
            .as_ref()
            .is_some_and(|dynamic| dynamic.width == width && dynamic.height == height)
        {
            return Ok(());
        }
        if let Some(old) = self.dynamic.take() {
            old.destroy();
        }
        let dynamic = DynamicResources::create(width, height)?;
        self.compute_bindings = sg::Bindings::new();
        self.compute_bindings.views[shader::VIEW_STATE_BUFFER] = dynamic.state_view;
        self.compute_bindings.views[shader::VIEW_RADIANCE_BUFFER] = dynamic.radiance_view;
        self.compute_bindings.views[shader::VIEW_BVH_NODE_BUFFER] = self.scene_resources.node_view;
        self.compute_bindings.views[shader::VIEW_TRIANGLE_BUFFER] = self.scene_resources.triangle_view;
        self.compute_bindings.views[shader::VIEW_GBUFFER_BUFFER] = dynamic.gbuffer_view;
        self.compute_bindings.views[shader::VIEW_OUTPUT_IMAGE] = dynamic.output_storage_view;
        self.compute_bindings.views[shader::VIEW_MATERIAL_ATLAS] = self.scene_resources.material_view;
        self.compute_bindings.samplers[shader::SMP_MATERIAL_SAMPLER] = self.scene_resources.material_sampler;
        self.compute_bindings.views[shader::VIEW_IRRADIANCE_ATLAS_STORAGE] =
            dynamic.irradiance_atlas_storage_view;
        self.compute_bindings.views[shader::VIEW_IRRADIANCE_ATLAS] = dynamic.irradiance_atlas_texture_view;
        self.compute_bindings.samplers[shader::SMP_IRRADIANCE_SAMPLER] = dynamic.sampler;
        self.present_bindings = sg::Bindings::new();
        self.present_bindings.views[shader::VIEW_DISPLAY_TEXTURE] = dynamic.output_texture_view;
        self.present_bindings.samplers[shader::SMP_DISPLAY_SAMPLER] = dynamic.sampler;
        self.dynamic = Some(dynamic);
        self.frame = 0;
        Ok(())
    }

    fn reset_temporal_history(&mut self) -> Result<(), GpuError> {
        if let Some(old) = self.dynamic.take() {
            old.destroy();
        }
        self.frame = 0;
        self.previous_view_projection = Mat4::IDENTITY;
        self.resize_if_needed()
    }

    fn replace_scene(
        &mut self,
        scene: Scene,
        camera: &mut Camera,
        asset_root: &Path,
    ) -> Result<(), GpuError> {
        let resources = SceneResources::create(&scene, asset_root)?;
        let old = std::mem::replace(&mut self.scene_resources, resources);
        old.destroy();
        self.scene = scene;
        camera.reset(
            self.scene.settings.camera.position,
            self.scene.settings.camera.target,
        );
        self.reset_temporal_history()?;
        let dynamic = self.dynamic.as_ref().expect("dynamic resources exist");
        self.compute_bindings.views[shader::VIEW_BVH_NODE_BUFFER] = self.scene_resources.node_view;
        self.compute_bindings.views[shader::VIEW_TRIANGLE_BUFFER] = self.scene_resources.triangle_view;
        self.compute_bindings.views[shader::VIEW_STATE_BUFFER] = dynamic.state_view;
        self.compute_bindings.views[shader::VIEW_MATERIAL_ATLAS] = self.scene_resources.material_view;
        self.compute_bindings.samplers[shader::SMP_MATERIAL_SAMPLER] = self.scene_resources.material_sampler;
        Ok(())
    }

    fn render(
        &mut self,
        camera: &Camera,
        presented_duration: Duration,
        collect_gpu_timestamp: bool,
    ) -> Result<(), GpuError> {
        self.resize_if_needed()?;
        let gpu_timer = if self.gpu_profile_enabled && collect_gpu_timestamp {
            match crate::capture::D3d11GpuTimer::new(GPU_STAGE_NAMES.len()) {
                Ok(timer) => Some(timer),
                Err(error) => {
                    eprintln!("GPU timestamp profiler unavailable: {error}");
                    None
                }
            }
        } else {
            None
        };
        if let Some(timer) = &gpu_timer {
            timer.begin();
        }
        self.profiler.begin_frame();
        let (internal_width, internal_height, state, radiance) = {
            let dynamic = self.dynamic.as_ref().expect("dynamic resources exist");
            (
                dynamic.width,
                dynamic.height,
                dynamic.state_layout,
                dynamic.radiance_layout,
            )
        };
        if self.animate {
            self.animation_time += sapp::frame_duration() as f32;
        }
        let aspect = internal_width as f32 / internal_height as f32;
        let view_projection =
            Mat4::perspective_rh(60_f32.to_radians(), aspect, 0.01, 2_000.0) * camera.view();
        let inverse_view_projection = view_projection.inverse();
        let lights = self.scene.settings.animated_lights(self.animation_time);
        let mut params = shader::FrameParams {
            inverse_view_projection: inverse_view_projection.to_cols_array(),
            view_projection: view_projection.to_cols_array(),
            previous_view_projection: self.previous_view_projection.to_cols_array(),
            camera_time: camera.position.extend(self.animation_time).to_array(),
            environment_base_spacing: self
                .scene
                .settings
                .environment
                .extend(self.scene.settings.base_spacing)
                .to_array(),
            sun_direction_intensity: lights
                .sun_direction
                .extend(self.scene.settings.sun_intensity)
                .to_array(),
            point_position_intensity: lights
                .point_position
                .extend(self.scene.settings.point_intensity)
                .to_array(),
            point_color_exposure: self
                .scene
                .settings
                .point_color
                .extend(self.scene.settings.exposure)
                .to_array(),
            dimensions_frame_i: [
                internal_width as i32,
                internal_height as i32,
                self.frame as i32,
                RAY_STRIDE as i32,
            ],
            state_layout_0_i: [
                state.hash_offset as i32,
                state.hash_frame_stride as i32,
                state.meta_offset as i32,
                state.meta_frame_stride as i32,
            ],
            state_layout_1_i: [
                state.counters_offset as i32,
                state.ray_map_offset as i32,
                state.blocks_offset as i32,
                state.block_count as i32,
            ],
            state_layout_2_i: [
                state.sample_count as i32,
                state.ray_slot_count as i32,
                state.total_words as i32,
                radiance.total_words as i32,
            ],
            radiance_layout_0_i: [
                radiance.accum_offset as i32,
                radiance.cone_offset as i32,
                radiance.cone_frame_stride as i32,
                radiance.irradiance_offset as i32,
            ],
            radiance_layout_1_i: [
                radiance.irradiance_frame_stride as i32,
                i32::from(self.temporal_jitter),
                0,
                0,
            ],
            pass_params_i: [0; 4],
            feature_flags_i: [
                i32::from(self.c_minus_one),
                i32::from(self.rough_specular),
                i32::from(self.temporal_jitter),
                i32::from(self.indirect_only),
            ],
        };

        self.profiler.begin_stage("clear");
        let current = self.frame & 1;
        self.clear_state(
            &mut params,
            state.hash_offset + current * state.hash_frame_stride,
            state.hash_frame_stride,
            u32::MAX,
        );
        self.clear_state(
            &mut params,
            state.meta_offset + current * state.meta_frame_stride,
            state.meta_frame_stride,
            0,
        );
        self.clear_state(&mut params, state.counters_offset, COUNTER_WORDS, 0);
        self.clear_state(
            &mut params,
            state.ray_map_offset,
            state.ray_slot_count * 2,
            u32::MAX,
        );
        self.clear_state(
            &mut params,
            state.hit_frame_offset(current),
            state.sample_count * 8,
            0,
        );
        self.clear_radiance(
            &mut params,
            radiance.accum_offset + current * ACCUM_FRAME_STRIDE,
            ACCUM_FRAME_STRIDE,
        );
        if let Some(timer) = &gpu_timer {
            timer.mark(0);
        }

        self.profiler.begin_stage("primary");
        self.dispatch_2d(
            self.pipelines.primary,
            &params,
            internal_width,
            internal_height,
            8,
            8,
        );
        if let Some(timer) = &gpu_timer {
            timer.mark(1);
        }

        self.profiler.begin_stage("sparse probes");
        self.dispatch_1d(
            self.pipelines.insert_primary,
            &params,
            state.sample_count * 2,
            256,
        );
        self.dispatch_1d(self.pipelines.insert_secondary, &params, state.sample_count, 256);
        let retry_counter_base = state.counters_offset + 4 + TOTAL_PROBES * 2 + 4;
        for retry_pass in 0..31u32 {
            let read_queue = retry_pass & 1;
            let write_queue = 1 - read_queue;
            self.clear_state(&mut params, retry_counter_base + write_queue, 1, 0);
            params.pass_params_i = [
                read_queue as i32,
                write_queue as i32,
                i32::from(retry_pass == 30),
                0,
            ];
            self.dispatch_1d(self.pipelines.insert_retry, &params, state.ray_slot_count, 256);
        }
        params.pass_params_i[0] = 0;
        params.pass_params_i[1..].fill(0);
        self.dispatch_1d(self.pipelines.canonicalize, &params, HASH_CAPACITY[0], 64);
        for cascade in 1..CASCADE_COUNT as u32 {
            params.pass_params_i[0] = cascade as i32;
            self.clear_state(&mut params, retry_counter_base, 2, 0);
            params.pass_params_i = [cascade as i32, 0, 0, 0];
            self.dispatch_1d(
                self.pipelines.insert_parent,
                &params,
                PROBE_CAPACITY[cascade as usize - 1],
                64,
            );
            for retry_pass in 0..31u32 {
                let read_queue = retry_pass & 1;
                let write_queue = 1 - read_queue;
                self.clear_state(&mut params, retry_counter_base + write_queue, 1, 0);
                params.pass_params_i = [
                    cascade as i32,
                    read_queue as i32,
                    write_queue as i32,
                    i32::from(retry_pass == 30),
                ];
                self.dispatch_1d(
                    self.pipelines.insert_parent_retry,
                    &params,
                    PROBE_CAPACITY[cascade as usize - 1],
                    64,
                );
            }
            params.pass_params_i = [cascade as i32, 0, 0, 0];
            self.dispatch_1d(
                self.pipelines.canonicalize,
                &params,
                HASH_CAPACITY[cascade as usize],
                64,
            );
        }
        self.clear_state(
            &mut params,
            state.blocks_offset,
            PROBE_CAPACITY[0] * state.block_count,
            0,
        );
        params.pass_params_i = [0, 0, 0, 0];
        self.dispatch_1d(self.pipelines.map_primary, &params, state.sample_count * 2, 256);
        self.dispatch_1d(self.pipelines.map_secondary, &params, state.sample_count, 256);
        self.dispatch_1d(self.pipelines.prefix_blocks, &params, PROBE_CAPACITY[0], 64);
        for cascade in 1..CASCADE_COUNT as u32 {
            params.pass_params_i = [cascade as i32, 0, 0, 0];
            self.dispatch_1d(
                self.pipelines.propagate_parent,
                &params,
                PROBE_CAPACITY[cascade as usize - 1],
                64,
            );
        }
        for cascade in (0..CASCADE_COUNT as u32).rev() {
            params.pass_params_i[0] = cascade as i32;
            self.dispatch_1d(
                self.pipelines.assign_offsets,
                &params,
                PROBE_CAPACITY[cascade as usize],
                64,
            );
        }
        if let Some(timer) = &gpu_timer {
            timer.mark(2);
        }

        self.profiler.begin_stage("split trace");
        let history_weight_milli = if presented_duration.as_secs_f32() > 0.05 {
            0
        } else if self.animate || self.frame < 120 {
            920
        } else {
            980
        };
        params.pass_params_i = [
            0,
            i32::from(self.animate),
            i32::from(self.c_minus_one),
            history_weight_milli,
        ];
        self.dispatch_1d(self.pipelines.trace_split, &params, state.ray_slot_count, 64);
        if let Some(timer) = &gpu_timer {
            timer.mark(3);
        }

        self.profiler.begin_stage("merge");
        for cascade in (0..CASCADE_COUNT as u32).rev() {
            params.pass_params_i[0] = cascade as i32;
            self.dispatch_1d(
                self.pipelines.merge,
                &params,
                PROBE_CAPACITY[cascade as usize] * CASCADE_DIRECTIONS[cascade as usize],
                64,
            );
        }
        self.dispatch_1d(self.pipelines.irradiance, &params, PROBE_CAPACITY[0] * 64, 64);
        if let Some(timer) = &gpu_timer {
            timer.mark(4);
        }

        self.profiler.begin_stage("shade");
        self.dispatch_2d(
            self.pipelines.shade,
            &params,
            internal_width,
            internal_height,
            8,
            8,
        );
        if let Some(timer) = &gpu_timer {
            timer.mark(5);
        }

        self.profiler.begin_stage("present");
        let mut pass_action = sg::PassAction::new();
        pass_action.colors[0] = sg::ColorAttachmentAction {
            load_action: sg::LoadAction::Dontcare,
            ..Default::default()
        };
        sg::begin_pass(&sg::Pass {
            action: pass_action,
            swapchain: sglue::swapchain(),
            ..Default::default()
        });
        sg::apply_pipeline(self.pipelines.present);
        sg::apply_bindings(&self.present_bindings);
        sg::draw(0, 3, 1);
        sg::end_pass();
        if let Some(timer) = &gpu_timer {
            timer.mark(6);
            timer.end();
        }
        sg::commit();
        self.previous_view_projection = view_projection;
        if let Some(timer) = gpu_timer {
            match timer.resolve_ms() {
                Ok(milliseconds) => {
                    for (name, value) in GPU_STAGE_NAMES.iter().zip(milliseconds) {
                        self.gpu_stage_samples_ms
                            .entry((*name).to_owned())
                            .or_default()
                            .push(value);
                    }
                }
                Err(error) => eprintln!("GPU timestamp resolve failed: {error}"),
            }
        }
        let mut diagnostics = self.last_diagnostics;
        diagnostics.triangle_count = self.scene.source_triangle_count;
        diagnostics.ray_count = u64::from(state.ray_slot_count);
        self.profiler.set_diagnostics(diagnostics);
        self.profiler.end_frame(presented_duration);
        self.frame = self.frame.wrapping_add(1);

        self.adaptation_frames += 1;
        self.adaptation_recovery_seconds += presented_duration.as_secs_f32();
        if self.adaptive_resolution && self.adaptation_frames >= 90 {
            self.adaptation_frames = 0;
            let elapsed = presented_duration.as_secs_f32();
            let old_scale = self.resolution_scale;
            let recovery_probe_ready = self.adaptation_recovery_seconds >= 5.0;
            self.resolution_scale = adaptive_scale_step(old_scale, elapsed, recovery_probe_ready);
            if (old_scale - self.resolution_scale).abs() >= 0.01 {
                self.adaptation_recovery_seconds = 0.0;
                self.resize_if_needed()?;
            } else if recovery_probe_ready {
                self.adaptation_recovery_seconds = 0.0;
            }
        }
        if self.frame % 240 == 0 {
            if let Ok(diagnostics) = self.readback_diagnostics() {
                self.last_diagnostics = diagnostics;
                self.profiler.set_diagnostics(diagnostics);
            }
            let report = self.profiler.report();
            eprintln!(
                "Split RC {:?} {}x{}: frame-callback median {:.2} ms, p95 {:.2} ms, {} triangles, probes {:?}, diagnostics hash={} key-collision={} key-range={} stack={} (GPU timestamps unavailable)",
                sg::query_backend(),
                internal_width,
                internal_height,
                report.frame.median_ms,
                report.frame.p95_ms,
                report.triangle_count,
                report.probe_counts,
                report.hash_overflows,
                self.last_diagnostics.key_hash_collisions,
                self.last_diagnostics.key_range_rejections,
                report.bvh_stack_overflows,
            );
        }
        Ok(())
    }

    fn capture(&self, destination: &Path) -> Result<(), String> {
        let dynamic = self.dynamic.as_ref().ok_or("dynamic resources are unavailable")?;
        crate::capture::capture_rgba8(dynamic.output, dynamic.width, dynamic.height, destination)
    }

    fn read_pixels(&self) -> Result<(u32, u32, Vec<u8>), String> {
        let dynamic = self.dynamic.as_ref().ok_or("dynamic resources are unavailable")?;
        let pixels = crate::capture::read_rgba8(dynamic.output, dynamic.width, dynamic.height)?;
        Ok((dynamic.width, dynamic.height, pixels))
    }

    fn readback_diagnostics(&self) -> Result<FrameDiagnostics, String> {
        let dynamic = self.dynamic.as_ref().ok_or("dynamic resources are unavailable")?;
        let probes =
            crate::capture::read_buffer_words(dynamic.state, dynamic.state_layout.counters_offset, 4)?;
        let diagnostic_offset = dynamic.state_layout.counters_offset + 4 + TOTAL_PROBES * 2;
        let counters = crate::capture::read_buffer_words(dynamic.state, diagnostic_offset, 4)?;
        Ok(FrameDiagnostics {
            triangle_count: self.scene.source_triangle_count,
            ray_count: u64::from(dynamic.state_layout.ray_slot_count),
            probe_counts: probes.try_into().map_err(|_| "invalid probe counter readback")?,
            hash_overflows: counters[0],
            key_hash_collisions: counters[1],
            key_range_rejections: counters[2],
            bvh_stack_overflows: counters[3],
        })
    }

    fn readback_hash_publication_audit(&self) -> Result<HashPublicationAudit, String> {
        let dynamic = self.dynamic.as_ref().ok_or("dynamic resources are unavailable")?;
        let completed_frame = self.frame.wrapping_sub(1) & 1;
        let words = crate::capture::read_buffer_words(
            dynamic.state,
            dynamic.state_layout.hash_offset + completed_frame * dynamic.state_layout.hash_frame_stride,
            dynamic.state_layout.hash_frame_stride,
        )?;
        let mut audit = HashPublicationAudit::default();
        let mut exact_keys = BTreeSet::new();
        let mut base_slot = 0usize;
        for (cascade, &capacity) in HASH_CAPACITY.iter().enumerate() {
            for slot in 0..capacity as usize {
                let address = (base_slot + slot) * 5;
                if words[address] == u32::MAX {
                    continue;
                }
                if words[address] == 0xffff_fffc {
                    audit.tombstone_slots += 1;
                    continue;
                }
                audit.occupied_slots += 1;
                match words[address + 4] {
                    u32::MAX => audit.unpublished_slots += 1,
                    0xffff_fffe => audit.reserved_slots += 1,
                    0xffff_fffd => audit.duplicate_slots += 1,
                    _ => audit.canonical_slots += 1,
                }
                if !exact_keys.insert((
                    cascade as u32,
                    words[address + 1],
                    words[address + 2],
                    words[address + 3],
                )) {
                    audit.repeated_exact_keys += 1;
                }
            }
            base_slot += capacity as usize;
        }
        Ok(audit)
    }

    fn clear_state(&self, params: &mut shader::FrameParams, offset: u32, count: u32, value: u32) {
        params.pass_params_i = [offset as i32, count as i32, value as i32, 0];
        self.dispatch_1d(self.pipelines.clear_state, params, count, 256);
    }

    fn clear_radiance(&self, params: &mut shader::FrameParams, offset: u32, count: u32) {
        params.pass_params_i = [offset as i32, count as i32, 0, 0];
        self.dispatch_1d(self.pipelines.clear_radiance, params, count, 256);
    }

    fn dispatch_1d(
        &self,
        pipeline: sg::Pipeline,
        params: &shader::FrameParams,
        invocations: u32,
        group_size: u32,
    ) {
        sg::begin_pass(&sg::Pass {
            compute: true,
            ..Default::default()
        });
        sg::apply_pipeline(pipeline);
        sg::apply_bindings(&self.compute_bindings);
        sg::apply_uniforms(shader::UB_FRAME_PARAMS, &sg::value_as_range(params));
        sg::dispatch(invocations.div_ceil(group_size) as usize, 1, 1);
        sg::end_pass();
    }

    fn dispatch_2d(
        &self,
        pipeline: sg::Pipeline,
        params: &shader::FrameParams,
        width: u32,
        height: u32,
        group_x: u32,
        group_y: u32,
    ) {
        sg::begin_pass(&sg::Pass {
            compute: true,
            ..Default::default()
        });
        sg::apply_pipeline(pipeline);
        sg::apply_bindings(&self.compute_bindings);
        sg::apply_uniforms(shader::UB_FRAME_PARAMS, &sg::value_as_range(params));
        sg::dispatch(
            width.div_ceil(group_x) as usize,
            height.div_ceil(group_y) as usize,
            1,
        );
        sg::end_pass();
    }

    fn destroy(mut self) {
        if let Some(dynamic) = self.dynamic.take() {
            dynamic.destroy();
        }
        self.scene_resources.destroy();
        self.pipelines.destroy();
        for shader in self.shaders {
            sg::destroy_shader(shader);
        }
    }
}

#[derive(Clone, Debug)]
struct Camera {
    position: Vec3,
    yaw: f32,
    pitch: f32,
}

impl Camera {
    fn new(position: Vec3, target: Vec3) -> Self {
        let mut camera = Self {
            position,
            yaw: 0.0,
            pitch: 0.0,
        };
        camera.reset(position, target);
        camera
    }

    fn reset(&mut self, position: Vec3, target: Vec3) {
        self.position = position;
        let direction = (target - position).normalize_or_zero();
        self.yaw = direction.z.atan2(direction.x);
        self.pitch = direction.y.asin();
    }

    fn direction(&self) -> Vec3 {
        Vec3::new(
            self.pitch.cos() * self.yaw.cos(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.sin(),
        )
        .normalize_or_zero()
    }

    fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.position + self.direction(), Vec3::Y)
    }

    fn update(&mut self, keys: &[bool; 512], delta: f32) {
        let forward = self.direction();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let speed = if key(keys, sapp::Keycode::LeftShift) {
            16.0
        } else {
            6.0
        } * delta;
        if key(keys, sapp::Keycode::W) {
            self.position += forward * speed;
        }
        if key(keys, sapp::Keycode::S) {
            self.position -= forward * speed;
        }
        if key(keys, sapp::Keycode::D) {
            self.position += right * speed;
        }
        if key(keys, sapp::Keycode::A) {
            self.position -= right * speed;
        }
        if key(keys, sapp::Keycode::E) {
            self.position += Vec3::Y * speed;
        }
        if key(keys, sapp::Keycode::Q) {
            self.position -= Vec3::Y * speed;
        }
    }
}

struct App {
    asset_root: PathBuf,
    renderer: Option<Renderer>,
    camera: Camera,
    keys: [bool; 512],
    pending_scene: Option<SceneId>,
    fatal_error: Option<String>,
    capture_dir: Option<PathBuf>,
    capture_scene: usize,
    capture_frames: u32,
    capture_paths: Vec<PathBuf>,
    stability_output: Option<PathBuf>,
    stability_scene: SceneId,
    stability_frame: u32,
    stability_reference_frame: u32,
    stability_reference: Option<Vec<u8>>,
    cache_motion_output: Option<PathBuf>,
    cache_motion_frame: u32,
    cache_motion_accumulated: Option<(u32, u32, Vec<u8>)>,
    cache_motion_accumulated_diagnostics: Option<FrameDiagnostics>,
    cache_motion_accumulated_publication: Option<HashPublicationAudit>,
    animation_response_output: Option<PathBuf>,
    animation_response_frame: u32,
    animation_response_reference: Option<Vec<u8>>,
    animation_response_reference_time: Option<f32>,
    benchmark_target: Option<u32>,
    benchmark_warmup: u32,
    benchmark_started: Option<Instant>,
    benchmark_samples_ms: Vec<f64>,
    benchmark_output: PathBuf,
}

impl App {
    fn new() -> Self {
        let preset = SceneId::Sponza.info();
        let _ = preset;
        Self {
            asset_root: asset_root(),
            renderer: None,
            camera: Camera::new(Vec3::new(-8.0, 8.0, -0.5), Vec3::new(5.0, 2.0, -0.5)),
            keys: [false; 512],
            pending_scene: None,
            fatal_error: None,
            capture_dir: env::var_os("SPLIT_RC_CAPTURE_DIR").map(PathBuf::from),
            capture_scene: 0,
            capture_frames: 0,
            capture_paths: Vec::new(),
            stability_output: env::var_os("SPLIT_RC_STABILITY_OUT").map(PathBuf::from),
            stability_scene: env::var("SPLIT_RC_STABILITY_SCENE")
                .ok()
                .and_then(|name| SceneId::from_name(&name))
                .unwrap_or(SceneId::Laboratory),
            stability_frame: 0,
            stability_reference_frame: env::var("SPLIT_RC_STABILITY_REFERENCE_FRAME")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(1_200),
            stability_reference: None,
            cache_motion_output: env::var_os("SPLIT_RC_CACHE_MOTION_OUT").map(PathBuf::from),
            cache_motion_frame: 0,
            cache_motion_accumulated: None,
            cache_motion_accumulated_diagnostics: None,
            cache_motion_accumulated_publication: None,
            animation_response_output: env::var_os("SPLIT_RC_ANIMATION_RESPONSE_OUT").map(PathBuf::from),
            animation_response_frame: 0,
            animation_response_reference: None,
            animation_response_reference_time: None,
            benchmark_target: env::var("SPLIT_RC_BENCH_FRAMES")
                .ok()
                .and_then(|value| value.parse().ok()),
            benchmark_warmup: BENCHMARK_WARMUP_FRAMES,
            benchmark_started: None,
            benchmark_samples_ms: Vec::new(),
            benchmark_output: env::var_os("SPLIT_RC_BENCH_OUT")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("profile/native-benchmark.json")),
        }
    }

    fn load_scene(&mut self, id: SceneId) {
        self.capture_frames = 0;
        match Scene::load(id, &self.asset_root) {
            Ok(scene) => {
                let result = if let Some(renderer) = &mut self.renderer {
                    renderer.replace_scene(scene, &mut self.camera, &self.asset_root)
                } else {
                    self.camera
                        .reset(scene.settings.camera.position, scene.settings.camera.target);
                    Renderer::create(scene, &self.asset_root).map(|renderer| self.renderer = Some(renderer))
                };
                if let Err(error) = result {
                    self.fatal_error = Some(error.to_string());
                }
            }
            Err(error) => self.fatal_error = Some(error.to_string()),
        }
    }

    fn finish_benchmark(&mut self) {
        if self.benchmark_samples_ms.is_empty() {
            return;
        }
        // Sokol may deliver one final callback after request_quit(). Mark the
        // run complete before serializing so the report is emitted exactly
        // once.
        self.benchmark_target = None;
        let elapsed = self
            .benchmark_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64());
        let mut sorted = self.benchmark_samples_ms.clone();
        sorted.sort_by(f64::total_cmp);
        let percentile = |fraction: f64| sorted[((sorted.len() - 1) as f64 * fraction).round() as usize];
        let dynamic = self
            .renderer
            .as_ref()
            .and_then(|renderer| renderer.dynamic.as_ref());
        let diagnostics = self
            .renderer
            .as_ref()
            .and_then(|renderer| renderer.readback_diagnostics().ok())
            .unwrap_or_default();
        let hash_publication = self
            .renderer
            .as_ref()
            .and_then(|renderer| renderer.readback_hash_publication_audit().ok())
            .unwrap_or_default();
        let gpu_stage_timing: BTreeMap<String, GpuStageTimingSummary> = self
            .renderer
            .as_ref()
            .map(|renderer| {
                renderer
                    .gpu_stage_samples_ms
                    .iter()
                    .map(|(name, samples)| (name.clone(), summarize_gpu_samples(samples)))
                    .collect()
            })
            .unwrap_or_default();
        let gpu_timestamp_sample_count = gpu_stage_timing
            .values()
            .map(|summary| summary.sample_count)
            .min()
            .unwrap_or(0);
        let gpu_stage_ms: BTreeMap<String, f64> = gpu_stage_timing
            .iter()
            .map(|(name, summary)| (name.clone(), summary.median_ms))
            .collect();
        let report = serde_json::json!({
            "backend": format!("{:?}", sg::query_backend()),
            "scene": self.renderer.as_ref().map(|renderer| renderer.scene.info.name),
            "width": dynamic.map(|resource| resource.width),
            "height": dynamic.map(|resource| resource.height),
            "resolution_scale": self.renderer.as_ref().map(|renderer| renderer.resolution_scale),
            "adaptive_resolution": self.renderer.as_ref().map(|renderer| renderer.adaptive_resolution),
            "triangles": self.renderer.as_ref().map(|renderer| renderer.scene.source_triangle_count),
            "frames": sorted.len(),
            "benchmark_warmup_frames": BENCHMARK_WARMUP_FRAMES,
            "elapsed_seconds": elapsed,
            "throughput_fps": sorted.len() as f64 / elapsed.max(1.0e-9),
            "median_callback_ms": percentile(0.5),
            "p95_callback_ms": percentile(0.95),
            "maximum_callback_ms": sorted[sorted.len() - 1],
            "swap_interval": 0,
            "gpu_timestamp_available": gpu_timestamp_sample_count > 0,
            "gpu_timestamp_sample_interval_frames": GPU_PROFILE_SAMPLE_INTERVAL,
            "gpu_timestamp_sample_count": gpu_timestamp_sample_count,
            "gpu_timestamp_sampling_scope": "measured_interval_after_warmup",
            "gpu_timestamp_first_measured_frame_index": (gpu_timestamp_sample_count > 0).then_some(0),
            "gpu_timestamp_last_measured_frame_index": (gpu_timestamp_sample_count > 0)
                .then_some((gpu_timestamp_sample_count - 1) * GPU_PROFILE_SAMPLE_INTERVAL),
            "gpu_stage_ms": gpu_stage_ms,
            "gpu_stage_timing": gpu_stage_timing,
            "probe_counts": diagnostics.probe_counts,
            "hash_overflows": diagnostics.hash_overflows,
            "key_hash_collisions": diagnostics.key_hash_collisions,
            "key_range_rejections": diagnostics.key_range_rejections,
            "bvh_stack_overflows": diagnostics.bvh_stack_overflows,
            "hash_publication": hash_publication,
        });
        if let Some(parent) = self.benchmark_output.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match serde_json::to_vec_pretty(&report)
            .map_err(|error| error.to_string())
            .and_then(|bytes| {
                std::fs::write(&self.benchmark_output, bytes).map_err(|error| error.to_string())
            }) {
            Ok(()) => eprintln!("benchmark: {}", self.benchmark_output.display()),
            Err(error) => self.fatal_error = Some(format!("benchmark write failed: {error}")),
        }
    }
}

extern "C" fn init(user_data: *mut ffi::c_void) {
    let app = unsafe { &mut *user_data.cast::<App>() };
    sg::setup(&sg::Desc {
        environment: sglue::environment(),
        logger: sg::Logger {
            func: Some(slog::slog_func),
            ..Default::default()
        },
        ..Default::default()
    });
    eprintln!("Split RC native backend: {:?}", sg::query_backend());
    app.load_scene(if app.stability_output.is_some() {
        app.stability_scene
    } else if app.cache_motion_output.is_some() {
        SceneId::Sponza
    } else if app.animation_response_output.is_some() {
        SceneId::Sponza
    } else if app.capture_dir.is_some() {
        SceneId::ALL[0]
    } else {
        SceneId::Sponza
    });
    if app.capture_dir.is_some() || app.stability_output.is_some() || app.cache_motion_output.is_some() {
        if let Some(renderer) = &mut app.renderer {
            renderer.animate = false;
            renderer.indirect_only = app.cache_motion_output.is_some();
        }
    }
    if let Some(error) = &app.fatal_error {
        eprintln!("Split RC initialization failed: {error}");
    }
}

const CACHE_MOTION_WARMUP_FRAMES: u32 = 48;
const CACHE_MOTION_TRANSLATION_FRAMES: u32 = 54;
const CACHE_MOTION_HOLD_FRAMES: u32 = 48;
const CACHE_MOTION_RECOVERY_FRAMES: u32 = 48;
const CACHE_MOTION_STEP: f32 = 0.12;
const CACHE_MOTION_ACCUMULATED_FRAME: u32 =
    CACHE_MOTION_WARMUP_FRAMES + CACHE_MOTION_TRANSLATION_FRAMES + CACHE_MOTION_HOLD_FRAMES - 1;
const CACHE_MOTION_RESET_FRAME: u32 = CACHE_MOTION_ACCUMULATED_FRAME + 1;
const CACHE_MOTION_RECOVERED_FRAME: u32 = CACHE_MOTION_RESET_FRAME + CACHE_MOTION_RECOVERY_FRAMES - 1;

extern "C" fn frame(user_data: *mut ffi::c_void) {
    let app = unsafe { &mut *user_data.cast::<App>() };
    if let Some(scene) = app.pending_scene.take() {
        app.fatal_error = None;
        app.load_scene(scene);
    }
    let callback_seconds = sapp::frame_duration().min(0.1);
    let delta = callback_seconds as f32;
    if app.cache_motion_output.is_some() {
        // Exact native replay of the WebGPU bug report: lower the camera in
        // Sponza, translate through the atrium, hold at the final pose, then
        // rebuild all temporal resources without moving the camera.
        let translated_steps = if app.cache_motion_frame < CACHE_MOTION_WARMUP_FRAMES {
            0
        } else {
            (app.cache_motion_frame - CACHE_MOTION_WARMUP_FRAMES + 1).min(CACHE_MOTION_TRANSLATION_FRAMES)
        };
        let translation = Vec3::X * (translated_steps as f32 * CACHE_MOTION_STEP);
        let target = Vec3::new(5.0, 1.75, -0.5) + translation;
        let position = Vec3::new(-9.274_269, 2.607_485, -0.5) + translation;
        app.camera.reset(position, target);
        if app.cache_motion_frame == CACHE_MOTION_RESET_FRAME {
            if let Some(renderer) = &mut app.renderer {
                if let Err(error) = renderer.reset_temporal_history() {
                    app.fatal_error = Some(format!("cache-motion history reset failed: {error}"));
                }
            }
        }
    } else if app.stability_output.is_some() {
        if let Some(renderer) = &app.renderer {
            let phase_index = app.stability_frame % 120;
            let angle = phase_index as f32 / 120.0 * std::f32::consts::TAU;
            let radius = (renderer.scene.settings.base_spacing * 0.5).max(0.2);
            // Smooth closed loop with exact zero offset at phase 0. The old
            // phase-0 special case teleported by one radius between frames
            // 0 and 1, invalidating history at the comparison pose.
            let offset = Vec3::new(
                (angle.cos() - 1.0) * radius,
                (angle * 2.0).sin() * radius * 0.22,
                angle.sin() * radius,
            );
            app.camera.reset(
                renderer.scene.settings.camera.position + offset,
                renderer.scene.settings.camera.target,
            );
        }
    } else {
        app.camera.update(&app.keys, delta);
    }
    let collect_gpu_timestamp = app.benchmark_target.is_some()
        && app.benchmark_warmup == 0
        && app.benchmark_samples_ms.len() % GPU_PROFILE_SAMPLE_INTERVAL == 0;
    if let Some(renderer) = &mut app.renderer {
        if let Err(error) =
            renderer.render(&app.camera, Duration::from_secs_f32(delta), collect_gpu_timestamp)
        {
            app.fatal_error = Some(error.to_string());
        }
    }
    if let Some(output) = app.cache_motion_output.clone() {
        if app.fatal_error.is_none()
            && matches!(
                app.cache_motion_frame,
                CACHE_MOTION_ACCUMULATED_FRAME | CACHE_MOTION_RECOVERED_FRAME
            )
        {
            let snapshot = app
                .renderer
                .as_ref()
                .ok_or_else(|| "renderer unavailable".to_owned())
                .and_then(Renderer::read_pixels);
            match snapshot {
                Ok((width, height, pixels)) if app.cache_motion_frame == CACHE_MOTION_ACCUMULATED_FRAME => {
                    let stem = output
                        .file_stem()
                        .and_then(|value| value.to_str())
                        .unwrap_or("cache-motion");
                    let accumulated_path = output.with_file_name(format!("{stem}-accumulated.png"));
                    let diagnostics = app
                        .renderer
                        .as_ref()
                        .and_then(|renderer| renderer.readback_diagnostics().ok())
                        .unwrap_or_default();
                    let publication = app
                        .renderer
                        .as_ref()
                        .and_then(|renderer| renderer.readback_hash_publication_audit().ok())
                        .unwrap_or_default();
                    if let Err(error) = crate::capture::save_rgba8(&pixels, width, height, &accumulated_path)
                    {
                        app.fatal_error = Some(format!("cache-motion accumulated save failed: {error}"));
                    } else {
                        app.cache_motion_accumulated = Some((width, height, pixels));
                        app.cache_motion_accumulated_diagnostics = Some(diagnostics);
                        app.cache_motion_accumulated_publication = Some(publication);
                    }
                }
                Ok((width, height, pixels)) => {
                    let comparison = app
                        .cache_motion_accumulated
                        .as_ref()
                        .ok_or_else(|| "cache-motion accumulated frame is unavailable".to_owned())
                        .and_then(|(old_width, old_height, accumulated)| {
                            if *old_width != width || *old_height != height {
                                return Err("cache-motion capture dimensions changed".to_owned());
                            }
                            crate::capture::image_delta(accumulated, &pixels)
                        });
                    match comparison {
                        Ok(delta_report) => {
                            let accumulated_diagnostics =
                                app.cache_motion_accumulated_diagnostics.unwrap_or_default();
                            let accumulated_publication = app
                                .cache_motion_accumulated_publication
                                .clone()
                                .unwrap_or_default();
                            let recovered_diagnostics = app
                                .renderer
                                .as_ref()
                                .and_then(|renderer| renderer.readback_diagnostics().ok())
                                .unwrap_or_default();
                            let recovered_publication = app
                                .renderer
                                .as_ref()
                                .and_then(|renderer| renderer.readback_hash_publication_audit().ok())
                                .unwrap_or_default();
                            let diagnostics_clean = [accumulated_diagnostics, recovered_diagnostics]
                                .into_iter()
                                .all(|diagnostics| {
                                    diagnostics.hash_overflows == 0
                                        && diagnostics.key_hash_collisions == 0
                                        && diagnostics.key_range_rejections == 0
                                        && diagnostics.bvh_stack_overflows == 0
                                });
                            let publication_clean = [&accumulated_publication, &recovered_publication]
                                .into_iter()
                                .all(|publication| {
                                    publication.unpublished_slots == 0
                                        && publication.reserved_slots == 0
                                        && publication.duplicate_slots == 0
                                        && publication.repeated_exact_keys == 0
                                        && publication.occupied_slots == publication.canonical_slots
                                });
                            let sparse_population_matched =
                                accumulated_diagnostics.probe_counts == recovered_diagnostics.probe_counts;
                            let image_stable = delta_report.mean_absolute_u8 <= 1.0
                                && delta_report.root_mean_square_u8 <= 4.0
                                && delta_report.p95_absolute_u8 <= 4
                                && delta_report.p99_absolute_u8 <= 12
                                && delta_report.p999_absolute_u8 <= 48;
                            let passed = diagnostics_clean
                                && publication_clean
                                && sparse_population_matched
                                && image_stable;
                            let stem = output
                                .file_stem()
                                .and_then(|value| value.to_str())
                                .unwrap_or("cache-motion");
                            let recovered_path = output.with_file_name(format!("{stem}-recovered.png"));
                            let heatmap_path = output.with_file_name(format!("{stem}-delta-heatmap.png"));
                            let report = serde_json::json!({
                                "scene": "Sponza atrium (paper scene)",
                                "width": width,
                                "height": height,
                                "camera_path": {
                                    "warmup_frames": CACHE_MOTION_WARMUP_FRAMES,
                                    "translation_frames": CACHE_MOTION_TRANSLATION_FRAMES,
                                    "translation_per_frame": CACHE_MOTION_STEP,
                                    "total_translation": CACHE_MOTION_STEP
                                        * CACHE_MOTION_TRANSLATION_FRAMES as f32,
                                    "hold_frames": CACHE_MOTION_HOLD_FRAMES,
                                    "recovery_frames_after_history_reset":
                                        CACHE_MOTION_RECOVERY_FRAMES,
                                    "accumulated_frame": CACHE_MOTION_ACCUMULATED_FRAME,
                                    "history_reset_frame": CACHE_MOTION_RESET_FRAME,
                                    "recovered_frame": CACHE_MOTION_RECOVERED_FRAME,
                                    "start_target": [5.0, 1.75, -0.5],
                                    "start_position": [-9.274269, 2.607485, -0.5],
                                },
                                "delta": delta_report,
                                "accumulated_probe_counts":
                                    accumulated_diagnostics.probe_counts,
                                "recovered_probe_counts":
                                    recovered_diagnostics.probe_counts,
                                "accumulated_diagnostics": {
                                    "hash_overflows": accumulated_diagnostics.hash_overflows,
                                    "key_hash_collisions":
                                        accumulated_diagnostics.key_hash_collisions,
                                    "key_range_rejections":
                                        accumulated_diagnostics.key_range_rejections,
                                    "bvh_stack_overflows":
                                        accumulated_diagnostics.bvh_stack_overflows,
                                },
                                "recovered_diagnostics": {
                                    "hash_overflows": recovered_diagnostics.hash_overflows,
                                    "key_hash_collisions":
                                        recovered_diagnostics.key_hash_collisions,
                                    "key_range_rejections":
                                        recovered_diagnostics.key_range_rejections,
                                    "bvh_stack_overflows":
                                        recovered_diagnostics.bvh_stack_overflows,
                                },
                                "accumulated_hash_publication": accumulated_publication,
                                "recovered_hash_publication": recovered_publication,
                                "gate": {
                                    "passes": passed,
                                    "diagnostics_clean": diagnostics_clean,
                                    "publication_clean": publication_clean,
                                    "sparse_population_matched": sparse_population_matched,
                                    "image_stable": image_stable,
                                    "maximum_mean_absolute_u8": 1.0,
                                    "maximum_root_mean_square_u8": 4.0,
                                    "maximum_p95_absolute_u8": 4,
                                    "maximum_p99_absolute_u8": 12,
                                    "maximum_p999_absolute_u8": 48,
                                },
                            });
                            let write_result =
                                crate::capture::save_rgba8(&pixels, width, height, &recovered_path)
                                    .and_then(|()| {
                                        let accumulated = &app
                                            .cache_motion_accumulated
                                            .as_ref()
                                            .ok_or_else(|| {
                                                "cache-motion accumulated frame is unavailable".to_owned()
                                            })?
                                            .2;
                                        crate::capture::save_delta_heatmap(
                                            accumulated,
                                            &pixels,
                                            width,
                                            height,
                                            &heatmap_path,
                                        )
                                    })
                                    .and_then(|()| {
                                        if let Some(parent) = output.parent() {
                                            std::fs::create_dir_all(parent)
                                                .map_err(|error| error.to_string())?;
                                        }
                                        serde_json::to_vec_pretty(&report)
                                            .map_err(|error| error.to_string())
                                            .and_then(|bytes| {
                                                std::fs::write(&output, bytes)
                                                    .map_err(|error| error.to_string())
                                            })
                                    });
                            if let Err(error) = write_result {
                                app.fatal_error = Some(format!("cache-motion report write failed: {error}"));
                            } else if !passed {
                                app.fatal_error = Some("cache-motion recovery gate failed".to_owned());
                            } else {
                                eprintln!("cache motion recovery: {}", output.display());
                                sapp::request_quit();
                            }
                        }
                        Err(error) => {
                            app.fatal_error = Some(format!("cache-motion comparison failed: {error}"));
                        }
                    }
                }
                Err(error) => {
                    app.fatal_error = Some(format!("cache-motion readback failed: {error}"));
                }
            }
        }
        app.cache_motion_frame += 1;
    }
    if let Some(output) = app.stability_output.clone() {
        let reference_frame = app.stability_reference_frame;
        let comparison_frame = reference_frame + 120;
        if app.fatal_error.is_none()
            && matches!(app.stability_frame, frame if frame == reference_frame || frame == comparison_frame)
        {
            let snapshot = app
                .renderer
                .as_ref()
                .ok_or_else(|| "renderer unavailable".to_owned())
                .and_then(Renderer::read_pixels);
            match snapshot {
                Ok((width, height, pixels)) if app.stability_frame == reference_frame => {
                    let stem = output
                        .file_stem()
                        .and_then(|value| value.to_str())
                        .unwrap_or("stability");
                    let reference_path = output.with_file_name(format!("{stem}-reference.png"));
                    if let Err(error) = crate::capture::save_rgba8(&pixels, width, height, &reference_path) {
                        app.fatal_error = Some(format!("stability reference save failed: {error}"));
                    } else {
                        app.stability_reference = Some(pixels);
                    }
                }
                Ok((width, height, pixels)) => {
                    let result = app
                        .stability_reference
                        .as_ref()
                        .ok_or_else(|| "stability reference is unavailable".to_owned())
                        .and_then(|reference| crate::capture::image_delta(reference, &pixels));
                    match result {
                        Ok(delta_report) => {
                            // This gate now measures the paper's raw world-space
                            // field directly. The removed recursive display
                            // history made the old mean/RMS-only limits
                            // inappropriate: rare silhouette changes dominated
                            // them while 95% of channels moved by at most one
                            // code value. Bound the full distribution instead,
                            // including the 99.9th percentile and maximum, so
                            // block corruption cannot hide in the tail.
                            let passes_visual_stability_gate = delta_report.mean_absolute_u8 <= 0.30
                                && delta_report.root_mean_square_u8 <= 1.0
                                && delta_report.p95_absolute_u8 <= 1
                                && delta_report.p99_absolute_u8 <= 3
                                && delta_report.p999_absolute_u8 <= 12
                                && delta_report.maximum_absolute_u8 <= 48;
                            let stem = output
                                .file_stem()
                                .and_then(|value| value.to_str())
                                .unwrap_or("stability");
                            let comparison_path = output.with_file_name(format!("{stem}-comparison.png"));
                            let heatmap_path = output.with_file_name(format!("{stem}-delta-heatmap.png"));
                            let diagnostics = app
                                .renderer
                                .as_ref()
                                .and_then(|renderer| renderer.readback_diagnostics().ok())
                                .unwrap_or_default();
                            let hash_publication = app
                                .renderer
                                .as_ref()
                                .and_then(|renderer| renderer.readback_hash_publication_audit().ok())
                                .unwrap_or_default();
                            let diagnostics_clean = diagnostics.hash_overflows == 0
                                && diagnostics.key_hash_collisions == 0
                                && diagnostics.key_range_rejections == 0
                                && diagnostics.bvh_stack_overflows == 0;
                            let publication_clean = hash_publication.duplicate_slots == 0
                                && hash_publication.reserved_slots == 0
                                && hash_publication.unpublished_slots == 0
                                && hash_publication.repeated_exact_keys == 0
                                && hash_publication.occupied_slots == hash_publication.canonical_slots;
                            let passes_stability_gate =
                                passes_visual_stability_gate && diagnostics_clean && publication_clean;
                            let (scene_name, scene_short, camera_path_radius) = app
                                .renderer
                                .as_ref()
                                .map(|renderer| {
                                    (
                                        renderer.scene.info.name,
                                        renderer.scene.info.short,
                                        (renderer.scene.settings.base_spacing * 0.5).max(0.2),
                                    )
                                })
                                .unwrap_or(("unknown", "unknown", 0.0));
                            let report = serde_json::json!({
                                "scene": scene_name,
                                "scene_short": scene_short,
                                "width": width,
                                "height": height,
                                "camera_path_period_frames": 120,
                                "camera_path_radius": camera_path_radius,
                                "reference_frame": reference_frame,
                                "comparison_frame": comparison_frame,
                                "animation_frozen": true,
                                "delta": delta_report,
                                "gate": {
                                    "passes": passes_stability_gate,
                                    "visual_stability": passes_visual_stability_gate,
                                    "diagnostics_clean": diagnostics_clean,
                                    "publication_clean": publication_clean,
                                    "maximum_mean_absolute_u8": 0.30,
                                    "maximum_root_mean_square_u8": 1.0,
                                    "maximum_p95_absolute_u8": 1,
                                    "maximum_p99_absolute_u8": 3,
                                    "maximum_p999_absolute_u8": 12,
                                    "maximum_absolute_u8": 48,
                                },
                                "diagnostics": {
                                    "hash_overflows": diagnostics.hash_overflows,
                                    "key_hash_collisions": diagnostics.key_hash_collisions,
                                    "key_range_rejections": diagnostics.key_range_rejections,
                                    "bvh_stack_overflows": diagnostics.bvh_stack_overflows,
                                },
                                "hash_publication": hash_publication,
                            });
                            let write_result =
                                crate::capture::save_rgba8(&pixels, width, height, &comparison_path)
                                    .and_then(|()| {
                                        crate::capture::save_delta_heatmap(
                                            app.stability_reference.as_ref().ok_or_else(|| {
                                                "stability reference is unavailable".to_owned()
                                            })?,
                                            &pixels,
                                            width,
                                            height,
                                            &heatmap_path,
                                        )
                                    })
                                    .and_then(|()| {
                                        if let Some(parent) = output.parent() {
                                            std::fs::create_dir_all(parent)
                                                .map_err(|error| error.to_string())?;
                                        }
                                        serde_json::to_vec_pretty(&report)
                                            .map_err(|error| error.to_string())
                                            .and_then(|bytes| {
                                                std::fs::write(&output, bytes)
                                                    .map_err(|error| error.to_string())
                                            })
                                    });
                            if let Err(error) = write_result {
                                app.fatal_error = Some(format!("stability report write failed: {error}"));
                            } else if !passes_stability_gate {
                                app.fatal_error =
                                    Some("camera-loop temporal stability gate failed".to_owned());
                            } else {
                                eprintln!("stability: {}", output.display());
                                sapp::request_quit();
                            }
                        }
                        Err(error) => {
                            app.fatal_error = Some(format!("stability comparison failed: {error}"));
                        }
                    }
                }
                Err(error) => app.fatal_error = Some(format!("stability readback failed: {error}")),
            }
        }
        app.stability_frame += 1;
    }
    if let Some(output) = app.animation_response_output.clone() {
        const REFERENCE_FRAME: u32 = 120;
        const COMPARISON_FRAME: u32 = 360;
        if app.fatal_error.is_none()
            && matches!(app.animation_response_frame, REFERENCE_FRAME | COMPARISON_FRAME)
        {
            let snapshot = app
                .renderer
                .as_ref()
                .ok_or_else(|| "renderer unavailable".to_owned())
                .and_then(Renderer::read_pixels);
            match snapshot {
                Ok((width, height, pixels)) if app.animation_response_frame == REFERENCE_FRAME => {
                    let stem = output
                        .file_stem()
                        .and_then(|value| value.to_str())
                        .unwrap_or("animation-response");
                    let reference_path = output.with_file_name(format!("{stem}-reference.png"));
                    if let Err(error) = crate::capture::save_rgba8(&pixels, width, height, &reference_path) {
                        app.fatal_error = Some(format!("animation response reference save failed: {error}"));
                    } else {
                        app.animation_response_reference = Some(pixels);
                        app.animation_response_reference_time =
                            app.renderer.as_ref().map(|renderer| renderer.animation_time);
                    }
                }
                Ok((width, height, pixels)) => {
                    let result = app
                        .animation_response_reference
                        .as_ref()
                        .ok_or_else(|| "animation response reference is unavailable".to_owned())
                        .and_then(|reference| crate::capture::image_delta(reference, &pixels));
                    match result {
                        Ok(delta_report) => {
                            let diagnostics = app
                                .renderer
                                .as_ref()
                                .and_then(|renderer| renderer.readback_diagnostics().ok())
                                .unwrap_or_default();
                            let hash_publication = app
                                .renderer
                                .as_ref()
                                .and_then(|renderer| renderer.readback_hash_publication_audit().ok())
                                .unwrap_or_default();
                            let response_is_visible = delta_report.mean_absolute_u8 >= 0.10
                                && delta_report.changed_channel_fraction >= 0.01;
                            let diagnostics_clean = diagnostics.hash_overflows == 0
                                && diagnostics.key_hash_collisions == 0
                                && diagnostics.key_range_rejections == 0
                                && diagnostics.bvh_stack_overflows == 0
                                && hash_publication.duplicate_slots == 0
                                && hash_publication.reserved_slots == 0
                                && hash_publication.unpublished_slots == 0
                                && hash_publication.occupied_slots == hash_publication.canonical_slots;
                            let comparison_time = app
                                .renderer
                                .as_ref()
                                .map_or(0.0, |renderer| renderer.animation_time);
                            let reference_time = app.animation_response_reference_time.unwrap_or(0.0);
                            let settings = app.renderer.as_ref().map(|renderer| renderer.scene.settings);
                            let reference_lights =
                                settings.map(|settings| settings.animated_lights(reference_time));
                            let comparison_lights =
                                settings.map(|settings| settings.animated_lights(comparison_time));
                            let stem = output
                                .file_stem()
                                .and_then(|value| value.to_str())
                                .unwrap_or("animation-response");
                            let comparison_path = output.with_file_name(format!("{stem}-comparison.png"));
                            let heatmap_path = output.with_file_name(format!("{stem}-delta-heatmap.png"));
                            let report = serde_json::json!({
                                "scene": "Sponza atrium (paper scene)",
                                "width": width,
                                "height": height,
                                "camera_fixed": true,
                                "animation_enabled": app.renderer.as_ref().is_some_and(|renderer| renderer.animate),
                                "reference_frame": REFERENCE_FRAME,
                                "comparison_frame": COMPARISON_FRAME,
                                "reference_animation_seconds": reference_time,
                                "comparison_animation_seconds": comparison_time,
                                "reference_sun_direction": reference_lights.map(|lights| lights.sun_direction.to_array()),
                                "comparison_sun_direction": comparison_lights.map(|lights| lights.sun_direction.to_array()),
                                "reference_point_position": reference_lights.map(|lights| lights.point_position.to_array()),
                                "comparison_point_position": comparison_lights.map(|lights| lights.point_position.to_array()),
                                "delta": delta_report,
                                "gate": {
                                    "passes": response_is_visible && diagnostics_clean,
                                    "minimum_mean_absolute_u8": 0.10,
                                    "minimum_changed_channel_fraction": 0.01,
                                    "visible_response": response_is_visible,
                                    "diagnostics_clean": diagnostics_clean,
                                },
                                "diagnostics": {
                                    "hash_overflows": diagnostics.hash_overflows,
                                    "key_hash_collisions": diagnostics.key_hash_collisions,
                                    "key_range_rejections": diagnostics.key_range_rejections,
                                    "bvh_stack_overflows": diagnostics.bvh_stack_overflows,
                                },
                                "hash_publication": hash_publication,
                            });
                            let write_result =
                                crate::capture::save_rgba8(&pixels, width, height, &comparison_path)
                                    .and_then(|()| {
                                        crate::capture::save_delta_heatmap(
                                            app.animation_response_reference.as_ref().ok_or_else(|| {
                                                "animation response reference is unavailable".to_owned()
                                            })?,
                                            &pixels,
                                            width,
                                            height,
                                            &heatmap_path,
                                        )
                                    })
                                    .and_then(|()| {
                                        if let Some(parent) = output.parent() {
                                            std::fs::create_dir_all(parent)
                                                .map_err(|error| error.to_string())?;
                                        }
                                        serde_json::to_vec_pretty(&report)
                                            .map_err(|error| error.to_string())
                                            .and_then(|bytes| {
                                                std::fs::write(&output, bytes)
                                                    .map_err(|error| error.to_string())
                                            })
                                    });
                            if let Err(error) = write_result {
                                app.fatal_error = Some(format!("animation response write failed: {error}"));
                            } else if !(response_is_visible && diagnostics_clean) {
                                app.fatal_error = Some("animated-light response gate failed".to_owned());
                            } else {
                                eprintln!("animation response: {}", output.display());
                                sapp::request_quit();
                            }
                        }
                        Err(error) => {
                            app.fatal_error = Some(format!("animation response comparison failed: {error}"));
                        }
                    }
                }
                Err(error) => {
                    app.fatal_error = Some(format!("animation response readback failed: {error}"));
                }
            }
        }
        app.animation_response_frame += 1;
    }
    if app.capture_dir.is_some() && app.fatal_error.is_none() {
        app.capture_frames += 1;
        if app.capture_frames >= 96 {
            let scene = SceneId::ALL[app.capture_scene];
            let destination = app.capture_dir.as_ref().unwrap().join(format!(
                "{:02}-{}.png",
                app.capture_scene + 1,
                scene.info().short.to_ascii_lowercase()
            ));
            let capture_result = app
                .renderer
                .as_ref()
                .ok_or_else(|| "renderer unavailable".to_owned())
                .and_then(|renderer| renderer.capture(&destination));
            match capture_result {
                Ok(()) => {
                    eprintln!("capture: {}", destination.display());
                    app.capture_paths.push(destination);
                    app.capture_scene += 1;
                    if app.capture_scene < SceneId::ALL.len() {
                        app.pending_scene = Some(SceneId::ALL[app.capture_scene]);
                    } else {
                        let sheet = app.capture_dir.as_ref().unwrap().join("contact-sheet.png");
                        if let Err(error) = crate::capture::write_contact_sheet(&app.capture_paths, &sheet) {
                            app.fatal_error = Some(format!("contact sheet failed: {error}"));
                        } else {
                            eprintln!("contact sheet: {}", sheet.display());
                            sapp::request_quit();
                        }
                    }
                }
                Err(error) => app.fatal_error = Some(format!("capture failed: {error}")),
            }
            app.capture_frames = 0;
        }
    }
    if let Some(target) = app.benchmark_target {
        if app.benchmark_warmup > 0 {
            app.benchmark_warmup -= 1;
        } else {
            if app.benchmark_started.is_none() {
                app.benchmark_started = Some(Instant::now());
            }
            app.benchmark_samples_ms.push(callback_seconds * 1_000.0);
            if app.benchmark_samples_ms.len() >= target as usize {
                app.finish_benchmark();
                sapp::request_quit();
            }
        }
    }
    if app.renderer.is_none() || app.fatal_error.is_some() {
        let mut action = sg::PassAction::new();
        action.colors[0] = sg::ColorAttachmentAction {
            load_action: sg::LoadAction::Clear,
            clear_value: sg::Color {
                r: 0.35,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
            ..Default::default()
        };
        sg::begin_pass(&sg::Pass {
            action,
            swapchain: sglue::swapchain(),
            ..Default::default()
        });
        sg::end_pass();
        sg::commit();
        if app.capture_dir.is_some()
            || app.stability_output.is_some()
            || app.cache_motion_output.is_some()
            || app.animation_response_output.is_some()
            || app.benchmark_target.is_some()
        {
            if let Some(error) = &app.fatal_error {
                eprintln!("automated run failed: {error}");
            }
            sapp::request_quit();
        }
    }
}

extern "C" fn event(event: *const sapp::Event, user_data: *mut ffi::c_void) {
    let app = unsafe { &mut *user_data.cast::<App>() };
    let event = unsafe { &*event };
    let key_index = event.key_code as usize;
    match event._type {
        sapp::EventType::KeyDown => {
            if key_index < app.keys.len() {
                app.keys[key_index] = true;
            }
            if !event.key_repeat {
                app.pending_scene = match event.key_code {
                    sapp::Keycode::Num1 => Some(SceneId::Laboratory),
                    sapp::Keycode::Num2 => Some(SceneId::Sponza),
                    sapp::Keycode::Num3 => Some(SceneId::Canyon),
                    sapp::Keycode::Num4 => Some(SceneId::Forest),
                    sapp::Keycode::Num5 => Some(SceneId::Atrium),
                    sapp::Keycode::Num6 => Some(SceneId::Pipes),
                    sapp::Keycode::Num7 => Some(SceneId::Temple),
                    sapp::Keycode::Num8 => Some(SceneId::Orbit),
                    sapp::Keycode::Num9 => Some(SceneId::Market),
                    sapp::Keycode::Num0 => Some(SceneId::Stress),
                    sapp::Keycode::C => Some(SceneId::Cornell),
                    sapp::Keycode::H => Some(SceneId::Heightmap),
                    _ => None,
                };
                if event.key_code == sapp::Keycode::Space {
                    if let Some(renderer) = &mut app.renderer {
                        renderer.animate = !renderer.animate;
                        if let Err(error) = renderer.reset_temporal_history() {
                            app.fatal_error = Some(format!("lighting history reset failed: {error}"));
                        }
                    }
                }
                if event.key_code == sapp::Keycode::M {
                    if let Some(renderer) = &mut app.renderer {
                        renderer.c_minus_one = !renderer.c_minus_one;
                        if let Err(error) = renderer.reset_temporal_history() {
                            app.fatal_error = Some(format!("C(-1) history reset failed: {error}"));
                        }
                        eprintln!("directional C(-1): {}", renderer.c_minus_one);
                    }
                }
            }
        }
        sapp::EventType::KeyUp => {
            if key_index < app.keys.len() {
                app.keys[key_index] = false;
            }
        }
        sapp::EventType::MouseDown if event.mouse_button == sapp::Mousebutton::Right => {
            sapp::lock_mouse(true);
        }
        sapp::EventType::MouseUp if event.mouse_button == sapp::Mousebutton::Right => {
            sapp::lock_mouse(false);
        }
        sapp::EventType::MouseMove if sapp::mouse_locked() => {
            app.camera.yaw += event.mouse_dx * 0.0025;
            app.camera.pitch = (app.camera.pitch - event.mouse_dy * 0.0025).clamp(-1.54, 1.54);
        }
        _ => {}
    }
}

extern "C" fn cleanup(user_data: *mut ffi::c_void) {
    let mut app = unsafe { Box::from_raw(user_data.cast::<App>()) };
    if let Some(renderer) = app.renderer.take() {
        renderer.destroy();
    }
    sg::shutdown();
}

pub fn run() {
    let app = Box::new(App::new());
    let unlocked = app.benchmark_target.is_some();
    let user_data = Box::into_raw(app).cast::<ffi::c_void>();
    sapp::run(&sapp::Desc {
        init_userdata_cb: Some(init),
        frame_userdata_cb: Some(frame),
        cleanup_userdata_cb: Some(cleanup),
        event_userdata_cb: Some(event),
        user_data,
        width: INITIAL_WIDTH,
        height: INITIAL_HEIGHT,
        sample_count: 1,
        swap_interval: if unlocked { 0 } else { 1 },
        high_dpi: true,
        window_title: c"Split Radiance Cascades - Rust + Sokol GPU".as_ptr(),
        icon: sapp::IconDesc {
            sokol_default: true,
            ..Default::default()
        },
        logger: sapp::Logger {
            func: Some(slog::slog_func),
            ..Default::default()
        },
        ..Default::default()
    });
}

fn asset_root() -> PathBuf {
    if let Some(path) = env::var_os("SPLIT_RC_ASSETS") {
        return PathBuf::from(path);
    }
    for candidate in [
        Path::new("../public/models"),
        Path::new("public/models"),
        Path::new("assets"),
    ] {
        if candidate.join("sponza.rcb").is_file() {
            return candidate.to_owned();
        }
    }
    PathBuf::from("assets")
}

fn key(keys: &[bool; 512], code: sapp::Keycode) -> bool {
    keys.get(code as usize).copied().unwrap_or(false)
}

fn storage_buffer(size: usize, name: &'static str) -> Result<sg::Buffer, GpuError> {
    let buffer = sg::make_buffer(&sg::BufferDesc {
        size,
        usage: sg::BufferUsage {
            storage_buffer: true,
            ..Default::default()
        },
        ..Default::default()
    });
    if sg::query_buffer_state(buffer) != sg::ResourceState::Valid {
        return Err(GpuError::Resource(name));
    }
    Ok(buffer)
}

fn immutable_storage_buffer<T>(values: &[T], name: &'static str) -> Result<sg::Buffer, GpuError> {
    let buffer = sg::make_buffer(&sg::BufferDesc {
        usage: sg::BufferUsage {
            storage_buffer: true,
            immutable: true,
            ..Default::default()
        },
        data: sg::slice_as_range(values),
        ..Default::default()
    });
    if sg::query_buffer_state(buffer) != sg::ResourceState::Valid {
        return Err(GpuError::Resource(name));
    }
    Ok(buffer)
}

fn storage_view(buffer: sg::Buffer, name: &'static str) -> Result<sg::View, GpuError> {
    let view = sg::make_view(&sg::ViewDesc {
        storage_buffer: sg::BufferViewDesc {
            buffer,
            ..Default::default()
        },
        ..Default::default()
    });
    valid_view(view, name)?;
    Ok(view)
}

fn valid_shader(shader: sg::Shader, name: &'static str) -> Result<(), GpuError> {
    if sg::query_shader_state(shader) == sg::ResourceState::Valid {
        Ok(())
    } else {
        Err(GpuError::Resource(name))
    }
}

fn valid_pipeline(pipeline: sg::Pipeline, name: &'static str) -> Result<(), GpuError> {
    if sg::query_pipeline_state(pipeline) == sg::ResourceState::Valid {
        Ok(())
    } else {
        Err(GpuError::Resource(name))
    }
}

fn valid_view(view: sg::View, name: &'static str) -> Result<(), GpuError> {
    if sg::query_view_state(view) == sg::ResourceState::Valid {
        Ok(())
    } else {
        Err(GpuError::Resource(name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_layouts_do_not_overlap() {
        let state = StateLayout::new(1280, 720);
        assert_eq!(RAY_STRIDE, 1);
        assert_eq!(state.sample_count, 1280 * 720);
        assert_eq!(state.ray_slot_count, state.sample_count * 3);
        assert_eq!(
            state.hit_frame_offset(1) - state.hit_frame_offset(0),
            state.sample_count * 8
        );
        assert!(state.meta_offset >= state.hash_offset + state.hash_frame_stride * 2);
        assert!(state.counters_offset >= state.meta_offset + state.meta_frame_stride * 2);
        assert!(state.blocks_offset > state.ray_map_offset);
        assert_eq!(
            state.total_words,
            state.blocks_offset + PROBE_CAPACITY[0] * state.block_count
        );
        let radiance = RadianceLayout::new();
        assert!(radiance.cone_offset >= ACCUM_FRAME_STRIDE * 2);
        assert!(radiance.irradiance_offset >= radiance.cone_offset + radiance.cone_frame_stride * 2);
        assert!(radiance.total_words > radiance.irradiance_offset);
        assert!(state.total_words < i32::MAX as u32);
        assert!(radiance.total_words < i32::MAX as u32);
    }

    #[test]
    fn presentation_is_current_world_space_split_rc_without_screen_history() {
        let source = include_str!("../shaders/split_rc.glsl");
        assert!(!source.contains("presentation_history_base"));
        assert!(!source.contains("display_history"));
        assert!(!source.contains("previous_indirect"));
        assert!(source.contains("feature_flags.w != 0u ? vec3(0.0) : surf.direct_emissive.xyz"));
        assert!(source.contains("imageStore(output_image, ivec2(pixel), vec4(color, 1.0))"));
    }

    #[test]
    fn cache_motion_gate_replays_reported_translation_and_clean_rebuild() {
        assert_eq!(CACHE_MOTION_WARMUP_FRAMES, 48);
        assert_eq!(CACHE_MOTION_TRANSLATION_FRAMES, 54);
        assert_eq!(CACHE_MOTION_HOLD_FRAMES, 48);
        assert_eq!(CACHE_MOTION_RECOVERY_FRAMES, 48);
        assert!((CACHE_MOTION_STEP * CACHE_MOTION_TRANSLATION_FRAMES as f32 - 6.48).abs() < 1.0e-6);
        assert_eq!(CACHE_MOTION_ACCUMULATED_FRAME, 149);
        assert_eq!(CACHE_MOTION_RESET_FRAME, 150);
        assert_eq!(CACHE_MOTION_RECOVERED_FRAME, 197);
    }

    #[test]
    fn larger_rank_blocks_preserve_exact_global_owner_order() {
        let owners: Vec<Option<usize>> = (0..4_097)
            .map(|slot| (slot % 11 != 0).then_some((slot * 17 + slot / 13) % 23))
            .collect();
        let rank_for = |slot: usize, block_size: usize| {
            let probe = owners[slot].unwrap();
            let block_start = slot / block_size * block_size;
            let previous_blocks = owners[..block_start]
                .iter()
                .filter(|owner| **owner == Some(probe))
                .count();
            previous_blocks
                + owners[block_start..slot]
                    .iter()
                    .filter(|owner| **owner == Some(probe))
                    .count()
        };
        assert_eq!(RANK_BLOCK_SIZE, 512);
        for (slot, owner) in owners.iter().enumerate() {
            if owner.is_some() {
                let expected = rank_for(slot, 256);
                assert_eq!(rank_for(slot, RANK_BLOCK_SIZE as usize), expected);
                assert_eq!(
                    expected,
                    owners[..slot]
                        .iter()
                        .filter(|candidate| candidate == &owner)
                        .count()
                );
            }
        }
    }

    #[test]
    fn adaptive_resolution_recovers_under_sixty_hertz_vsync() {
        assert_eq!(adaptive_scale_step(0.8, 1.0 / 60.0, false), 0.8);
        assert!((adaptive_scale_step(0.8, 1.0 / 60.0, true) - 0.84).abs() < 1.0e-6);
        assert!((adaptive_scale_step(0.8, 1.0 / 120.0, false) - 0.84).abs() < 1.0e-6);
        assert!((adaptive_scale_step(0.8, 1.0 / 30.0, true) - 0.72).abs() < 1.0e-6);
        assert_eq!(adaptive_scale_step(1.0, 1.0 / 60.0, true), 1.0);
    }

    #[test]
    fn rough_specular_extension_is_explicit_opt_in() {
        assert!(!feature_value_enabled(None));
        assert!(!feature_value_enabled(Some("false")));
        assert!(feature_value_enabled(Some("1")));
        assert!(feature_value_enabled(Some("on")));
    }

    #[test]
    fn gpu_timestamp_summary_reports_repeated_sample_distribution() {
        let summary = summarize_gpu_samples(&[7.0, 1.0, 9.0, 5.0, 3.0]);
        assert_eq!(summary.sample_count, 5);
        assert_eq!(summary.minimum_ms, 1.0);
        assert_eq!(summary.median_ms, 5.0);
        assert_eq!(summary.p95_ms, 9.0);
        assert_eq!(summary.maximum_ms, 9.0);
    }
}

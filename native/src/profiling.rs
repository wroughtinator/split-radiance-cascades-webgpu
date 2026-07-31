//! Low-overhead frame-stage profiling and machine-readable performance gates.

use std::{
    collections::{BTreeMap, VecDeque},
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};

pub const HISTORY_FRAMES: usize = 240;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StageSummary {
    pub samples: usize,
    pub median_ms: f64,
    pub p95_ms: f64,
    pub maximum_ms: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProfileReport {
    pub frame: StageSummary,
    pub stages: BTreeMap<String, StageSummary>,
    pub gpu_timestamp_available: bool,
    pub triangle_count: usize,
    pub ray_count: u64,
    pub probe_counts: [u32; 4],
    pub hash_overflows: u32,
    pub key_hash_collisions: u32,
    pub key_range_rejections: u32,
    pub bvh_stack_overflows: u32,
}

#[derive(Debug, Default)]
pub struct FrameProfiler {
    frames: VecDeque<Duration>,
    stages: BTreeMap<&'static str, VecDeque<Duration>>,
    stage_start: Option<(&'static str, Instant)>,
    diagnostics: FrameDiagnostics,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FrameDiagnostics {
    pub triangle_count: usize,
    pub ray_count: u64,
    pub probe_counts: [u32; 4],
    pub hash_overflows: u32,
    pub key_hash_collisions: u32,
    pub key_range_rejections: u32,
    pub bvh_stack_overflows: u32,
}

impl FrameProfiler {
    pub fn begin_frame(&mut self) {
        self.stage_start = None;
    }

    pub fn begin_stage(&mut self, name: &'static str) {
        self.end_stage();
        self.stage_start = Some((name, Instant::now()));
    }

    pub fn end_stage(&mut self) {
        if let Some((name, started)) = self.stage_start.take() {
            push_bounded(self.stages.entry(name).or_default(), started.elapsed());
        }
    }

    pub fn end_frame(&mut self, presented_duration: Duration) {
        self.end_stage();
        push_bounded(&mut self.frames, presented_duration);
    }

    pub fn set_diagnostics(&mut self, diagnostics: FrameDiagnostics) {
        self.diagnostics = diagnostics;
    }

    #[must_use]
    pub fn report(&self) -> ProfileReport {
        ProfileReport {
            frame: summarize(&self.frames),
            stages: self
                .stages
                .iter()
                .map(|(name, samples)| ((*name).to_owned(), summarize(samples)))
                .collect(),
            gpu_timestamp_available: false,
            triangle_count: self.diagnostics.triangle_count,
            ray_count: self.diagnostics.ray_count,
            probe_counts: self.diagnostics.probe_counts,
            hash_overflows: self.diagnostics.hash_overflows,
            key_hash_collisions: self.diagnostics.key_hash_collisions,
            key_range_rejections: self.diagnostics.key_range_rejections,
            bvh_stack_overflows: self.diagnostics.bvh_stack_overflows,
        }
    }
}

fn push_bounded(samples: &mut VecDeque<Duration>, value: Duration) {
    if samples.len() == HISTORY_FRAMES {
        samples.pop_front();
    }
    samples.push_back(value);
}

fn summarize(samples: &VecDeque<Duration>) -> StageSummary {
    let mut values: Vec<f64> = samples
        .iter()
        .map(|duration| duration.as_secs_f64() * 1000.0)
        .collect();
    values.sort_by(f64::total_cmp);
    if values.is_empty() {
        return StageSummary::default();
    }
    let percentile = |p: f64| {
        let index = ((values.len() - 1) as f64 * p).round() as usize;
        values[index]
    };
    StageSummary {
        samples: values.len(),
        median_ms: percentile(0.5),
        p95_ms: percentile(0.95),
        maximum_ms: values[values.len() - 1],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_history_and_percentiles() {
        let mut samples = VecDeque::new();
        for millis in 0..300 {
            push_bounded(&mut samples, Duration::from_millis(millis));
        }
        assert_eq!(samples.len(), HISTORY_FRAMES);
        let report = summarize(&samples);
        assert_eq!(report.maximum_ms, 299.0);
        assert!(report.p95_ms >= 285.0);
    }
}

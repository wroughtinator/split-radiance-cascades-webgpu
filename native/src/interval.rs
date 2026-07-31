//! Radiance interval representation and Eq. 6 composition.

use glam::Vec3;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RadianceInterval {
    pub radiance: Vec3,
    pub transmittance: f32,
}

impl RadianceInterval {
    pub const TRANSPARENT: Self = Self {
        radiance: Vec3::ZERO,
        transmittance: 1.0,
    };

    pub const OPAQUE_BLACK: Self = Self {
        radiance: Vec3::ZERO,
        transmittance: 0.0,
    };

    #[must_use]
    pub const fn new(radiance: Vec3, transmittance: f32) -> Self {
        Self {
            radiance,
            transmittance,
        }
    }

    /// Compose `far` behind `self`: `J_near + beta_near * J_far`.
    #[must_use]
    pub fn compose(self, far: Self) -> Self {
        Self {
            radiance: self.radiance + self.transmittance * far.radiance,
            transmittance: self.transmittance * far.transmittance,
        }
    }

    #[must_use]
    pub fn temporal_blend(self, previous: Self, weight: f32) -> Self {
        let weight = weight.clamp(0.0, 1.0);
        Self {
            radiance: self.radiance.lerp(previous.radiance, weight),
            transmittance: self
                .transmittance
                .mul_add(1.0 - weight, previous.transmittance * weight),
        }
    }

    #[must_use]
    pub fn sanitized(self, max_radiance: f32) -> Self {
        let radiance = if self.radiance.is_finite() {
            self.radiance.clamp(Vec3::ZERO, Vec3::splat(max_radiance))
        } else {
            Vec3::ZERO
        };
        Self {
            radiance,
            transmittance: if self.transmittance.is_finite() {
                self.transmittance.clamp(0.0, 1.0)
            } else {
                1.0
            },
        }
    }
}

/// Temporal resolve with explicit validity. Missing current and history
/// intervals stay missing; they are not promoted to transparent transport.
#[must_use]
pub fn temporal_resolve(
    current: Option<RadianceInterval>,
    previous: Option<RadianceInterval>,
    history_weight: f32,
) -> Option<RadianceInterval> {
    match (current, previous) {
        (Some(current), Some(previous)) => Some(current.temporal_blend(previous, history_weight)),
        (Some(current), None) => Some(current),
        (None, Some(previous)) => Some(previous),
        (None, None) => None,
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct IntervalAccumulator {
    radiance_sum: Vec3,
    transmittance_sum: f32,
    count: u32,
}

impl IntervalAccumulator {
    pub fn add(&mut self, interval: RadianceInterval) {
        self.radiance_sum += interval.radiance;
        self.transmittance_sum += interval.transmittance;
        self.count += 1;
    }

    #[must_use]
    pub fn resolve(self) -> Option<RadianceInterval> {
        (self.count > 0).then(|| {
            let reciprocal = 1.0 / self.count as f32;
            RadianceInterval::new(
                self.radiance_sum * reciprocal,
                self.transmittance_sum * reciprocal,
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn composition_is_associative() {
        let a = RadianceInterval::new(Vec3::new(1.0, 0.0, 0.0), 0.75);
        let b = RadianceInterval::new(Vec3::new(0.0, 2.0, 0.0), 0.5);
        let c = RadianceInterval::new(Vec3::new(0.0, 0.0, 4.0), 0.25);
        let left = a.compose(b).compose(c);
        let right = a.compose(b.compose(c));
        assert_abs_diff_eq!(left.radiance.x, right.radiance.x, epsilon = 1.0e-6);
        assert_abs_diff_eq!(left.radiance.y, right.radiance.y, epsilon = 1.0e-6);
        assert_abs_diff_eq!(left.radiance.z, right.radiance.z, epsilon = 1.0e-6);
        assert_abs_diff_eq!(left.transmittance, right.transmittance, epsilon = 1.0e-6);
    }

    #[test]
    fn transparent_is_identity() {
        let value = RadianceInterval::new(Vec3::new(1.0, 2.0, 3.0), 0.3);
        assert_eq!(RadianceInterval::TRANSPARENT.compose(value), value);
        assert_eq!(value.compose(RadianceInterval::TRANSPARENT), value);
    }

    #[test]
    fn missing_current_and_history_remain_invalid() {
        assert_eq!(temporal_resolve(None, None, 0.9), None);
        let current = RadianceInterval::new(Vec3::new(1.0, 2.0, 3.0), 0.25);
        let previous = RadianceInterval::new(Vec3::new(4.0, 5.0, 6.0), 0.75);
        assert_eq!(temporal_resolve(Some(current), None, 0.9), Some(current));
        assert_eq!(temporal_resolve(None, Some(previous), 0.9), Some(previous));
        assert_eq!(
            temporal_resolve(Some(current), Some(previous), 0.0),
            Some(current)
        );
    }
}

//! Deterministic framebuffer capture and contact-sheet utilities.

use std::path::{Path, PathBuf};

use image::{imageops, ImageBuffer, Rgba, RgbaImage};
use serde::Serialize;
use sokol::gfx as sg;

#[cfg(windows)]
pub struct D3d11GpuTimer {
    disjoint: windows::Win32::Graphics::Direct3D11::ID3D11Query,
    timestamps: Vec<windows::Win32::Graphics::Direct3D11::ID3D11Query>,
}

#[cfg(windows)]
impl D3d11GpuTimer {
    pub fn new(stage_count: usize) -> Result<Self, String> {
        use std::mem::ManuallyDrop;

        use windows::{
            core::Interface,
            Win32::Graphics::Direct3D11::{
                ID3D11Device, ID3D11Query, D3D11_QUERY_DESC, D3D11_QUERY_TIMESTAMP,
                D3D11_QUERY_TIMESTAMP_DISJOINT,
            },
        };

        if sg::query_backend() != sg::Backend::D3d11 {
            return Err("GPU timestamp profiling currently requires D3D11".to_owned());
        }
        let device_ptr = sg::d3d11_device();
        if device_ptr.is_null() {
            return Err("Sokol returned a null D3D11 device".to_owned());
        }
        let device = ManuallyDrop::new(unsafe { ID3D11Device::from_raw(device_ptr.cast_mut().cast()) });
        let create = |query_type| -> Result<ID3D11Query, String> {
            let description = D3D11_QUERY_DESC {
                Query: query_type,
                MiscFlags: 0,
            };
            let mut query = None;
            unsafe { device.CreateQuery(&description, Some(&mut query)) }
                .map_err(|error| format!("CreateQuery failed: {error}"))?;
            query.ok_or_else(|| "CreateQuery returned no query".to_owned())
        };
        let disjoint = create(D3D11_QUERY_TIMESTAMP_DISJOINT)?;
        let timestamps = (0..=stage_count)
            .map(|_| create(D3D11_QUERY_TIMESTAMP))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { disjoint, timestamps })
    }

    pub fn begin(&self) {
        use std::mem::ManuallyDrop;

        use windows::{core::Interface, Win32::Graphics::Direct3D11::ID3D11DeviceContext};
        let context = ManuallyDrop::new(unsafe {
            ID3D11DeviceContext::from_raw(sg::d3d11_device_context().cast_mut().cast())
        });
        unsafe {
            context.Begin(&self.disjoint);
            context.End(&self.timestamps[0]);
        }
    }

    pub fn mark(&self, boundary: usize) {
        use std::mem::ManuallyDrop;

        use windows::{core::Interface, Win32::Graphics::Direct3D11::ID3D11DeviceContext};
        let context = ManuallyDrop::new(unsafe {
            ID3D11DeviceContext::from_raw(sg::d3d11_device_context().cast_mut().cast())
        });
        unsafe { context.End(&self.timestamps[boundary + 1]) };
    }

    pub fn end(&self) {
        use std::mem::ManuallyDrop;

        use windows::{core::Interface, Win32::Graphics::Direct3D11::ID3D11DeviceContext};
        let context = ManuallyDrop::new(unsafe {
            ID3D11DeviceContext::from_raw(sg::d3d11_device_context().cast_mut().cast())
        });
        unsafe { context.End(&self.disjoint) };
    }

    pub fn resolve_ms(&self) -> Result<Vec<f64>, String> {
        use std::{ffi::c_void, mem, mem::ManuallyDrop, thread};

        use windows::{
            core::Interface,
            Win32::Graphics::Direct3D11::{ID3D11DeviceContext, D3D11_QUERY_DATA_TIMESTAMP_DISJOINT},
        };
        let context = ManuallyDrop::new(unsafe {
            ID3D11DeviceContext::from_raw(sg::d3d11_device_context().cast_mut().cast())
        });
        unsafe { context.Flush() };
        let mut disjoint = D3D11_QUERY_DATA_TIMESTAMP_DISJOINT::default();
        for _ in 0..1_000_000 {
            disjoint.Frequency = 0;
            unsafe {
                context
                    .GetData(
                        &self.disjoint,
                        Some((&mut disjoint as *mut D3D11_QUERY_DATA_TIMESTAMP_DISJOINT).cast::<c_void>()),
                        mem::size_of::<D3D11_QUERY_DATA_TIMESTAMP_DISJOINT>() as u32,
                        0,
                    )
                    .map_err(|error| format!("timestamp disjoint GetData failed: {error}"))?;
            }
            if disjoint.Frequency != 0 {
                break;
            }
            thread::yield_now();
        }
        if disjoint.Frequency == 0 {
            return Err("GPU timestamp disjoint query timed out".to_owned());
        }
        if disjoint.Disjoint.as_bool() {
            return Err("GPU timestamps were disjoint".to_owned());
        }
        let mut values = Vec::with_capacity(self.timestamps.len());
        for query in &self.timestamps {
            let mut value = u64::MAX;
            for _ in 0..1_000_000 {
                unsafe {
                    context
                        .GetData(
                            query,
                            Some((&mut value as *mut u64).cast::<c_void>()),
                            mem::size_of::<u64>() as u32,
                            0,
                        )
                        .map_err(|error| format!("timestamp GetData failed: {error}"))?;
                }
                if value != u64::MAX {
                    break;
                }
                thread::yield_now();
            }
            if value == u64::MAX {
                return Err("GPU timestamp query timed out".to_owned());
            }
            values.push(value);
        }
        Ok(values
            .windows(2)
            .map(|pair| (pair[1] - pair[0]) as f64 * 1_000.0 / disjoint.Frequency as f64)
            .collect())
    }
}

#[cfg(not(windows))]
pub struct D3d11GpuTimer;

#[cfg(not(windows))]
impl D3d11GpuTimer {
    pub fn new(_stage_count: usize) -> Result<Self, String> {
        Err("GPU timestamp profiling currently requires D3D11".to_owned())
    }

    pub fn begin(&self) {}
    pub fn mark(&self, _boundary: usize) {}
    pub fn end(&self) {}

    pub fn resolve_ms(&self) -> Result<Vec<f64>, String> {
        Err("GPU timestamp profiling currently requires D3D11".to_owned())
    }
}

#[cfg(windows)]
pub fn read_rgba8(image: sg::Image, width: u32, height: u32) -> Result<Vec<u8>, String> {
    use std::mem::ManuallyDrop;

    use windows::{
        core::Interface,
        Win32::Graphics::Direct3D11::{
            ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D, D3D11_CPU_ACCESS_READ,
            D3D11_MAPPED_SUBRESOURCE, D3D11_MAP_READ, D3D11_TEXTURE2D_DESC, D3D11_USAGE_STAGING,
        },
    };

    if sg::query_backend() != sg::Backend::D3d11 {
        return Err(format!(
            "framebuffer readback is currently implemented for D3D11, got {:?}",
            sg::query_backend()
        ));
    }
    let native = sg::d3d11_query_image_info(image);
    if native.tex2d.is_null() {
        return Err("Sokol returned a null D3D11 texture".to_owned());
    }
    let device_ptr = sg::d3d11_device();
    let context_ptr = sg::d3d11_device_context();
    if device_ptr.is_null() || context_ptr.is_null() {
        return Err("Sokol returned a null D3D11 device/context".to_owned());
    }
    let source = ManuallyDrop::new(unsafe { ID3D11Texture2D::from_raw(native.tex2d.cast_mut().cast()) });
    let device = ManuallyDrop::new(unsafe { ID3D11Device::from_raw(device_ptr.cast_mut().cast()) });
    let context = ManuallyDrop::new(unsafe { ID3D11DeviceContext::from_raw(context_ptr.cast_mut().cast()) });
    let mut description = D3D11_TEXTURE2D_DESC::default();
    unsafe { source.GetDesc(&mut description) };
    description.Usage = D3D11_USAGE_STAGING;
    description.BindFlags = 0;
    description.CPUAccessFlags = D3D11_CPU_ACCESS_READ.0 as u32;
    description.MiscFlags = 0;
    let mut staging: Option<ID3D11Texture2D> = None;
    unsafe { device.CreateTexture2D(&description, None, Some(&mut staging)) }
        .map_err(|error| format!("CreateTexture2D staging failed: {error}"))?;
    let staging = staging.ok_or("CreateTexture2D returned no staging texture")?;
    unsafe {
        context.CopyResource(&staging, &**source);
    }
    let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
    unsafe {
        context
            .Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
            .map_err(|error| format!("D3D11 Map failed: {error}"))?;
    }
    let row_bytes = width as usize * 4;
    let mut pixels = vec![0_u8; row_bytes * height as usize];
    for row in 0..height as usize {
        let source_row = unsafe {
            std::slice::from_raw_parts(
                mapped.pData.cast::<u8>().add(row * mapped.RowPitch as usize),
                row_bytes,
            )
        };
        // The fullscreen present pass maps raw storage-image row zero to the
        // bottom of the D3D framebuffer. Mirror that mapping so PNG captures
        // have the same orientation as the interactive window.
        let destination_row = height as usize - 1 - row;
        pixels[destination_row * row_bytes..(destination_row + 1) * row_bytes].copy_from_slice(source_row);
    }
    unsafe { context.Unmap(&staging, 0) };
    Ok(pixels)
}

pub fn save_rgba8(pixels: &[u8], width: u32, height: u32, destination: &Path) -> Result<(), String> {
    if pixels.len() != width as usize * height as usize * 4 {
        return Err("RGBA8 byte count does not match the requested dimensions".to_owned());
    }
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    image::save_buffer_with_format(
        destination,
        pixels,
        width,
        height,
        image::ColorType::Rgba8,
        image::ImageFormat::Png,
    )
    .map_err(|error| error.to_string())
}

pub fn capture_rgba8(image: sg::Image, width: u32, height: u32, destination: &Path) -> Result<(), String> {
    let pixels = read_rgba8(image, width, height)?;
    save_rgba8(&pixels, width, height, destination)
}

#[cfg(not(windows))]
pub fn read_rgba8(_image: sg::Image, _width: u32, _height: u32) -> Result<Vec<u8>, String> {
    Err("framebuffer readback is not implemented for this Sokol backend".to_owned())
}

#[cfg(windows)]
pub fn read_buffer_words(buffer: sg::Buffer, word_offset: u32, word_count: u32) -> Result<Vec<u32>, String> {
    use std::mem::ManuallyDrop;

    use windows::{
        core::Interface,
        Win32::Graphics::Direct3D11::{
            ID3D11Buffer, ID3D11Device, ID3D11DeviceContext, D3D11_BOX, D3D11_BUFFER_DESC,
            D3D11_CPU_ACCESS_READ, D3D11_MAPPED_SUBRESOURCE, D3D11_MAP_READ, D3D11_USAGE_STAGING,
        },
    };

    if sg::query_backend() != sg::Backend::D3d11 {
        return Err("diagnostic readback currently requires D3D11".to_owned());
    }
    let native = sg::d3d11_query_buffer_info(buffer);
    if native.buf.is_null() {
        return Err("Sokol returned a null D3D11 buffer".to_owned());
    }
    let source = ManuallyDrop::new(unsafe { ID3D11Buffer::from_raw(native.buf.cast_mut().cast()) });
    let device = ManuallyDrop::new(unsafe { ID3D11Device::from_raw(sg::d3d11_device().cast_mut().cast()) });
    let context = ManuallyDrop::new(unsafe {
        ID3D11DeviceContext::from_raw(sg::d3d11_device_context().cast_mut().cast())
    });
    let description = D3D11_BUFFER_DESC {
        ByteWidth: word_count * 4,
        Usage: D3D11_USAGE_STAGING,
        BindFlags: 0,
        CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
        MiscFlags: 0,
        StructureByteStride: 0,
    };
    let mut staging: Option<ID3D11Buffer> = None;
    unsafe { device.CreateBuffer(&description, None, Some(&mut staging)) }
        .map_err(|error| format!("CreateBuffer staging failed: {error}"))?;
    let staging = staging.ok_or("CreateBuffer returned no staging buffer")?;
    let source_box = D3D11_BOX {
        left: word_offset * 4,
        right: (word_offset + word_count) * 4,
        top: 0,
        bottom: 1,
        front: 0,
        back: 1,
    };
    unsafe {
        context.CopySubresourceRegion(&staging, 0, 0, 0, 0, &**source, 0, Some(&source_box));
    }
    let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
    unsafe {
        context
            .Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
            .map_err(|error| format!("D3D11 buffer Map failed: {error}"))?;
    }
    let words =
        unsafe { std::slice::from_raw_parts(mapped.pData.cast::<u32>(), word_count as usize).to_vec() };
    unsafe { context.Unmap(&staging, 0) };
    Ok(words)
}

#[cfg(not(windows))]
pub fn read_buffer_words(
    _buffer: sg::Buffer,
    _word_offset: u32,
    _word_count: u32,
) -> Result<Vec<u32>, String> {
    Err("diagnostic readback is not implemented for this backend".to_owned())
}

pub fn write_contact_sheet(paths: &[PathBuf], destination: &Path) -> Result<(), String> {
    const THUMBNAIL_WIDTH: u32 = 320;
    const THUMBNAIL_HEIGHT: u32 = 180;
    const COLUMNS: u32 = 3;
    let rows = (paths.len() as u32).div_ceil(COLUMNS);
    let mut sheet: RgbaImage = ImageBuffer::from_pixel(
        THUMBNAIL_WIDTH * COLUMNS,
        THUMBNAIL_HEIGHT * rows,
        Rgba([8, 10, 14, 255]),
    );
    for (index, path) in paths.iter().enumerate() {
        let source = image::open(path).map_err(|error| error.to_string())?.into_rgba8();
        let thumbnail = imageops::resize(
            &source,
            THUMBNAIL_WIDTH,
            THUMBNAIL_HEIGHT,
            imageops::FilterType::Lanczos3,
        );
        let x = index as u32 % COLUMNS * THUMBNAIL_WIDTH;
        let y = index as u32 / COLUMNS * THUMBNAIL_HEIGHT;
        imageops::overlay(&mut sheet, &thumbnail, i64::from(x), i64::from(y));
    }
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    sheet.save(destination).map_err(|error| error.to_string())
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct ImageDelta {
    pub mean_absolute_u8: f64,
    pub root_mean_square_u8: f64,
    pub p99_absolute_u8: u8,
    pub maximum_absolute_u8: u8,
    pub changed_channel_fraction: f64,
}

pub fn image_delta(first: &[u8], second: &[u8]) -> Result<ImageDelta, String> {
    if first.len() != second.len() || first.len() % 4 != 0 {
        return Err("RGBA8 comparison inputs have incompatible lengths".to_owned());
    }
    let mut histogram = [0_u64; 256];
    let mut absolute_sum = 0_u64;
    let mut square_sum = 0_u64;
    let mut changed = 0_u64;
    let mut samples = 0_u64;
    for (a, b) in first
        .chunks_exact(4)
        .zip(second.chunks_exact(4))
        .flat_map(|(a, b)| a[..3].iter().zip(&b[..3]))
    {
        let delta = a.abs_diff(*b);
        histogram[delta as usize] += 1;
        absolute_sum += u64::from(delta);
        square_sum += u64::from(delta) * u64::from(delta);
        changed += u64::from(delta != 0);
        samples += 1;
    }
    if samples == 0 {
        return Err("RGBA8 comparison input is empty".to_owned());
    }
    let percentile_target = samples * 99 / 100;
    let mut cumulative = 0_u64;
    let mut p99 = 0_u8;
    let mut found_p99 = false;
    let mut maximum = 0_u8;
    for (delta, count) in histogram.into_iter().enumerate() {
        if count > 0 {
            maximum = delta as u8;
        }
        cumulative += count;
        if cumulative >= percentile_target && !found_p99 {
            p99 = delta as u8;
            found_p99 = true;
        }
    }
    Ok(ImageDelta {
        mean_absolute_u8: absolute_sum as f64 / samples as f64,
        root_mean_square_u8: (square_sum as f64 / samples as f64).sqrt(),
        p99_absolute_u8: p99,
        maximum_absolute_u8: maximum,
        changed_channel_fraction: changed as f64 / samples as f64,
    })
}

pub fn save_delta_heatmap(
    first: &[u8],
    second: &[u8],
    width: u32,
    height: u32,
    destination: &Path,
) -> Result<(), String> {
    if first.len() != second.len() || first.len() != width as usize * height as usize * 4 {
        return Err("RGBA8 heatmap inputs have incompatible lengths".to_owned());
    }
    let mut heatmap = vec![0_u8; first.len()];
    for ((first_pixel, second_pixel), output) in first
        .chunks_exact(4)
        .zip(second.chunks_exact(4))
        .zip(heatmap.chunks_exact_mut(4))
    {
        for channel in 0..3 {
            output[channel] = first_pixel[channel]
                .abs_diff(second_pixel[channel])
                .saturating_mul(8);
        }
        output[3] = 255;
    }
    save_rgba8(&heatmap, width, height, destination)
}

#[cfg(test)]
mod tests {
    use super::image_delta;

    #[test]
    fn image_delta_ignores_alpha_and_reports_rgb_change() {
        let report = image_delta(&[0, 0, 0, 0], &[3, 4, 0, 255]).unwrap();
        assert_eq!(report.maximum_absolute_u8, 4);
        assert!((report.root_mean_square_u8 - (25.0_f64 / 3.0).sqrt()).abs() < 1.0e-12);
        assert!((report.changed_channel_fraction - 2.0 / 3.0).abs() < 1.0e-12);
    }
}

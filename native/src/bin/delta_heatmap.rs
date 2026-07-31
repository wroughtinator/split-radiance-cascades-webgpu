use std::{env, path::PathBuf, process::ExitCode};

use image::ImageReader;
use split_radiance_cascades_native::capture::save_delta_heatmap;

fn main() -> ExitCode {
    let arguments: Vec<PathBuf> = env::args_os().skip(1).map(PathBuf::from).collect();
    if arguments.len() != 3 {
        eprintln!("usage: delta_heatmap <reference.png> <comparison.png> <output.png>");
        return ExitCode::FAILURE;
    }
    let result = (|| {
        let first = ImageReader::open(&arguments[0])
            .map_err(|error| error.to_string())?
            .decode()
            .map_err(|error| error.to_string())?
            .to_rgba8();
        let second = ImageReader::open(&arguments[1])
            .map_err(|error| error.to_string())?
            .decode()
            .map_err(|error| error.to_string())?
            .to_rgba8();
        if first.dimensions() != second.dimensions() {
            return Err("image dimensions differ".to_owned());
        }
        save_delta_heatmap(
            first.as_raw(),
            second.as_raw(),
            first.width(),
            first.height(),
            &arguments[2],
        )
    })();
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

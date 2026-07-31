use std::{env, path::PathBuf, process::ExitCode};

use split_radiance_cascades_native::validation::audit;

fn main() -> ExitCode {
    let asset_root = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .or_else(|| env::var_os("SPLIT_RC_ASSETS").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("../public/models"));
    let report = audit(&asset_root, true);
    match serde_json::to_string_pretty(&report) {
        Ok(json) => println!("{json}"),
        Err(error) => {
            eprintln!("failed to encode validation report: {error}");
            return ExitCode::FAILURE;
        }
    }
    if report.passed() && !report.skipped_sponza && report.scene_count == 12 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

//! OpenPyTEA Tauri shell.
//!
//! Spawns the bundled PyInstaller backend (`openpytea-backend`) as a child
//! process at startup, captures the port it prints on stdout, and exposes
//! that port to the frontend via the `get_api_base` IPC command.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use tauri::{AppHandle, Emitter, Manager, RunEvent};

/// Shared state holding the spawned child handle and the discovered port.
#[derive(Default)]
struct BackendState {
    child: Mutex<Option<Child>>,
    port: Mutex<Option<u16>>,
}

/// Locate the backend binary.
///
/// In a packaged build the PyInstaller `onedir` output lives in the app's
/// resource directory under `openpytea-backend/`. In `tauri dev` we fall
/// back to the repo's `dist/` (built by `python scripts/build_sidecar.py`).
fn find_backend_binary(app: &AppHandle) -> Result<PathBuf, String> {
    let bin_name = if cfg!(windows) {
        "openpytea-backend.exe"
    } else {
        "openpytea-backend"
    };

    // Bundled location (production).
    if let Ok(rd) = app.path().resource_dir() {
        let bundled = rd.join("openpytea-backend").join(bin_name);
        if bundled.exists() {
            return Ok(bundled);
        }
    }

    // Dev fallback: <repo-root>/dist/openpytea-backend/<bin>
    let dev = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .ok_or_else(|| "could not resolve repo root from CARGO_MANIFEST_DIR".to_string())?
        .join("dist")
        .join("openpytea-backend")
        .join(bin_name);
    if dev.exists() {
        Ok(dev)
    } else {
        Err(format!(
            "backend binary not found in resource dir or at {}",
            dev.display()
        ))
    }
}

/// Spawn the backend and watch its stdout for the port marker.
fn spawn_backend(app: AppHandle) {
    let binary = match find_backend_binary(&app) {
        Ok(p) => p,
        Err(e) => {
            log::error!("backend binary missing: {}", e);
            return;
        }
    };
    log::info!("spawning backend: {}", binary.display());

    let mut child = match Command::new(&binary)
        .arg("--port")
        .arg("0")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            log::error!("failed to spawn backend ({}): {}", binary.display(), e);
            return;
        }
    };

    let stdout = match child.stdout.take() {
        Some(s) => s,
        None => {
            log::error!("backend child has no piped stdout");
            return;
        }
    };
    let stderr = child.stderr.take();

    // Stash the child handle so the Exit event can kill it.
    {
        let state = app.state::<BackendState>();
        *state.child.lock().unwrap() = Some(child);
    }

    // Reader thread for stdout: scan for the port marker.
    let app_clone = app.clone();
    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines().map_while(Result::ok) {
            log::info!("backend: {}", line);
            if let Some(rest) = line.strip_prefix("OPENPYTEA_BACKEND_PORT=") {
                if let Ok(port) = rest.trim().parse::<u16>() {
                    let state = app_clone.state::<BackendState>();
                    *state.port.lock().unwrap() = Some(port);
                    log::info!("backend ready on port {}", port);
                    let _ = app_clone.emit("backend-ready", port);
                }
            }
        }
        log::info!("backend stdout closed");
    });

    // Reader thread for stderr: just drain it so uvicorn never blocks on a
    // full pipe buffer (~64 KB on macOS). uvicorn writes access logs and
    // warnings here; we forward each line to our logger.
    if let Some(stderr) = stderr {
        std::thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines().map_while(Result::ok) {
                log::warn!("backend[stderr]: {}", line);
            }
            log::info!("backend stderr closed");
        });
    }
}

/// IPC command: returns the API base URL once the backend has reported its
/// port. The frontend polls this on startup.
#[tauri::command]
fn get_api_base(state: tauri::State<'_, BackendState>) -> Option<String> {
    state
        .port
        .lock()
        .ok()
        .and_then(|g| *g)
        .map(|p| format!("http://127.0.0.1:{}/api", p))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let context = tauri::generate_context!();
    let app = tauri::Builder::default()
        .manage(BackendState::default())
        .invoke_handler(tauri::generate_handler![get_api_base])
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            spawn_backend(app.handle().clone());
            Ok(())
        })
        .build(context)
        .expect("error while building tauri application");

    app.run(|app_handle, event| {
        if matches!(event, RunEvent::Exit) {
            let maybe_child = {
                let state = app_handle.state::<BackendState>();
                let mut guard = state.child.lock().unwrap();
                guard.take()
            };
            if let Some(mut child) = maybe_child {
                log::info!("terminating backend (pid {})", child.id());
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    });
}

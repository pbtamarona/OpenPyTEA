//! OpenPyTEA Tauri shell.
//!
//! Spawns the bundled PyInstaller backend (`openpytea-backend`) as a child
//! process at startup, captures the port it prints on stdout, and exposes
//! that port to the frontend via the `get_api_base` IPC command.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use tauri::{AppHandle, Emitter, Manager, Runtime, RunEvent, WindowEvent};
use tauri::menu::{Menu, MenuItem, PredefinedMenuItem, Submenu};
use tauri_plugin_log::{Target, TargetKind};

/// Shared state holding the spawned child handle and the discovered port.
#[derive(Default)]
struct BackendState {
    child: Mutex<Option<Child>>,
    port: Mutex<Option<u16>>,
}

/// Set to true after the frontend has confirmed it's OK to actually quit
/// (user clicked Save → success or Don't Save in the unsaved-work modal).
/// Without this we'd loop: app.exit() → ExitRequested → prevent → ask JS
/// again → JS calls force_quit → ExitRequested → …
#[derive(Default)]
struct ConfirmedExit(AtomicBool);

/// File paths the OS asked us to open before the webview was ready. The
/// frontend drains this list on mount via take_pending_open_files() to
/// catch cold-start file double-clicks.
#[derive(Default)]
struct OpenFileQueue(Mutex<Vec<String>>);

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

/// IPC command: drain and return any file paths the OS asked us to open
/// before the frontend was ready (cold-start double-click case).
#[tauri::command]
fn take_pending_open_files(q: tauri::State<'_, OpenFileQueue>) -> Vec<String> {
    let mut guard = q.0.lock().unwrap();
    std::mem::take(&mut *guard)
}

/// IPC command: actually quit the app, bypassing the
/// CloseRequested / ExitRequested confirmation cycle. Called by the
/// frontend after the user has either saved or chosen "Don't Save" in the
/// unsaved-work modal.
#[tauri::command]
fn force_quit(app: AppHandle, confirmed: tauri::State<'_, ConfirmedExit>) {
    confirmed.0.store(true, Ordering::Relaxed);
    app.exit(0);
}

/// Build the native macOS-style menu bar (App / File / Edit). The same
/// structure is used on Windows/Linux, where it renders as a window menu
/// bar.
fn build_app_menu<R: Runtime>(app: &AppHandle<R>) -> tauri::Result<Menu<R>> {
    let new_proj = MenuItem::with_id(app, "menu:new", "New Project", true, Some("CmdOrCtrl+N"))?;
    let open = MenuItem::with_id(app, "menu:open", "Open…", true, Some("CmdOrCtrl+O"))?;
    let save = MenuItem::with_id(app, "menu:save", "Save", true, Some("CmdOrCtrl+S"))?;
    let save_as = MenuItem::with_id(app, "menu:save-as", "Save As…", true, Some("CmdOrCtrl+Shift+S"))?;

    let file = Submenu::with_items(
        app,
        "File",
        true,
        &[
            &new_proj,
            &open,
            &PredefinedMenuItem::separator(app)?,
            &save,
            &save_as,
            &PredefinedMenuItem::separator(app)?,
            &PredefinedMenuItem::close_window(app, None)?,
        ],
    )?;

    // Standard text-editing items — required for Cut/Copy/Paste to work in
    // form fields on macOS, since those go through the menu system.
    let edit = Submenu::with_items(
        app,
        "Edit",
        true,
        &[
            &PredefinedMenuItem::undo(app, None)?,
            &PredefinedMenuItem::redo(app, None)?,
            &PredefinedMenuItem::separator(app)?,
            &PredefinedMenuItem::cut(app, None)?,
            &PredefinedMenuItem::copy(app, None)?,
            &PredefinedMenuItem::paste(app, None)?,
            &PredefinedMenuItem::select_all(app, None)?,
        ],
    )?;

    // The leftmost macOS menu (named after the app) — About / Hide / Quit.
    let app_submenu = Submenu::with_items(
        app,
        "OpenPyTEA",
        true,
        &[
            &PredefinedMenuItem::about(app, Some("About OpenPyTEA"), None)?,
            &PredefinedMenuItem::separator(app)?,
            &PredefinedMenuItem::services(app, None)?,
            &PredefinedMenuItem::separator(app)?,
            &PredefinedMenuItem::hide(app, None)?,
            &PredefinedMenuItem::hide_others(app, None)?,
            &PredefinedMenuItem::show_all(app, None)?,
            &PredefinedMenuItem::separator(app)?,
            &PredefinedMenuItem::quit(app, None)?,
        ],
    )?;

    Menu::with_items(app, &[&app_submenu, &file, &edit])
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let context = tauri::generate_context!();
    let app = tauri::Builder::default()
        .manage(BackendState::default())
        .manage(ConfirmedExit::default())
        .manage(OpenFileQueue::default())
        .invoke_handler(tauri::generate_handler![
            get_api_base,
            take_pending_open_files,
            force_quit,
        ])
        .on_window_event(|window, event| {
            // Fired when the user clicks the red close button. Cancel the
            // close, ask the frontend to handle it (it may or may not show
            // a "save first?" modal depending on dirty state).
            if let WindowEvent::CloseRequested { api, .. } = event {
                let app = window.app_handle();
                let confirmed = app.state::<ConfirmedExit>();
                if !confirmed.0.load(Ordering::Relaxed) {
                    api.prevent_close();
                    let _ = app.emit("request-close", ());
                }
            }
        })
        .on_menu_event(|app, event| {
            // Every native menu item we own carries an id beginning with
            // "menu:". Forward those to the frontend; let predefined items
            // (cut/copy/paste/quit/etc.) be handled by the OS itself.
            let id = event.id().as_ref().to_string();
            if id.starts_with("menu:") {
                let _ = app.emit("menu", id);
            }
        })
        .setup(|app| {
            // Enable logging in release too — production issues are otherwise
            // invisible. Stdout target is visible when launched from terminal;
            // LogDir target writes to ~/Library/Logs/org.openpytea.app/ on
            // macOS, %LOCALAPPDATA%\org.openpytea.app\logs on Windows.
            app.handle().plugin(
                tauri_plugin_log::Builder::default()
                    .level(log::LevelFilter::Info)
                    .targets([
                        Target::new(TargetKind::Stdout),
                        Target::new(TargetKind::LogDir { file_name: None }),
                    ])
                    .build(),
            )?;
            // Native file dialogs (open / save panels) and direct read/write
            // access to the path the user picks. Used by the File ▸ Save /
            // Save As / Open flow.
            app.handle().plugin(tauri_plugin_dialog::init())?;
            app.handle().plugin(tauri_plugin_fs::init())?;
            // Install the system menu bar (macOS top-of-screen, window menu
            // on Win/Linux). Menu items emit a `menu` event the frontend
            // listens for.
            let menu = build_app_menu(app.handle())?;
            app.set_menu(menu)?;
            log::info!(
                "OpenPyTEA shell starting (release={}, version={})",
                !cfg!(debug_assertions),
                env!("CARGO_PKG_VERSION"),
            );
            spawn_backend(app.handle().clone());
            Ok(())
        })
        .build(context)
        .expect("error while building tauri application");

    app.run(|app_handle, event| {
        match event {
            // Cmd+Q on macOS, or any other "quit the app" path.
            RunEvent::ExitRequested { api, .. } => {
                let confirmed = app_handle.state::<ConfirmedExit>();
                if !confirmed.0.load(Ordering::Relaxed) {
                    api.prevent_exit();
                    let _ = app_handle.emit("request-close", ());
                }
            }
            RunEvent::Exit => {
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
            // macOS sends this when the user double-clicks a .openpytea file
            // (or chooses Open With → OpenPyTEA) — once the file association
            // declared in tauri.conf.json is registered with the OS.
            //
            // We BOTH emit live (warm-open case: the app is running and the
            // frontend's listener is ready) AND queue (cold-open case: the
            // event fires during app startup before React has mounted).
            RunEvent::Opened { urls } => {
                let q = app_handle.state::<OpenFileQueue>();
                for url in urls {
                    if let Ok(path) = url.to_file_path() {
                        let path_str = path.to_string_lossy().to_string();
                        log::info!("open-file requested: {}", path_str);
                        q.0.lock().unwrap().push(path_str.clone());
                        let _ = app_handle.emit("open-file", path_str);
                    }
                }
            }
            _ => {}
        }
    });
}

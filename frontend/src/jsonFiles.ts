/** Opening and saving the library's JSON files (equipment / plant /
    analysis inputs, run_tea results).

    Desktop: native dialogs, with the file read and written by the Rust
    project-file commands (.json only). Browser: a file input to open; the
    save picker where there is one (Chromium), else a download. */

async function isTauri(): Promise<boolean> {
  try {
    const { isTauri } = await import("@tauri-apps/api/core");
    return isTauri();
  } catch {
    return false;
  }
}

/** Let the user pick a JSON file and parse it; null when cancelled. */
export async function openJsonFile(title: string): Promise<{ name: string; data: unknown } | null> {
  if (await isTauri()) {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const path = await open({ title, multiple: false, filters: [{ name: "JSON", extensions: ["json"] }] });
    if (!path || typeof path !== "string") return null;
    const { invoke } = await import("@tauri-apps/api/core");
    const text = await invoke<string>("read_project_text", { path });
    return { name: path.split(/[/\\]/).pop() ?? path, data: parseJson(text) };
  }
  const file = await new Promise<File | null>((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json,application/json";
    input.onchange = () => resolve(input.files?.[0] ?? null);
    input.oncancel = () => resolve(null);
    input.click();
  });
  return file ? { name: file.name, data: parseJson(await file.text()) } : null;
}

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    throw new Error("The file is not valid JSON");
  }
}

/** Save a JSON file under `name`, letting the user choose where. Resolves
    to where it went (a path on desktop, the name in a browser), or null
    when cancelled. */
export async function saveJsonFile(name: string, content: string, title = "Save JSON file"): Promise<string | null> {
  if (await isTauri()) {
    const { save } = await import("@tauri-apps/plugin-dialog");
    const path = await save({ title, defaultPath: name, filters: [{ name: "JSON", extensions: ["json"] }] });
    if (!path) return null;
    const { invoke } = await import("@tauri-apps/api/core");
    await invoke("write_project_text", { path, contents: content });
    return path;
  }
  const blob = new Blob([content], { type: "application/json" });
  const picker = (window as unknown as {
    showSaveFilePicker?: (opts: unknown) => Promise<{
      name: string;
      createWritable: () => Promise<{ write: (b: Blob) => Promise<void>; close: () => Promise<void> }>;
    }>;
  }).showSaveFilePicker;
  if (picker) {
    try {
      const handle = await picker({
        suggestedName: name,
        id: "openpytea-json",
        types: [{ description: "JSON file", accept: { "application/json": [".json"] } }],
      });
      const w = await handle.createWritable();
      await w.write(blob);
      await w.close();
      return handle.name;
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return null;
      console.warn("Save picker failed, falling back to a download", e);
    }
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
  return name;
}

/** Save one of run_tea's result files (<plant>_*_results.json). */
export async function saveResultsFile(
  fetchFiles: () => Promise<{ files: { name: string; content: string }[] }>,
): Promise<string | null> {
  const { files } = await fetchFiles();
  const f = files[0];
  return saveJsonFile(f.name, f.content, "Save results JSON");
}

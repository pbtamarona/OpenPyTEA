/** Saving downloaded figures to a folder the user picks. */

const DIR_KEY = "openpytea-figure-dir";

function rememberDir(path: string) {
  const i = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  if (i > 0) {
    try { localStorage.setItem(DIR_KEY, path.slice(0, i)); } catch { /* private mode */ }
  }
}

function defaultPath(name: string): string {
  let dir: string | null = null;
  try { dir = localStorage.getItem(DIR_KEY); } catch { /* private mode */ }
  if (!dir) return name;
  const sep = dir.includes("\\") && !dir.includes("/") ? "\\" : "/";
  return `${dir}${sep}${name}`;
}

/** Save a PNG figure, letting the user choose folder and file name first.

    Desktop: native Save dialog (opens in the last folder used), then the
    Rust `write_figure` command writes the file. Chromium browsers: the
    File System Access save picker (remembers its folder). Other browsers:
    a regular download. Resolves false when the user cancels. */
export async function saveFigure(blob: Blob, name: string): Promise<boolean> {
  const filename = name.toLowerCase().endsWith(".png") ? name : `${name}.png`;

  let tauri = false;
  try {
    const { isTauri } = await import("@tauri-apps/api/core");
    tauri = isTauri();
  } catch { /* browser */ }

  if (tauri) {
    const { save, message } = await import("@tauri-apps/plugin-dialog");
    const path = await save({
      title: "Save figure",
      defaultPath: defaultPath(filename),
      filters: [{ name: "PNG image", extensions: ["png"] }],
    });
    if (!path) return false;
    rememberDir(path);
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      const bytes = Array.from(new Uint8Array(await blob.arrayBuffer()));
      await invoke("write_figure", { path, bytes });
      return true;
    } catch (e) {
      await message(`Could not save the figure: ${e}`, { title: "Save figure", kind: "error" });
      return false;
    }
  }

  const picker = (window as unknown as {
    showSaveFilePicker?: (opts: unknown) => Promise<{
      createWritable: () => Promise<{ write: (b: Blob) => Promise<void>; close: () => Promise<void> }>;
    }>;
  }).showSaveFilePicker;
  if (picker) {
    try {
      const handle = await picker({
        suggestedName: filename,
        id: "openpytea-figures", // the browser reopens the last folder for this id
        types: [{ description: "PNG image", accept: { "image/png": [".png"] } }],
      });
      const w = await handle.createWritable();
      await w.write(blob);
      await w.close();
      return true;
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return false;
      console.warn("Save picker failed, falling back to a download", e);
    }
  }

  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
  return true;
}

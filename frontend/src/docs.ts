/** Open the bundled documentation, optionally at a page/section
    ("user_guide/plant.html#configuration-reference"). Desktop: the docs
    window (OpenPyTEA ▸ Documentation). Browser dev: a new tab. */
export async function openDocs(path = "index.html"): Promise<void> {
  try {
    const { isTauri, invoke } = await import("@tauri-apps/api/core");
    if (isTauri()) {
      await invoke("open_docs", { path });
      return;
    }
  } catch (e) {
    console.warn("open_docs failed, falling back to a browser tab", e);
  }
  window.open(`/docs/${path}`, "openpytea-docs");
}

/** Open an external http(s) link in the system browser. In the desktop app
    a plain link would navigate the app's own window away from the GUI. */
export async function openExternal(url: string): Promise<void> {
  try {
    const { isTauri, invoke } = await import("@tauri-apps/api/core");
    if (isTauri()) {
      await invoke("open_external", { url });
      return;
    }
  } catch (e) {
    console.warn("open_external failed, falling back to a browser tab", e);
  }
  window.open(url, "_blank", "noopener");
}

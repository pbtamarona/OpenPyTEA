"""Build the Sphinx docs into the GUI and extract its (?) help text.

    python scripts/build_docs.py [--help-only]

1. Builds ``docs/`` to HTML in ``frontend/public/docs`` (served at
   ``/docs/`` by Vite and bundled into the desktop app). MathJax and the
   DataTables bundle are downloaded into the build so the docs render
   offline — the desktop app's CSP only allows its own files. If a
   download fails the CDN URLs are kept (fine online / in the browser).
2. Writes ``frontend/src/help/docHelp.json``: the reference tables of the
   user guide (plant configuration keys, equipment parameters, Monte
   Carlo uncertainty keys, metrics) as plain text keyed by table and key,
   with the docs page + section anchor each one links to. The JSON is
   committed, so the frontend builds without Sphinx; rerun this script
   after editing those tables.

Requires ``pip install -r docs/requirements.txt`` for step 1.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
OUT = ROOT / "frontend" / "public" / "docs"
HELP_JSON = ROOT / "frontend" / "src" / "help" / "docHelp.json"

# Windows CI consoles default to cp1252; Sphinx warnings quote docstrings
# with characters like "σ" and would raise UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

VENDOR = {
    # MathJax 3's SVG build is self-contained (no separately loaded fonts)
    "mathjax/tex-svg.js": "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js",
    "datatables/datatables.min.js": "https://cdn.datatables.net/v/dt/dt-2.3.5/fc-5.0.4/datatables.min.js",
    "datatables/datatables.min.css": "https://cdn.datatables.net/v/dt/dt-2.3.5/fc-5.0.4/datatables.min.css",
}

# (help-key prefix, rst file, section title above the table, first header,
#  description column(s), "default" column + its label or None). A title
#  used by several sections ("Constructor parameters" for Equipment and
#  for CompositeEquipment) takes "title#n" for its n-th occurrence.
TABLES = [
    ("plant", "user_guide/plant.rst", "Configuration reference", "Key", [2], (1, "Default")),
    ("equipment", "user_guide/equipment.rst", "Constructor parameters#1", "Parameter", [2], None),
    ("composite", "user_guide/equipment.rst", "Constructor parameters#2", "Parameter", [2], None),
    ("project_uncertainty", "user_guide/analysis.rst", "Configuring input uncertainties", "Key",
     [1], (2, "Default std")),
    ("dist", "user_guide/analysis.rst", "Configuring input uncertainties", "``dist_id``", [1, 2, 3], None),
    ("metric", "user_guide/analysis.rst", "One-way sensitivity analysis", "Value", [1], None),
]


# ── Sphinx build ──────────────────────────────────────────────────────


def _download_vendor(dest: Path) -> dict[str, str]:
    """Fetch the offline assets; return config overrides for the ones fetched."""
    got = {}
    for rel, url in VENDOR.items():
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            # cdn.datatables.net answers 403 to urllib's default User-Agent
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (openpytea docs build)"})
            with urllib.request.urlopen(req, timeout=60) as r:
                target.write_bytes(r.read())
            got[rel] = rel
        except Exception as e:  # offline: keep the CDN URL
            print(f"warning: could not download {url} ({e}); using the CDN", file=sys.stderr)
    overrides = {}
    if "mathjax/tex-svg.js" in got:
        overrides["mathjax_path"] = "mathjax/tex-svg.js"  # relative to _static
    return overrides, got


# Remote status badges (PyPI version, license...) on the landing page: the
# desktop app's CSP blocks remote images, so drop them from the bundle
_BADGE = re.compile(r'(<a [^>]*>\s*)?<img [^>]*src="https://img\.shields\.io/[^"]*"[^>]*>(\s*</a>)?')


def _postprocess_html(got: dict[str, str]):
    """Point the pages' CDN DataTables links at the downloaded copies, and
    drop remote badges.

    sphinx-datatables doesn't honour ``-D datatables_js=...`` overrides
    (it's registered after overrides are applied), so rewrite the HTML.
    """
    swaps = {VENDOR[rel]: rel for rel in got if rel.startswith("datatables/")}
    if len(swaps) < 2:
        swaps = {}
    for page in OUT.rglob("*.html"):
        depth = len(page.relative_to(OUT).parts) - 1
        static = "../" * depth + "_static/"
        html = page.read_text(encoding="utf-8")
        new = _BADGE.sub("", html)
        for url, rel in swaps.items():
            new = new.replace(url, static + rel)
        if new != html:
            page.write_text(new, encoding="utf-8")


def build_html():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        vendor = tmp / "vendor"
        overrides, got = _download_vendor(vendor)
        cmd = [
            sys.executable, "-m", "sphinx", "-q", "-b", "html",
            "-d", str(tmp / "doctrees"),  # keep the build cache out of the app
            "-D", "html_copy_source=0",   # no raw .rst copies in the bundle
            "-D", f"html_static_path=_static,{vendor}",
        ]
        for k, v in overrides.items():
            cmd += ["-D", f"{k}={v}"]
        cmd += [str(DOCS), str(OUT)]
        subprocess.run(cmd, check=True, env={**os.environ, "PYTHONUTF8": "1"})
    _postprocess_html(got)
    print(f"docs built into {OUT.relative_to(ROOT)}")


# ── Help text extraction ──────────────────────────────────────────────


def _anchor(title: str) -> str:
    """Docutils' section id for a title."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def _plain(rst: str) -> str:
    """Inline reST → plain text (good enough for tooltips)."""
    s = rst
    s = re.sub(r":(?:ref|doc):`([^`<]+?)\s*<[^>]+>`", r"\1", s)          # :ref:`text <label>`
    s = re.sub(r":(?:ref|doc):`([^`]+)`", r"\1", s)
    s = re.sub(r":(?:class|func|meth|mod|attr):`~?([^`]+)`",
               lambda m: m.group(1).split(".")[-1] + ("()" if ":func:" in m.group(0) or ":meth:" in m.group(0) else ""), s)
    s = re.sub(r":math:`([^`]+)`", lambda m: re.sub(r"\\text\{([^}]*)\}", r"\1", m.group(1)).replace("\\", ""), s)
    s = re.sub(r"`([^`<]+?)\s*<[^>]+>`_+", r"\1", s)                       # `text <url>`_
    s = s.replace("``", "")
    s = re.sub(r"\*\*?([^*]+)\*\*?", r"\1", s)
    s = s.replace("\\$", "$")
    return re.sub(r"\s+", " ", s).strip()


def _sections(lines: list[str]) -> list[tuple[int, str]]:
    """(line index, title) of every section heading."""
    out = []
    for i in range(1, len(lines)):
        under = lines[i].rstrip()
        title = lines[i - 1].strip()
        if (title and len(under) >= 3 and len(set(under)) == 1
                and under[0] in "=-~^\"'`#*+" and not lines[i - 1].startswith(" ")):
            out.append((i - 1, title))
    return out


def _list_tables(lines: list[str]):
    """Yield (start line, rows) for each list-table; rows are lists of cells."""
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith(".. list-table::"):
            start, rows, cell = i, [], None
            i += 1
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith("   ")):
                line = lines[i]
                m_row = re.match(r"^   \* - ?(.*)$", line)
                m_cell = re.match(r"^     - ?(.*)$", line)
                if m_row:
                    rows.append([m_row.group(1)])
                elif m_cell and rows:
                    rows[-1].append(m_cell.group(1))
                elif rows and line.startswith("       "):
                    rows[-1][-1] += " " + line.strip()
                i += 1
            yield start, rows
        else:
            i += 1


def extract_help() -> dict:
    help_ = {}
    for prefix, rel, section, first_header, desc_cols, default in TABLES:
        lines = (DOCS / rel).read_text(encoding="utf-8").splitlines()
        sections = _sections(lines)
        titles = [t for _, t in sections]
        page = rel.replace(".rst", ".html")
        section, _, nth = section.partition("#")
        for start, rows in _list_tables(lines):
            if not rows or rows[0][0].strip() != first_header:
                continue
            before = [(ln, t) for ln, t in sections if ln < start]
            if not before or before[-1][1] != section:
                continue
            if nth and sum(t == section for _, t in before) != int(nth):
                continue
            # Docutils gives a repeated title a generated id ("id1"), so
            # link such a table to the nearest enclosing unique section
            anchor_title = next(t for _, t in reversed(before) if titles.count(t) == 1)
            for row in rows[1:]:
                # ``a``, ``b`` cells name several keys; "0 / 1" two dist ids
                keys = re.findall(r"``([^`]+)``", row[0]) or [k.strip() for k in row[0].split("/")]
                text = " — ".join(_plain(row[c]) for c in desc_cols if c < len(row) and row[c].strip())
                if default and default[0] < len(row) and row[default[0]].strip():
                    text += f" ({default[1]}: {_plain(row[default[0]])}.)"
                for key in keys:
                    key = key.strip('"')
                    help_[f"{prefix}.{key}"] = {"text": text, "doc": f"{page}#{_anchor(anchor_title)}"}
    return help_


def write_help():
    help_ = extract_help()
    HELP_JSON.parent.mkdir(parents=True, exist_ok=True)
    HELP_JSON.write_text(json.dumps(help_, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(help_)} help entries written to {HELP_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    write_help()
    if "--help-only" not in sys.argv:
        build_html()

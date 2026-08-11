import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("../src"))

project = "OpenPyTEA"
copyright = "2026, Panji B. Tamarona, Thijs J.H. Vlugt, Mahinder Ramdin"
author = "Panji B. Tamarona, Thijs J.H. Vlugt, Mahinder Ramdin"
release = "2.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.youtube",
    "sphinxcontrib.jquery",
    "sphinx_datatables",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["caption_links.js"]
html_favicon = "_static/logo-opt.png"
# html_logo = "_static/logo-blue.png"

html_theme_options = {
    "sidebar_hide_name": True,
    "light_logo": "logo-black.png",
    "dark_logo": "logo-white.png",
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#538DFF",
        "color-brand-content": "#538DFF",
        "color-background-primary": "#FFFFFF",
        "color-background-secondary": "#F0F4FF",
        "color-background-border": "#D6E4FF",
        "color-foreground-primary": "#303030",
        "color-foreground-secondary": "#555555",
        "color-foreground-muted": "#777777",
        "color-foreground-border": "#DDDDDD",
        "color-highlight-on-target": "#EBF2FF",
        "color-link": "#538DFF",
        "color-link--hover": "#2B6AE0",
        "color-link-underline": "#99BDFF",
        "color-link-underline--hover": "#538DFF",
    },
    "dark_css_variables": {
        "color-brand-primary": "#538DFF",
        "color-brand-content": "#7AAAFF",
        "color-background-primary": "#1C1C1C",
        "color-background-secondary": "#262626",
        "color-background-border": "#3A3A3A",
        "color-foreground-primary": "#FFFFFF",
        "color-foreground-secondary": "#CCCCCC",
        "color-foreground-muted": "#999999",
        "color-foreground-border": "#444444",
        "color-highlight-on-target": "#1A2A4A",
        "color-link": "#7AAAFF",
        "color-link--hover": "#99BDFF",
        "color-link-underline": "#3A5A99",
        "color-link-underline--hover": "#7AAAFF",
    },
}

html_title = "OpenPyTEA"

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autosummary_generate = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

datatables_class = "sphinx-datatable"
datatables_options = {
    "order": [],
    "pageLength": 10,
    "lengthMenu": [10, 25, 50, 100],
    "scrollX": True,
    "fixedColumns": {"start": 2},
}
# Combined DataTables + FixedColumns bundle (freezes the Category/Type
# columns while scrolling horizontally through the rest of the table).
datatables_js = "https://cdn.datatables.net/v/dt/dt-2.3.5/fc-5.0.4/datatables.min.js"
datatables_css = "https://cdn.datatables.net/v/dt/dt-2.3.5/fc-5.0.4/datatables.min.css"

# --- Generate the searchable cost-correlations table shown in the user
# guide from the package's bundled CSV, so the docs never drift out of
# sync with the actual database. Row order follows the source CSV
# (datatables_options["order"] is left empty so it isn't re-sorted).
_COST_DB_COLUMNS = [
    "category",
    "type",
    "units",
    "s_lower",
    "s_upper",
    "form",
    "cost_year",
    "source",
    "Remarks",
    "key",
]
_COST_DB_HEADERS = [
    "Category",
    "Type",
    "Units",
    "Min size",
    "Max size",
    "Form",
    "Year",
    "Source",
    "Remarks",
    "Key",
]


def _superscript_units(units: str) -> str:
    """Render ``^`` in unit strings (e.g. ``m^3``) as reST superscript."""
    out = []
    for i, ch in enumerate(units):
        if ch == "^" and i + 1 < len(units):
            continue
        if i > 0 and units[i - 1] == "^":
            needs_boundary = bool(out) and out[-1][-1] not in " ([{"
            if needs_boundary:
                out.append("\\ ")
            out.append(f":sup:`{ch}`")
        else:
            out.append(ch)
    return "".join(out)


def _format_size(value: str) -> str:
    """Format a size bound with thousands separators, e.g. 30000 -> 30,000."""
    value = value.strip()
    if not value:
        return ""
    num = float(value)
    if num == int(num):
        return f"{int(num):,}"
    return f"{num:,}"


def _link_for_source(raw_link: str) -> str | None:
    """Normalize the CSV's doi/link column into a single usable URL."""
    first = raw_link.split(" / ")[0].strip()
    if not first:
        return None
    if first.startswith("http://") or first.startswith("https://"):
        return first
    if first.startswith("doi:"):
        return "https://doi.org/" + first[len("doi:") :]
    if first.startswith("doi.org/"):
        return "https://" + first
    return None


def _generate_cost_correlations_table(app):
    src = (
        Path(app.confdir)
        / ".."
        / "src"
        / "openpytea"
        / "data"
        / "cost_correlations.csv"
    )
    dst = Path(app.confdir) / "_static" / "cost_correlations_table.csv"

    with open(src, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            cells = {col: row.get(col, "") or "" for col in _COST_DB_COLUMNS}
            cells["category"] = f"**{cells['category'].strip()}**"
            cells["type"] = f"**{cells['type'].strip()}**"
            cells["units"] = _superscript_units(cells["units"])
            cells["s_lower"] = _format_size(cells["s_lower"])
            cells["s_upper"] = _format_size(cells["s_upper"])
            link = _link_for_source(row.get("doi /  link", ""))
            if link:
                cells["source"] = f"`{cells['source']} <{link}>`__"
            cells["key"] = f"``{cells['key']}``"
            rows.append([cells[col] for col in _COST_DB_COLUMNS])

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_COST_DB_HEADERS)
        writer.writerows(rows)


def setup(app):
    app.connect("builder-inited", _generate_cost_correlations_table)

import os
import sys

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
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["caption_links.js"]

html_theme_options = {
    "sidebar_hide_name": True,
    "light_logo": "OpenPyTEA-logo-black.png",
    "dark_logo": "OpenPyTEA-logo.png",
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

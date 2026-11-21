import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Xtructure"
copyright = "2024, tinker495"
author = "tinker495"
release = "0.1.2"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "AGENTS.md", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
_static_dir = os.path.join(os.path.dirname(__file__), "_static")
html_static_path = ["_static"] if os.path.isdir(_static_dir) else []

napoleon_google_docstring = True
napoleon_numpy_docstring = True

autodoc_default_options = {
    "no-index": True,
}

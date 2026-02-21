import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Xtructure"
copyright = "2024, tinker495"
author = "tinker495"
release = "0.1.2"

# Extensions
extensions = [
    "sphinx.ext.autodoc",  # Core library for html generation from docstrings
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "myst_parser",  # Parse Markdown files
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Theme
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Napoleon settings (optional, defaults are usually fine)
napoleon_google_docstring = True
napoleon_numpy_docstring = True

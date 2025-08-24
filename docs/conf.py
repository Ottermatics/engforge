# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import toml


# Ensure _static and _build folders exist in the same directory as conf.py
conf_dir = os.path.dirname(os.path.abspath(__file__))
for folder in ["_static", "_build","_autosummary"]:
    folder_path = os.path.join(conf_dir, folder)
    if os.path.exists(folder_path):
        if os.path.isdir(folder_path):
            os.rmdir(folder_path)
        else:
            os.remove(folder_path)
    os.makedirs(folder_path,exist_ok=True)

project = "engforge"
copyright = "2024, Kevin Russell"
author = "Kevin Russell"

pyproject_path = os.path.join(conf_dir, "..", "pyproject.toml")
with open(pyproject_path, "r") as f:
    pyproject_data = toml.load(f)
release = pyproject_data.get("project", {}).get("version")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "myst_parser",
    "sphinx_autodoc_typehints",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
html_show_sourcelink = (
    False  # Remove 'view source code' from top of page (for html, not python)
)
autodoc_inherit_docstrings = True  # If no docstring, inherit from base class
set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
# nbsphinx_allow_errors = True  # Continue through Jupyter errors
autodoc_typehints = (
    "description"  # Sphinx-native method. Not as good as sphinx_autodoc_typehints
)
add_module_names = False  # Remove namespaces from class/method signatures

html_theme = "sphinx_rtd_theme"
html_css_files = ["readthedocs-custom.css"]  # Override some CSS settings

# Set Assertion Bypass
os.environ["SPHINX_BUILD"] = "true"

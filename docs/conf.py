# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
# Add the src/ directory so autodoc can import testmechs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# -- Project information -----------------------------------------------------
project = "testmechs"
copyright = "2024-2025, Soonwoo Kwon, Jonathan Roth, Xuanyu Cai, and Wenli Xu"
author = "Soonwoo Kwon, Jonathan Roth, Xuanyu Cai, and Wenli Xu"
version = "0.1.0"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "numpydoc",
]

# -- Options for MyST (Markdown support) -------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
    "dollarmath",
    "amsmath",
]

# -- Options for autodoc -----------------------------------------------------
# Keep member expansion explicit in each autoapi page. A global ``members``
# default expands the root package and re-registers objects already documented
# on their source-module pages.
autodoc_default_options = {}
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Mock imports that may not be available at doc-build time
autodoc_mock_imports = ["osqp", "matplotlib"]

# -- Options for Napoleon (NumPy/Google docstrings) --------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_rtype = True

# -- Options for numpydoc ----------------------------------------------------
numpydoc_show_class_members = False

# -- Options for intersphinx -------------------------------------------------
if os.environ.get("TESTMECHS_DOCS_OFFLINE") == "1":
    intersphinx_mapping = {}
else:
    intersphinx_mapping = {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable/", None),
        "pandas": ("https://pandas.pydata.org/docs/", None),
        "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    }

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}
html_static_path = []

# -- Exclude patterns --------------------------------------------------------
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

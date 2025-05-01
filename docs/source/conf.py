import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))  # Ensure this path is correct for your structure

# -- Project information -----------------------------------------------------
project = 'MCSCF for 2 electron molecules'
copyright = '2025, Evgueni Dinvay'
author = 'Evgueni Dinvay'
release = '0.1'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',  # Auto-generate documentation from docstrings
    'sphinx.ext.napoleon',  # Support for NumPy and Google style docstrings
    'sphinx_autodoc_typehints',  # Automatically add type hints
    'sphinx.ext.mathjax'  # MathJax for rendering math
]

# MathJax path (optional, ensures math renders correctly)
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-MML-AM_CHTML"

html_theme = 'sphinx_rtd_theme'

templates_path = ['_templates']
exclude_patterns = []

language = 'english'

# -- Options for HTML output -------------------------------------------------
html_static_path = ['_static']

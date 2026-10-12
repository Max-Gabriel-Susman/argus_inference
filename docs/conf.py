# Sphinx configuration for argus_inference.
#
# Build from the repository root:
#
#     pip install -r docs/requirements.txt
#     source /opt/ros/humble/setup.bash && source ~/Documents/argus_ws/install/setup.bash
#     sphinx-build -b html docs docs/_build/html
#
# With the workspace sourced, autodoc imports the real modules. Without it
# (CI without ROS, a laptop with only pip) the imports that are missing are
# mocked, so the build still succeeds: the same signatures and docstrings,
# only the bases and annotations lose their links.
import importlib.util
import os
import sys
from xml.etree import ElementTree

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

project = 'argus_inference'
author = 'Max Gabriel Susman'
project_copyright = '2026, Max Gabriel Susman'
release = ElementTree.parse(os.path.join(ROOT, 'package.xml')).findtext('version')
version = release

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

# The module's helpers are underscore-named on purpose: the node is the only
# public thing. They are the interesting part, so they are documented too.
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': True,
    'member-order': 'bysource',
    'show-inheritance': True,
}
autodoc_mock_imports = [
    name for name in ('rclpy', 'argus_core', 'geometry_msgs', 'h5py', 'numpy', 'sklearn')
    if importlib.util.find_spec(name) is None
]
if autodoc_mock_imports:
    print('autodoc: mocking ' + ', '.join(autodoc_mock_imports)
          + ' (source the workspace for the real imports)')

# ament_pep257 checks every docstring with the numpy section rules, so
# docstrings here are numpy style.
napoleon_google_docstring = False
napoleon_numpy_docstring = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

# Command flags like --save-model appear in prose; keep them as typed.
smartquotes = False

html_theme = 'furo'
html_title = 'argus_inference'

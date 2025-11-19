"""
Backward-compatible alias module for xtructure_numpy.

Historically, users imported the NumPy-like helpers via:

    from xtructure import numpy as xnp

This stub keeps that import path working while all functionality lives in
`xtructure.xtructure_numpy`. Keeping this file avoids Python import errors for
code that does `import xtructure.numpy`.
"""

from .xtructure_numpy import *  # noqa: F401,F403


"""
Backward-compatible alias module for xtructure_numpy.

Historically, users imported the NumPy-like helpers via::

    from xtructure import numpy as xnp

This stub keeps that import path working while all functionality lives in
``xtructure.xtructure_numpy``. Keeping this file avoids Python import errors for
code that does ``import xtructure.numpy``.

Prefer the new stable entrypoint::

    import xtructure as xt
    xnp = xt.xnp
"""

from __future__ import annotations

import warnings

from . import xtructure_numpy
from .xtructure_numpy import *  # noqa: F401,F403

warnings.warn(
    "`xtructure.numpy` is deprecated; use `import xtructure as xt; xnp = xt.xnp`",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = xtructure_numpy.__all__

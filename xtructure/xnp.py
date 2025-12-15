"""Official public alias for NumPy-like helpers.

Use ``import xtructure as xt`` and then ``xt.xnp`` for structured array
operations. This module simply re-exports :mod:`xtructure.xtructure_numpy`
without deprecation warnings so callers have a stable, single entrypoint.
"""

from . import xtructure_numpy
from .xtructure_numpy import *  # noqa: F401,F403

__all__ = xtructure_numpy.__all__

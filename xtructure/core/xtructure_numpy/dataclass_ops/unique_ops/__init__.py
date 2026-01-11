"""Deduplication utilities for dataclass batches."""

from __future__ import annotations

from .optimized_unique_ops import unique_mask

# -----------------------------------------------------------------------------
# Performance Benchmark Results (Legacy vs Optimized)
# Refined: Per-trial data generation, Skewed distribution, Multi-field records
# -----------------------------------------------------------------------------
# Size=4096,  Duplication=10% (Skewed) : 1.13x (12.5% faster)
# Size=4096,  Duplication=50% (Skewed) : 1.06x (5.8% faster)
# Size=4096,  Duplication=90% (Skewed) : 0.19x (413.2% slower)
# Size=16384, Duplication=10% (Skewed) : 1.04x (3.6% faster)
# Size=16384, Duplication=50% (Skewed) : 1.00x (0.4% faster)
# Size=16384, Duplication=90% (Skewed) : 1.75x (75.1% faster)
# -----------------------------------------------------------------------------

__all__ = ["unique_mask"]

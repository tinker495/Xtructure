from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict

import jax

SCHEMA_VERSION = "2.0"


@dataclass(slots=True)
class BenchmarkEnvironment:
    device: str
    backend: str
    jax_version: str
    jax_enable_x64: bool
    xla_flags: str
    python_version: str
    platform: str


@dataclass(slots=True)
class BenchmarkRecord:
    name: str
    mode: str
    transfer_policy: str
    params: Dict[str, Any]
    metrics: Dict[str, float]


@dataclass(slots=True)
class BenchmarkResult:
    schema_version: str = SCHEMA_VERSION
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    environment: BenchmarkEnvironment = field(
        default_factory=lambda: capture_environment()
    )
    run: Dict[str, Any] = field(default_factory=dict)
    records: list[BenchmarkRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def capture_environment() -> BenchmarkEnvironment:
    devices = jax.devices()
    device = devices[0].device_kind if devices else "unknown"
    return BenchmarkEnvironment(
        device=device,
        backend=jax.default_backend(),
        jax_version=jax.__version__,
        jax_enable_x64=bool(jax.config.jax_enable_x64),
        xla_flags=os.environ.get("XLA_FLAGS", ""),
        python_version=platform.python_version(),
        platform=platform.platform(),
    )

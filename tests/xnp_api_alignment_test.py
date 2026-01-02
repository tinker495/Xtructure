import difflib
import inspect

import jax.numpy as jnp

from xtructure import numpy as xnp


def _public_jnp_callables() -> set[str]:
    names = set()
    for name in dir(jnp):
        if name.startswith("_"):
            continue
        value = getattr(jnp, name, None)
        if callable(value):
            names.add(name)
    return names


def _signature_shape(sig: inspect.Signature) -> list[tuple[str, inspect._ParameterKind, object]]:
    return [(p.name, p.kind, p.default) for p in sig.parameters.values()]


def test_xnp_no_near_miss_jnp_names():
    jnp_names = _public_jnp_callables()
    xnp_names = set(getattr(xnp, "__all__", []))

    near_misses = []
    for name in xnp_names:
        if name in jnp_names:
            continue
        close = difflib.get_close_matches(name, jnp_names, n=1, cutoff=0.9)
        if close:
            near_misses.append((name, close[0]))

    assert not near_misses, f"xnp API has near-miss names vs jnp: {near_misses}"


def test_xnp_signatures_match_jnp():
    jnp_names = _public_jnp_callables()
    xnp_names = set(getattr(xnp, "__all__", []))

    mismatches = []
    for name in sorted(xnp_names & jnp_names):
        xnp_obj = getattr(xnp, name, None)
        jnp_obj = getattr(jnp, name, None)
        if xnp_obj is None or jnp_obj is None:
            continue
        try:
            xnp_sig = inspect.signature(xnp_obj)
            jnp_sig = inspect.signature(jnp_obj)
        except (TypeError, ValueError):
            continue
        if _signature_shape(xnp_sig) != _signature_shape(jnp_sig):
            diff = "\n".join(
                difflib.unified_diff(
                    [str(jnp_sig)],
                    [str(xnp_sig)],
                    fromfile="jnp",
                    tofile="xnp",
                    lineterm="",
                )
            )
            mismatches.append((name, diff))

    assert not mismatches, f"xnp signatures diverge from jnp: {mismatches}"

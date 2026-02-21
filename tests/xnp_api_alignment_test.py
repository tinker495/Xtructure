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


_ARRAY_PARAM_NAMES = {"a", "A", "arr", "ary", "array", "x"}


def _signature_shape(
    sig: inspect.Signature,
) -> list[tuple[str, inspect._ParameterKind, object]]:
    return [(p.name, p.kind, p.default) for p in sig.parameters.values()]


def _array_name_aligned(jnp_name: str, xnp_name: str) -> bool:
    if jnp_name == xnp_name:
        return True
    if xnp_name == "dataclass_instance" and jnp_name in _ARRAY_PARAM_NAMES:
        return True
    return False


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
        jnp_params = list(jnp_sig.parameters.values())
        xnp_params = list(xnp_sig.parameters.values())
        if len(jnp_params) != len(xnp_params):
            mismatches.append(
                (
                    name,
                    f"parameter count differs: jnp={len(jnp_params)} xnp={len(xnp_params)}",
                )
            )
            continue
        for jnp_param, xnp_param in zip(jnp_params, xnp_params, strict=True):
            if (
                jnp_param.kind != xnp_param.kind
                or jnp_param.default != xnp_param.default
            ):
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
                break
            if not _array_name_aligned(jnp_param.name, xnp_param.name):
                mismatches.append(
                    (
                        name,
                        f"parameter name mismatch: jnp='{jnp_param.name}' xnp='{xnp_param.name}'",
                    )
                )
                break

    assert not mismatches, f"xnp signatures diverge from jnp: {mismatches}"

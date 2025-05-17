import chex


def isnamedtupleinstance(x):
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, "_fields", None)
    if not isinstance(f, tuple):
        return False
    return all(type(n) == str for n in f)


def get_leaf_elements(tree: chex.Array):
    """
    Extracts leaf elements from a nested structure, typically composed of
    namedtuples and JAX arrays at the leaves.

    Args:
        tree: The nested structure (e.g., a namedtuple containing other
              namedtuples or JAX arrays, or a single JAX array).

    Yields:
        Leaf elements (non-namedtuple elements) within the nested structure.
    """
    if isnamedtupleinstance(tree):
        for item in tree:
            yield from get_leaf_elements(item)  # Recursively process sub-tuples
    else:
        yield tree  # Yield the leaf element

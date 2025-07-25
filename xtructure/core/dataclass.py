"""A JAX/dm-tree friendly dataclass implementation based on chex's dataclass, with unnecessary features removed."""

import dataclasses
import functools
import sys

import jax
from absl import logging
from typing_extensions import dataclass_transform  # pytype: disable=not-supported-yet

FrozenInstanceError = dataclasses.FrozenInstanceError
_RESERVED_DCLS_FIELD_NAMES = frozenset(("from_tuple", "replace", "to_tuple"))


@dataclass_transform()
def base_dataclass(
    cls=None,
    *,
    init=True,
    repr=True,  # pylint: disable=redefined-builtin
    eq=True,
    order=False,
    unsafe_hash=False,
    frozen=False,
    kw_only: bool = False,
):
    """JAX-friendly wrapper for :py:func:`dataclasses.dataclass`.

    This wrapper class registers new dataclasses with JAX so that tree utils
    operate correctly. Additionally a replace method is provided making it easy
    to operate on the class when made immutable (frozen=True).

    Args:
        cls: A class to decorate.
        init: See :py:func:`dataclasses.dataclass`.
        repr: See :py:func:`dataclasses.dataclass`.
        eq: See :py:func:`dataclasses.dataclass`.
        order: See :py:func:`dataclasses.dataclass`.
        unsafe_hash: See :py:func:`dataclasses.dataclass`.
        frozen: See :py:func:`dataclasses.dataclass`.
        kw_only: See :py:func:`dataclasses.dataclass`.

    Returns:
        A JAX-friendly dataclass.
    """

    def dcls(cls):
        # Make sure to create a separate _Dataclass instance for each `cls`.
        return _Dataclass(init, repr, eq, order, unsafe_hash, frozen, kw_only)(cls)

    if cls is None:
        return dcls
    return dcls(cls)


class _Dataclass:
    """JAX-friendly wrapper for `dataclasses.dataclass`."""

    def __init__(
        self,
        init=True,
        repr=True,  # pylint: disable=redefined-builtin
        eq=True,
        order=False,
        unsafe_hash=False,
        frozen=False,
        kw_only=False,
    ):
        self.init = init
        self.repr = repr  # pylint: disable=redefined-builtin
        self.eq = eq
        self.order = order
        self.unsafe_hash = unsafe_hash
        self.frozen = frozen
        self.kw_only = kw_only

    def __call__(self, cls):
        """Forwards class to dataclasses's wrapper and registers it with JAX."""

        # Remove once https://github.com/python/cpython/pull/24484 is merged.
        for base in cls.__bases__:
            if (
                dataclasses.is_dataclass(base)
                and getattr(base, "__dataclass_params__").frozen
                and not self.frozen
            ):
                raise TypeError("cannot inherit non-frozen dataclass from a frozen one")

        # `kw_only` is only available starting from 3.10.
        version_dependent_args = {}
        version = sys.version_info
        if version.major == 3 and version.minor >= 10:
            version_dependent_args = {"kw_only": self.kw_only}
        # pytype: disable=wrong-keyword-args
        dcls = dataclasses.dataclass(
            cls,
            init=self.init,
            repr=self.repr,
            eq=self.eq,
            order=self.order,
            unsafe_hash=self.unsafe_hash,
            frozen=self.frozen,
            **version_dependent_args,
        )
        # pytype: enable=wrong-keyword-args

        fields_names = set(f.name for f in dataclasses.fields(dcls))
        invalid_fields = fields_names.intersection(_RESERVED_DCLS_FIELD_NAMES)
        if invalid_fields:
            raise ValueError(
                f"The following dataclass fields are disallowed: " f"{invalid_fields} ({dcls})."
            )

        def _from_tuple(args):
            return dcls(zip(dcls.__dataclass_fields__.keys(), args))

        def _to_tuple(self):
            return tuple(getattr(self, k) for k in self.__dataclass_fields__.keys())

        def _replace(self, **kwargs):
            return dataclasses.replace(self, **kwargs)

        def _getstate(self):
            return self.__dict__

        # Register the dataclass at definition. As long as the dataclass is defined
        # outside __main__, this is sufficient to make JAX's PyTree registry
        # recognize the dataclass and the dataclass' custom PyTreeDef, especially
        # when unpickling either the dataclass object, its type, or its PyTreeDef,
        # in a different process, because the defining module will be imported.
        #
        # However, if the dataclass is defined in __main__, unpickling in a
        # subprocess does not trigger re-registration. Therefore we also need to
        # register when deserializing the object, or construction (e.g. when the
        # dataclass type is being unpickled). Unfortunately, there is not yet a way
        # to trigger re-registration when the treedef is unpickled as that's handled
        # by JAX.
        #
        # See internal dataclass_test for unit tests demonstrating the problems.
        # The registration below may result in pickling failures of the sort
        # _pickle.PicklingError: Can't pickle <functools._lru_cache_wrapper object>:
        # it's not the same object as register_dataclass_type_with_jax_tree_util
        # for modules defined in __main__ so we disable registration in this case.
        if dcls.__module__ != "__main__":
            register_dataclass_type_with_jax_tree_util(dcls)

        # Patch __setstate__ to register the dataclass on deserialization.
        def _setstate(self, state):
            register_dataclass_type_with_jax_tree_util(dcls)
            self.__dict__.update(state)

        orig_init = dcls.__init__

        # Patch __init__ such that the dataclass is registered on creation if it is
        # not registered on deserialization.
        @functools.wraps(orig_init)
        def _init(self, *args, **kwargs):
            register_dataclass_type_with_jax_tree_util(dcls)
            return orig_init(self, *args, **kwargs)

        setattr(dcls, "from_tuple", _from_tuple)
        setattr(dcls, "to_tuple", _to_tuple)
        setattr(dcls, "replace", _replace)
        setattr(dcls, "__getstate__", _getstate)
        setattr(dcls, "__setstate__", _setstate)
        setattr(dcls, "__init__", _init)

        return dcls


def _dataclass_unflatten(dcls, keys, values):
    """Creates a chex dataclass from a flatten jax.tree_util representation."""
    dcls_object = dcls.__new__(dcls)
    attribute_dict = dict(zip(keys, values))
    # Looping over fields instead of keys & values preserves the field order.
    # Using dataclasses.fields fails because dataclass uids change after
    # serialisation (eg, with cloudpickle).
    for field in dcls.__dataclass_fields__.values():
        if field.name in attribute_dict:  # Filter pseudo-fields.
            object.__setattr__(dcls_object, field.name, attribute_dict[field.name])
    # Need to manual call post_init here as we have avoided calling __init__
    if getattr(dcls_object, "__post_init__", None):
        dcls_object.__post_init__()
    return dcls_object


def _flatten_with_path(dcls):
    path = []
    keys = []
    for k, v in sorted(dcls.__dict__.items()):
        keys.append(k)  # generate same aux data as flatten without path
        k = jax.tree_util.GetAttrKey(k)
        path.append((k, v))
    return path, tuple(keys)


@functools.cache
def register_dataclass_type_with_jax_tree_util(data_class):
    """Register an existing dataclass so JAX knows how to handle it.

    This means that functions in jax.tree_util operate over the fields
    of the dataclass. See
    https://jax.readthedocs.io/en/latest/pytrees.html#extending-pytrees
    for further information.

    Args:
        data_class: A class created using dataclasses.dataclass. It must be
            constructable from keyword arguments corresponding to the members exposed
            in instance.__dict__.
    """

    def flatten(d):
        if d.__dict__:
            return tuple(zip(*sorted(d.__dict__.items())))[::-1]
        return ((), ())

    unflatten = functools.partial(_dataclass_unflatten, data_class)
    try:
        jax.tree_util.register_pytree_with_keys(
            nodetype=data_class,
            flatten_with_keys=_flatten_with_path,
            flatten_func=flatten,
            unflatten_func=unflatten,
        )
    except ValueError:
        logging.info("%s is already registered as JAX PyTree node.", data_class)

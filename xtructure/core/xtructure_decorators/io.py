"""Decorator to add save and load methods to xtructure dataclasses."""

from typing import Type, TypeVar

from xtructure import io

T = TypeVar("T")


def add_io_methods(cls: Type[T]) -> Type[T]:
    """
    Augments the class with `save` and `load` methods.

    The `save` method allows an instance to be saved to a file.
    The `load` method allows the class to load an instance from a file.
    """

    def save_method(self, path: str):
        """Saves the instance to a .npz file."""
        return io.save(path, self)

    @classmethod
    def load_method(cls: Type[T], path: str) -> T:
        """Loads an instance from a .npz file."""
        loaded_instance = io.load(path)
        if not isinstance(loaded_instance, cls):
            raise TypeError(
                f"Loaded instance is of type {type(loaded_instance).__name__}, "
                f"but expected {cls.__name__}."
            )
        return loaded_instance

    setattr(cls, "save", save_method)
    setattr(cls, "load", load_method)

    return cls

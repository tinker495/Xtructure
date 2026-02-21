from xtructure.core.type_utils import (
    is_xtructure_dataclass_instance,
    is_xtructure_dataclass_type,
)


class MockXtructedClass:
    is_xtructed = True


class MockNonXtructedClass:
    pass


def test_is_xtructure_dataclass_type():
    assert is_xtructure_dataclass_type(MockXtructedClass) is True
    assert is_xtructure_dataclass_type(MockNonXtructedClass) is False
    assert (
        is_xtructure_dataclass_type(MockXtructedClass()) is False
    )  # instance, not type
    assert (
        is_xtructure_dataclass_type(type("DynamicClass", (), {"is_xtructed": False}))
        is False
    )


def test_is_xtructure_dataclass_instance():
    assert is_xtructure_dataclass_instance(MockXtructedClass()) is True
    assert is_xtructure_dataclass_instance(MockNonXtructedClass()) is False
    # Notice that bool(getattr(MockXtructedClass, "is_xtructed", False)) would be True
    assert is_xtructure_dataclass_instance(MockXtructedClass) is True
    assert is_xtructure_dataclass_instance(123) is False

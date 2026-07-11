"""Small architecture smoke checks for Xtructure.

Detailed layout/decorator rules live in CONTEXT.md.  The test file only keeps
cheap checks that catch accidental file moves without reimplementing a policy
engine in pytest.
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "xtructure"
DECORATOR_DIR = PACKAGE_DIR / "core" / "xtructure_decorators"


def test_layout_entrypoints_exist() -> None:
    assert (PACKAGE_DIR / "core" / "layout" / "type_layout.py").is_file()
    assert (PACKAGE_DIR / "core" / "layout" / "instance_layout.py").is_file()
    assert (PACKAGE_DIR / "core" / "layout" / "bitpack.py").is_file()


def test_decorator_adapters_stay_in_adapter_packages() -> None:
    assert (DECORATOR_DIR / "layout_adapters" / "__init__.py").is_file()
    assert (DECORATOR_DIR / "pytree_adapters" / "__init__.py").is_file()

    forbidden_root_names = {
        "aggregate_bitpack.py",
        "bitpack_accessors.py",
        "default.py",
        "hash.py",
        "indexing.py",
        "io.py",
        "ops.py",
        "shape.py",
        "string_format.py",
        "structure_util.py",
        "validation.py",
    }
    assert not {
        path.name for path in DECORATOR_DIR.glob("*.py") if path.name in forbidden_root_names
    }


def test_size_dtype_defined_once() -> None:
    definitions = []
    for path in PACKAGE_DIR.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        if any(line.lstrip().startswith("SIZE_DTYPE = ") for line in path.read_text().splitlines()):
            definitions.append(path.relative_to(PACKAGE_DIR).as_posix())
    assert definitions == ["core/dtype_facts.py"]

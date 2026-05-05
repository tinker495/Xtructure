"""Architecture guard for the Xtructure Layout seam.

Locks the **Layout Interface** by enforcing the AST-detectable rules from
``CONTEXT.md`` "Architecture guard criteria". Each rule is a parametrized
pytest entry, so a failing rule names itself in the test report.

Coverage notes:

- R1, R3, R4, R5, R6 are AST-detectable and enforced here.
- The CONTEXT.md "FieldDescriptor instance attribute read" item has no sound
  AST translation: layout fact types (FieldLayout, AdapterFieldPlan,
  AggregateLeafLayout, PackedFieldLayout, ...) intentionally share attribute
  names with FieldDescriptor (``.bits``, ``.packed_bits``,
  ``.intrinsic_shape``, ...) so first-party Layout Adapters can consume
  them. AST cannot distinguish the two without flow analysis. R1 already
  locks the entry path (``get_field_descriptors``) through which
  FieldDescriptor instances flow into adapters; the residual semantic case
  (constructing a descriptor and reading its facts back) remains as written
  guidance in CONTEXT.md.
- "Local byte-length math when Bitpack Layout owns the fact" is also
  semantic and stays as written guidance.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
from typing import Callable, Iterator

import pytest

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "xtructure"
PROJECT_DIR = PACKAGE_DIR.parent


@dataclasses.dataclass(frozen=True)
class _Violation:
    file: Path
    line: int
    col: int
    detail: str

    def render(self, rule_name: str, remediation: str) -> str:
        rel = self.file.relative_to(PROJECT_DIR)
        location = (
            f"{rel}:{self.line}:{self.col}"  # noqa: E231 — flake8 false-positive on f-string ":"
        )
        return f"{location} violates {rule_name}\n  {self.detail}\n  Fix: {remediation}"


@dataclasses.dataclass(frozen=True)
class _Rule:
    name: str
    description: str
    excluded_relpaths: tuple[str, ...]
    check: Callable[[Path, ast.Module], Iterator[_Violation]]
    remediation: str

    def is_excluded(self, path: Path) -> bool:
        rel = path.relative_to(PACKAGE_DIR).as_posix()
        for ex in self.excluded_relpaths:
            ex = ex.rstrip("/")
            if rel == ex or rel.startswith(ex + "/"):
                return True
        return False


def _iter_in_scope_files(rule: _Rule) -> Iterator[Path]:
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if rule.is_excluded(path):
            continue
        yield path


def _first_violation(rule: _Rule) -> _Violation | None:
    for path in _iter_in_scope_files(rule):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:  # pragma: no cover
            pytest.fail(f"Failed to parse {path}: {exc}")
        for v in rule.check(path, tree):
            return v
    return None


# ----- Rule check implementations -----------------------------------------


def _check_r1_get_field_descriptors_call(path: Path, tree: ast.Module) -> Iterator[_Violation]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (isinstance(func, ast.Name) and func.id == "get_field_descriptors") or (
            isinstance(func, ast.Attribute) and func.attr == "get_field_descriptors"
        ):
            yield _Violation(
                path,
                node.lineno,
                node.col_offset,
                "Call to get_field_descriptors() — only the Layout module may "
                "interpret Xtructure Schema.",
            )


_FORBIDDEN_DC_FIELD_ATTRS = frozenset({"type", "metadata", "default", "default_factory"})


def _iter_is_dc_fields_call(expr: ast.expr, has_bare_fields: bool) -> bool:
    if not isinstance(expr, ast.Call):
        return False
    func = expr.func
    # dataclasses.fields(...)
    if (
        isinstance(func, ast.Attribute)
        and func.attr == "fields"
        and isinstance(func.value, ast.Name)
        and func.value.id == "dataclasses"
    ):
        return True
    # bare fields(...) — only when `from dataclasses import fields`
    if has_bare_fields and isinstance(func, ast.Name) and func.id == "fields":
        return True
    # <expr>.__dataclass_fields__.values()
    if (
        isinstance(func, ast.Attribute)
        and func.attr == "values"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "__dataclass_fields__"
    ):
        return True
    return False


def _file_imports_dataclasses_fields(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "dataclasses":
            for alias in node.names:
                if alias.name == "fields":
                    return True
    return False


class _DataclassFieldsLoopVisitor(ast.NodeVisitor):
    """Detect ``.type``/``.metadata``/``.default``/``.default_factory`` reads
    on the loop variable of ``for X in dataclasses.fields(...)`` or
    ``for X in <expr>.__dataclass_fields__.values()``.
    """

    def __init__(self, path: Path, has_bare_fields: bool) -> None:
        self.path = path
        self.has_bare_fields = has_bare_fields
        self.violations: list[_Violation] = []
        self._loop_targets: list[set[str]] = []

    def visit_For(self, node: ast.For) -> None:
        if _iter_is_dc_fields_call(node.iter, self.has_bare_fields):
            scope: set[str] = set()
            target = node.target
            if isinstance(target, ast.Name):
                scope.add(target.id)
            elif isinstance(target, ast.Tuple):
                scope.update(n.id for n in target.elts if isinstance(n, ast.Name))
            self._loop_targets.append(scope)
            self.generic_visit(node)
            self._loop_targets.pop()
        else:
            self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name)
            and node.attr in _FORBIDDEN_DC_FIELD_ATTRS
        ):
            for scope in self._loop_targets:
                if node.value.id in scope:
                    self.violations.append(
                        _Violation(
                            self.path,
                            node.lineno,
                            node.col_offset,
                            f"Read of `.{node.attr}` on a "
                            "dataclasses.fields() / "
                            "__dataclass_fields__.values() loop target — "
                            "schema facts must come from get_type_layout().",
                        )
                    )
                    break
        self.generic_visit(node)


def _check_r3_dataclass_field_metadata_read(path: Path, tree: ast.Module) -> Iterator[_Violation]:
    has_bare_fields = _file_imports_dataclasses_fields(tree)
    visitor = _DataclassFieldsLoopVisitor(path, has_bare_fields)
    visitor.visit(tree)
    yield from visitor.violations


def _check_r4_dataclasses_fields_call(path: Path, tree: ast.Module) -> Iterator[_Violation]:
    has_bare_fields = _file_imports_dataclasses_fields(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_qualified = (
            isinstance(func, ast.Attribute)
            and func.attr == "fields"
            and isinstance(func.value, ast.Name)
            and func.value.id == "dataclasses"
        )
        is_bare = has_bare_fields and isinstance(func, ast.Name) and func.id == "fields"
        if is_qualified or is_bare:
            yield _Violation(
                path,
                node.lineno,
                node.col_offset,
                "Call to dataclasses.fields() — only the Layout module and "
                "core/dataclass.py may use it.",
            )


def _check_r5_bitpack_math_import(path: Path, tree: ast.Module) -> Iterator[_Violation]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.endswith("bitpack_math") or "bitpack_math" in alias.name:
                    yield _Violation(
                        path,
                        node.lineno,
                        node.col_offset,
                        f"Import of `{alias.name}` — bitpack_math is restricted "
                        "to layout, field_descriptors, and io/bitpack.",
                    )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.endswith("bitpack_math") or "bitpack_math" in mod:
                yield _Violation(
                    path,
                    node.lineno,
                    node.col_offset,
                    f"Import from `{mod}` — bitpack_math is restricted to "
                    "layout, field_descriptors, and io/bitpack.",
                )


_LAYOUT_FACT_TYPES = frozenset(
    {
        "FieldLayout",
        "LeafLayout",
        "AdapterFieldPlan",
        "PackedFieldLayout",
        "AggregateLeafLayout",
        "AggregateViewFieldLayout",
        "AggregateBitpackLayout",
        "TypeLayout",
        "InstanceFieldLayout",
        "InstanceLayout",
    }
)


def _check_r6_layout_fact_construction(path: Path, tree: ast.Module) -> Iterator[_Violation]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name in _LAYOUT_FACT_TYPES:
            yield _Violation(
                path,
                node.lineno,
                node.col_offset,
                f"Construction of layout fact type `{name}()` — facts are "
                "produced by get_type_layout()/get_instance_layout(); never "
                "authored at the call site.",
            )


# ----- Rule registry ------------------------------------------------------


_ALL_RULES: tuple[_Rule, ...] = (
    _Rule(
        name="R1_no_get_field_descriptors_outside_layout",
        description=(
            "get_field_descriptors() may only be called from inside the " "Layout module."
        ),
        excluded_relpaths=(
            "core/layout",
            "core/field_descriptors.py",
            "core/field_descriptor_utils.py",
        ),
        check=_check_r1_get_field_descriptors_call,
        remediation=("Use get_type_layout(cls) and read facts from the returned " "TypeLayout."),
    ),
    _Rule(
        name="R3_no_dataclass_field_metadata_read_outside_layout",
        description=(
            "Reading .type/.metadata/.default/.default_factory on "
            "dataclasses.fields() or __dataclass_fields__.values() loop "
            "targets is forbidden outside Layout."
        ),
        excluded_relpaths=("core/layout",),
        check=_check_r3_dataclass_field_metadata_read,
        remediation=(
            "Use get_type_layout(cls); per-field facts live on FieldLayout " "and AdapterFieldPlan."
        ),
    ),
    _Rule(
        name="R4_no_dataclasses_fields_call_outside_allowed",
        description=(
            "dataclasses.fields(...) may only be called from Layout or " "core/dataclass.py."
        ),
        excluded_relpaths=("core/layout", "core/dataclass.py"),
        check=_check_r4_dataclasses_fields_call,
        remediation=(
            "Use get_type_layout(cls).field_names or "
            "self.__dataclass_fields__.keys() for value-level field-order "
            "traversal (Value Traversal)."
        ),
    ),
    _Rule(
        name="R5_no_bitpack_math_import_outside_allowed",
        description=(
            "xtructure.core.bitpack_math is restricted to layout, "
            "field_descriptors, and io/bitpack."
        ),
        excluded_relpaths=(
            "core/layout",
            "core/field_descriptors.py",
            "io/bitpack.py",
        ),
        check=_check_r5_bitpack_math_import,
        remediation=("Read packed_byte_count from PackedFieldLayout (Bitpack Layout)."),
    ),
    _Rule(
        name="R6_no_layout_fact_construction_outside_layout",
        description=(
            "Layout fact types (FieldLayout, AdapterFieldPlan, ...) are "
            "produced by the Layout module; constructing them at the call "
            "site bypasses the seam."
        ),
        excluded_relpaths=("core/layout",),
        check=_check_r6_layout_fact_construction,
        remediation=("Use get_type_layout(cls) / get_instance_layout(instance)."),
    ),
)


# ----- Pytest entries -----------------------------------------------------


@pytest.mark.parametrize("rule", _ALL_RULES, ids=[r.name for r in _ALL_RULES])
def test_layout_seam_rule(rule: _Rule) -> None:
    violation = _first_violation(rule)
    if violation is not None:
        pytest.fail(violation.render(rule.name, rule.remediation))


def test_architecture_guard_walks_xtructure_package() -> None:
    """Sanity: the rule walker reaches the xtructure package files."""
    sentinel = _Rule(
        name="sentinel",
        description="",
        excluded_relpaths=(),
        check=lambda path, tree: iter(()),
        remediation="",
    )
    files = list(_iter_in_scope_files(sentinel))
    assert any(f.name == "__init__.py" for f in files), files
    assert any(
        f.relative_to(PACKAGE_DIR).as_posix().startswith("core/layout/") for f in files
    ), files
    assert any(
        f.relative_to(PACKAGE_DIR).as_posix().startswith("core/xtructure_decorators/")
        for f in files
    ), files


def test_architecture_guard_excludes_layout_for_r1() -> None:
    """Sanity: scope exclusion actually drops the layout module."""
    rule = next(r for r in _ALL_RULES if r.name.startswith("R1_"))
    files = [f.relative_to(PACKAGE_DIR).as_posix() for f in _iter_in_scope_files(rule)]
    assert all(not f.startswith("core/layout/") for f in files), files

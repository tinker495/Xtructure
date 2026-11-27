#!/usr/bin/env python3
"""AST-based seam-leakage guard for standalone Xtructure CI."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SOURCE_ROOTS = ("xtructure", "xtructure_benchmarks")

SANCTIONED_XTRUCTURE_SEAMS = frozenset(
    {
        "xtructure",
        "xtructure.numpy",
        "xtructure.xtructure_numpy",
        "xtructure.io",
        "xtructure.core.layout",
        "xtructure.core.display",
        "xtructure.core.packing",
    }
)

UNAPPROVED_XTRUCTURE_CHILDREN = frozenset({"core", "hashtable", "bgpq", "queue", "stack"})

EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".venv",
    ".git",
    "build",
    "dist",
    "runs",
    "docs",
    "tests",
    "scripts",
}


@dataclass(frozen=True)
class SeamViolation:
    path: str
    lineno: int
    kind: str
    raw_import: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "lineno": self.lineno,
            "kind": self.kind,
            "raw_import": self.raw_import,
            "reason": self.reason,
        }


def iter_source_files(root: Path) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for top_pkg in SOURCE_ROOTS:
        base = root / top_pkg
        candidates: Iterable[Path] = base.rglob("*.py") if base.is_dir() else ()
        for path in candidates:
            rel = path.relative_to(root)
            if any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in rel.parts):
                continue
            if path.name.startswith("test_") or path.name.endswith("_test.py"):
                continue
            if path not in seen:
                files.append((top_pkg, path))
                seen.add(path)
    return sorted(files, key=lambda item: item[1].relative_to(root).as_posix())


def _is_private_name(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _has_private_segment(dotted_name: str) -> bool:
    return any(_is_private_name(part) for part in dotted_name.split(".") if part)


def _display_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _infer_source_top(root: Path, path: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return "xtructure_benchmarks"
    if rel.parts and rel.parts[0] in SOURCE_ROOTS:
        return rel.parts[0]
    return "xtructure_benchmarks"


def _is_submodule(root: Path, dotted_name: str) -> bool:
    target = root.joinpath(*dotted_name.split("."))
    return target.is_dir() or target.with_suffix(".py").is_file()


def check_file_for_leaks(root: Path, path: Path, source_top: str) -> list[SeamViolation]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    rel_path = _display_path(root, path)
    violations: list[SeamViolation] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            raw = (ast.get_source_segment(source, node) or "").strip()
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in SOURCE_ROOTS or top == source_top:
                    continue
                if source_top == "xtructure" and top == "xtructure_benchmarks":
                    violations.append(
                        SeamViolation(
                            path=rel_path,
                            lineno=node.lineno,
                            kind="reverse-package-dependency",
                            raw_import=raw,
                            reason="`xtructure` must not import `xtructure_benchmarks`.",
                        )
                    )
                    continue
                if _has_private_segment(alias.name):
                    violations.append(
                        SeamViolation(
                            path=rel_path,
                            lineno=node.lineno,
                            kind="private-cross-package-import",
                            raw_import=raw,
                            reason=(
                                f"Private import `{alias.name}` from `xtructure`; "
                                "use a public seam instead."
                            ),
                        )
                    )
                elif alias.name not in SANCTIONED_XTRUCTURE_SEAMS:
                    violations.append(
                        SeamViolation(
                            path=rel_path,
                            lineno=node.lineno,
                            kind="internal-cross-package-seam",
                            raw_import=raw,
                            reason=(
                                f"Internal import `{alias.name}` from `xtructure`; "
                                "use a sanctioned public seam."
                            ),
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module:
                continue
            mod = node.module
            top = mod.split(".")[0]
            if top not in SOURCE_ROOTS or top == source_top:
                continue
            raw = (ast.get_source_segment(source, node) or "").strip()
            if source_top == "xtructure" and top == "xtructure_benchmarks":
                violations.append(
                    SeamViolation(
                        path=rel_path,
                        lineno=node.lineno,
                        kind="reverse-package-dependency",
                        raw_import=raw,
                        reason="`xtructure` must not import `xtructure_benchmarks`.",
                    )
                )
                continue
            if _has_private_segment(mod) or any(
                _is_private_name(alias.name) for alias in node.names if alias.name != "*"
            ):
                violations.append(
                    SeamViolation(
                        path=rel_path,
                        lineno=node.lineno,
                        kind="private-cross-package-import",
                        raw_import=raw,
                        reason="Private symbol or module imported from `xtructure`.",
                    )
                )
                continue
            invalid_targets: list[str] = []
            for alias in node.names:
                if alias.name == "*":
                    if mod not in SANCTIONED_XTRUCTURE_SEAMS and mod not in invalid_targets:
                        invalid_targets.append(mod)
                    continue
                candidate = f"{mod}.{alias.name}"
                is_unapproved_child = (
                    mod == "xtructure" and alias.name in UNAPPROVED_XTRUCTURE_CHILDREN
                )
                if is_unapproved_child:
                    if candidate not in invalid_targets:
                        invalid_targets.append(candidate)
                    continue
                if candidate in SANCTIONED_XTRUCTURE_SEAMS:
                    continue
                if _is_submodule(root, candidate):
                    if candidate not in invalid_targets:
                        invalid_targets.append(candidate)
                    continue
                if mod not in SANCTIONED_XTRUCTURE_SEAMS and mod not in invalid_targets:
                    invalid_targets.append(mod)

            for invalid_target in invalid_targets:
                violations.append(
                    SeamViolation(
                        path=rel_path,
                        lineno=node.lineno,
                        kind="internal-cross-package-seam",
                        raw_import=raw,
                        reason=(
                            f"Unsanctioned internal seam `{invalid_target}` imported from "
                            "`xtructure`; expose it on the public `xtructure` seam instead."
                        ),
                    )
                )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root (default: current directory)")
    parser.add_argument(
        "--check-file",
        type=Path,
        default=None,
        help="Check a single Python file path instead of scanning the repository",
    )
    parser.add_argument(
        "--source-top",
        choices=SOURCE_ROOTS,
        default=None,
        help="Caller top-level package for --check-file (default: inferred from file path)",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    violations: list[SeamViolation] = []
    if args.check_file is not None:
        target_file = args.check_file.resolve()
        files_checked = 1
        source_top = args.source_top or _infer_source_top(root, target_file)
        violations.extend(check_file_for_leaks(root, target_file, source_top))
    else:
        files = iter_source_files(root)
        files_checked = len(files)
        for source_top, path in files:
            violations.extend(check_file_for_leaks(root, path, source_top))

    if args.json:
        print(
            json.dumps(
                {
                    "failed": bool(violations),
                    "files_checked": files_checked,
                    "violation_count": len(violations),
                    "violations": [v.to_dict() for v in violations],
                },
                indent=2,
            )
        )
    else:
        print(f"Files checked: {files_checked}, Seam violations: {len(violations)}")
        for v in violations:
            print(f"  {v.path}:{v.lineno} [{v.kind}] {v.raw_import}\n    -> {v.reason}")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())

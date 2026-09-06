from pathlib import Path

import pytest
import tyro

from xtructure_benchmarks import (
    benchmark_hashtable,
    benchmark_heap,
    benchmark_queue,
    benchmark_stack,
    run_all,
)
from xtructure_benchmarks.common import parse_batch_sizes


def test_batch_size_parser_preserves_empty_input_distinction():
    assert parse_batch_sizes("") is None
    assert parse_batch_sizes(",") == []


@pytest.mark.parametrize(
    "module",
    [benchmark_hashtable, benchmark_queue, benchmark_stack],
)
def test_benchmark_cli_forwards_options(monkeypatch, module):
    calls = []
    monkeypatch.setattr(module, "run_benchmarks", lambda **kwargs: calls.append(kwargs))

    tyro.cli(module.main, args=["--mode", "e2e", "--trials", "3", "--batch-sizes", "8,16"])

    assert calls == [{"mode": "e2e", "trials": 3, "batch_sizes": [8, 16]}]


def test_heap_cli_forwards_options(monkeypatch):
    calls = []
    monkeypatch.setattr(benchmark_heap, "run_benchmarks", lambda **kwargs: calls.append(kwargs))

    tyro.cli(
        benchmark_heap.main,
        args=[
            "--mode",
            "e2e",
            "--trials",
            "3",
            "--batch-sizes",
            "8,16",
            "--python-heap-insert-mode",
            "bulk",
        ],
    )

    assert calls == [
        {
            "mode": "e2e",
            "trials": 3,
            "batch_sizes": [8, 16],
            "python_heap_insert_mode": "bulk",
        }
    ]


@pytest.mark.parametrize(
    ("module", "expected"),
    [
        (benchmark_hashtable, {"mode": "kernel", "trials": 10, "batch_sizes": None}),
        (benchmark_queue, {"mode": "kernel", "trials": 10, "batch_sizes": None}),
        (benchmark_stack, {"mode": "kernel", "trials": 10, "batch_sizes": None}),
        (
            benchmark_heap,
            {
                "mode": "kernel",
                "trials": 10,
                "batch_sizes": None,
                "python_heap_insert_mode": "auto",
            },
        ),
    ],
)
def test_benchmark_cli_defaults(monkeypatch, module, expected):
    calls = []
    monkeypatch.setattr(module, "run_benchmarks", lambda **kwargs: calls.append(kwargs))

    tyro.cli(module.main, args=[])

    assert calls == [expected]


def test_run_all_cli_forwards_common_and_heap_options(monkeypatch):
    calls = []
    monkeypatch.setattr(
        run_all,
        "run_script",
        lambda path, extra_args=None: calls.append((path, extra_args)),
    )
    monkeypatch.setattr(run_all, "display_summary_table", lambda: None)

    tyro.cli(
        run_all.main,
        args=[
            "--mode",
            "kernel",
            "--trials",
            "2",
            "--batch-sizes",
            "8,16",
            "--python-heap-insert-mode",
            "incremental",
            "--use-float32",
        ],
    )

    benchmark_calls = [(path, args) for path, args in calls if path.name.startswith("benchmark_")]
    assert len(benchmark_calls) == 4
    for path, args in benchmark_calls:
        assert args[:6] == [
            "--mode",
            "kernel",
            "--trials",
            "2",
            "--batch-sizes",
            "8,16",
        ]
        if path.name == "benchmark_heap.py":
            assert args[-2:] == ["--python-heap-insert-mode", "incremental"]

    assert calls[-1] == (Path(run_all.__file__).parent / "visualize.py", None)

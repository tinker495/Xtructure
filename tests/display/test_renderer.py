import jax.numpy as jnp

from tests.dataclass.fixtures import SimpleData
from xtructure.core.display import BatchedRenderer, RichBackend
from xtructure.core.display.renderer import BatchedRenderer as _BatchedRendererCls


class FakeBackend:
    """Records calls for verifying BatchedRenderer's invocation policy."""

    def __init__(self):
        self.calls = []

    def cell(self, content):
        self.calls.append(("cell", content))
        return ("CELL", content)

    def ellipsis_cell(self):
        self.calls.append(("ellipsis",))
        return ("ELLIPSIS",)

    def row(self, cells):
        cells = tuple(cells)
        self.calls.append(("row", cells))
        return ("ROW", cells)

    def grid(self, rows):
        rows = tuple(rows)
        self.calls.append(("grid", rows))
        return ("GRID", rows)

    def frame(self, grid, *, title, subtitle):
        self.calls.append(("frame", grid, title, subtitle))
        return ("FRAME", grid, title, subtitle)

    def to_str(self, frame):
        self.calls.append(("to_str", frame))
        return "OK"


def _label_formatter(item, **_kwargs):
    """Stable string label so slicing can be verified through cell contents."""
    return f"id={int(item.id)}"


def _count(backend, kind):
    return sum(1 for entry in backend.calls if entry[0] == kind)


def test_truncate_indices_no_truncation():
    assert _BatchedRendererCls._truncate_indices(3, 2) == [0, 1, 2]
    assert _BatchedRendererCls._truncate_indices(4, 2) == [0, 1, 2, 3]


def test_truncate_indices_inserts_none_for_ellipsis():
    assert _BatchedRendererCls._truncate_indices(5, 2) == [0, 1, None, 3, 4]
    assert _BatchedRendererCls._truncate_indices(10, 3) == [0, 1, 2, None, 7, 8, 9]


def test_1d_batch_no_truncation_renders_all_cells():
    backend = FakeBackend()
    renderer = BatchedRenderer(_label_formatter, backend)
    instance = SimpleData.default(shape=(3,))

    out = renderer.render(instance, max_size=10, show_size=2, title="T")

    assert out == "OK"
    assert _count(backend, "cell") == 3
    assert _count(backend, "ellipsis") == 0
    assert _count(backend, "row") == 1


def test_1d_batch_truncation_inserts_one_ellipsis():
    backend = FakeBackend()
    renderer = BatchedRenderer(_label_formatter, backend)
    instance = SimpleData.default(shape=(8,))

    renderer.render(instance, max_size=4, show_size=2, title="T")

    assert _count(backend, "cell") == 4
    assert _count(backend, "ellipsis") == 1
    assert _count(backend, "row") == 1


def test_2d_batch_no_truncation():
    backend = FakeBackend()
    renderer = BatchedRenderer(_label_formatter, backend)
    instance = SimpleData.default(shape=(3, 3))

    renderer.render(instance, max_size=999, show_size=2, title="T")

    assert _count(backend, "cell") == 9
    assert _count(backend, "ellipsis") == 0
    assert _count(backend, "row") == 3


def test_2d_batch_row_and_column_truncation():
    backend = FakeBackend()
    renderer = BatchedRenderer(_label_formatter, backend)
    instance = SimpleData.default(shape=(5, 5))

    renderer.render(instance, max_size=999, show_size=2, title="T")

    # 4 non-None rows × 4 non-None cols = 16 real cells
    assert _count(backend, "cell") == 16
    # 4 non-None rows × 1 col-ellipsis + 1 ellipsis-row × 5 cells = 9 ellipsis
    assert _count(backend, "ellipsis") == 9
    # 2 prefix + 1 ellipsis-row + 2 suffix
    assert _count(backend, "row") == 5


def test_frame_receives_title_and_subtitle_with_shape():
    backend = FakeBackend()
    renderer = BatchedRenderer(_label_formatter, backend)
    instance = SimpleData.default(shape=(2,))

    renderer.render(instance, max_size=10, show_size=2, title="MyTitle")

    frame_record = next(c for c in backend.calls if c[0] == "frame")
    _, _grid, title, subtitle = frame_record
    assert title == "MyTitle"
    assert subtitle == "shape: (2,)"


def test_cell_content_comes_from_single_formatter():
    backend = FakeBackend()
    renderer = BatchedRenderer(_label_formatter, backend)
    instance = SimpleData.default(shape=(3,))

    renderer.render(instance, max_size=10, show_size=2, title="T")

    cell_contents = [entry[1] for entry in backend.calls if entry[0] == "cell"]
    expected = [f"id={int(instance[i].id)}" for i in range(3)]
    assert cell_contents == expected


def test_formatter_kwargs_pass_through_to_single_formatter():
    received = {}

    def capturing_formatter(item, **kwargs):
        received.update(kwargs)
        return "x"

    backend = FakeBackend()
    renderer = BatchedRenderer(capturing_formatter, backend)
    instance = SimpleData.default(shape=(2,))

    renderer.render(instance, max_size=10, show_size=2, title="T", extra="hello")

    assert received == {"extra": "hello"}


def test_rich_backend_produces_non_empty_string_with_title():
    renderer = BatchedRenderer(_label_formatter, RichBackend())
    instance = SimpleData.default(shape=(3,))

    out = renderer.render(instance, max_size=10, show_size=2, title="MyClass")

    assert isinstance(out, str)
    assert "MyClass" in out
    assert "shape" in out


def test_decorator_str_dispatches_on_structured_type():
    single = SimpleData.default()
    batched = SimpleData.default(shape=(3,))
    unstructured = SimpleData(id=jnp.array(1), value=jnp.array([2.0, 3.0, 4.0]))

    assert "SimpleData" in str(single)
    assert "Batched SimpleData" in str(batched)
    assert "Unstructured SimpleData" in str(unstructured)


def test_decorator_str_and_dunder_str_are_same_callable():
    assert SimpleData.__str__ is SimpleData.str


def test_decorator_caller_can_override_truncation_thresholds(monkeypatch):
    """``instance.str(max_size=..., show_size=...)`` flows into the renderer."""
    captured = []
    real_render = _BatchedRendererCls.render

    def capturing_render(self, instance, *, max_size, show_size, title, **kwargs):
        captured.append((max_size, show_size))
        return real_render(
            self,
            instance,
            max_size=max_size,
            show_size=show_size,
            title=title,
            **kwargs,
        )

    monkeypatch.setattr(_BatchedRendererCls, "render", capturing_render)

    instance = SimpleData.default(shape=(3,))
    str(instance)
    # MAX_PRINT_BATCH_SIZE, SHOW_BATCH_SIZE module defaults
    assert captured == [(4, 2)]

    captured.clear()
    instance.str(max_size=100, show_size=50)
    assert captured == [(100, 50)]

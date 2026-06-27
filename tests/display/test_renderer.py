import jax.numpy as jnp

from tests.dataclass.fixtures import SimpleData
from xtructure.core.display import BatchedRenderer
from xtructure.core.display.renderer import BatchedRenderer as _BatchedRendererCls


def _simple_batch(shape):
    total = 1
    for size in shape:
        total *= size
    ids = jnp.arange(total, dtype=jnp.uint32).reshape(shape)
    return SimpleData(id=ids, value=ids.astype(jnp.float32))


def _label_formatter(item, **_kwargs):
    """Stable string label so slicing can be verified through output text."""
    return f"id={int(item.id)}"


def _render(instance, *, max_size=10, show_size=2, title="T"):
    renderer = BatchedRenderer(_label_formatter)
    return renderer.render(instance, max_size=max_size, show_size=show_size, title=title)


def test_truncate_indices_no_truncation():
    assert _BatchedRendererCls._truncate_indices(3, 2) == [0, 1, 2]
    assert _BatchedRendererCls._truncate_indices(4, 2) == [0, 1, 2, 3]


def test_truncate_indices_inserts_none_for_ellipsis():
    assert _BatchedRendererCls._truncate_indices(5, 2) == [0, 1, None, 3, 4]
    assert _BatchedRendererCls._truncate_indices(10, 3) == [0, 1, 2, None, 7, 8, 9]


def test_1d_batch_no_truncation_renders_all_cells():
    out = _render(_simple_batch((3,)), max_size=10, show_size=2)

    assert "id=0" in out
    assert "id=1" in out
    assert "id=2" in out
    assert "..." not in out


def test_1d_batch_truncation_inserts_one_ellipsis():
    out = _render(_simple_batch((8,)), max_size=4, show_size=2)

    assert "id=0" in out
    assert "id=1" in out
    assert "id=6" in out
    assert "id=7" in out
    assert "id=2" not in out
    assert "..." in out


def test_2d_batch_no_truncation():
    out = _render(_simple_batch((3, 3)), max_size=999, show_size=2)

    for i in range(9):
        assert f"id={i}" in out
    assert "..." not in out


def test_2d_batch_row_and_column_truncation():
    out = _render(_simple_batch((5, 5)), max_size=999, show_size=2)

    assert "id=0" in out
    assert "id=24" in out
    assert "id=12" not in out
    assert "..." in out


def test_frame_includes_title_and_shape_subtitle():
    out = _render(_simple_batch((2,)), max_size=10, show_size=2, title="MyTitle")

    assert "MyTitle" in out
    assert "shape: (2,)" in out


def test_cell_content_comes_from_single_formatter():
    instance = _simple_batch((3,))

    out = _render(instance, max_size=10, show_size=2)

    expected = [f"id={int(instance[i].id)}" for i in range(3)]
    for label in expected:
        assert label in out


def test_formatter_kwargs_pass_through_to_single_formatter():
    received = {}

    def capturing_formatter(item, **kwargs):
        received.update(kwargs)
        return "x"

    renderer = BatchedRenderer(capturing_formatter)
    instance = SimpleData.default(shape=(2,))

    renderer.render(instance, max_size=10, show_size=2, title="T", extra="hello")

    assert received == {"extra": "hello"}


def test_renderer_produces_non_empty_string_with_title():
    out = _render(_simple_batch((3,)), max_size=10, show_size=2, title="MyClass")

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

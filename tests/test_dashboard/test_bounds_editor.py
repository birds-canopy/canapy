# Licence: BSD-3-Clause
"""The bounds editor: what it reads off the box, and what it warns about."""
import types

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from dashboard.view.bounds_editor import BoundsEditor

SR = 44100


@pytest.fixture()
def song(tmp_path):
    rng = np.random.default_rng(0)
    t = np.arange(0, 3.0, 1 / SR)
    y = rng.normal(0, 1e-3, len(t))
    for start in (0.5, 1.5, 2.2):
        s, e = int(start * SR), int((start + 0.3) * SR)
        y[s:e] += np.sin(2 * np.pi * 4000 * t[s:e])
    path = tmp_path / "song.wav"
    sf.write(str(path), y, SR)
    return str(path)


def _row(song, onset, offset, label="A"):
    return next(pd.DataFrame([{
        "label": label, "onset_s": onset, "offset_s": offset,
        "notated_path": song,
    }]).itertuples())


def _editor(song, config, onset=1.5, offset=1.8, neighbours=()):
    return BoundsEditor(
        _row(song, onset, offset), "k", on_validate=lambda *a: None,
        on_cancel=lambda: None, config=config, neighbours=list(neighbours),
    )


def test_the_box_opens_on_the_current_bounds(song, config):
    editor = _editor(song, config)
    assert editor.bounds() == pytest.approx((1.5, 1.8), abs=1e-6)


def test_the_window_leaves_room_on_both_sides(song, config):
    editor = _editor(song, config)
    t0, t1 = editor.window
    assert t0 < 1.5 and t1 > 1.8


def test_a_long_segment_still_gets_room(song, config):
    # the thumbnail fits a fixed window, which a long segment fills entirely
    editor = _editor(song, config, onset=0.2, offset=2.8)
    t0, t1 = editor.window
    assert t0 < 0.2 and t1 > 2.8


def _drag(editor, onset, offset):
    """What the JS leaves behind when the mouse is released."""
    editor.committed.data = dict(onset=[onset], offset=[offset])


def test_dragging_a_handle_moves_that_boundary(song, config):
    editor = _editor(song, config)
    _drag(editor, 1.42, 1.8)
    assert editor.bounds() == pytest.approx((1.42, 1.8), abs=1e-6)


def test_the_handles_stay_on_one_line(song, config):
    editor = _editor(song, config)
    assert len(set(editor.handles.data["y"])) == 1


def test_the_three_handles_are_onset_middle_and_offset(song, config):
    editor = _editor(song, config)
    assert editor.handles.data["x"] == pytest.approx([1.5, 1.65, 1.8])


def test_boundaries_cannot_leave_the_window(song, config):
    editor = _editor(song, config)
    t0, t1 = editor.window
    _drag(editor, t0 - 5.0, t1 + 5.0)
    assert editor.bounds() == pytest.approx((t0, t1), abs=1e-6)


def test_handles_dragged_past_each_other_stay_ordered(song, config):
    editor = _editor(song, config)
    _drag(editor, 1.9, 1.6)
    onset, offset = editor.bounds()
    assert onset < offset


def test_an_empty_commit_falls_back_to_the_original_bounds(song, config):
    editor = _editor(song, config)
    editor.committed.data = dict(onset=[], offset=[])
    assert editor.bounds() == pytest.approx((1.5, 1.8), abs=1e-6)


def test_the_drag_redraws_while_the_mouse_moves(song, config):
    # the live redraw is what PointDrawTool did not give: assert the pan
    # callback is attached and updates the box, not just the release one
    editor = _editor(song, config)
    events = editor.figure.js_event_callbacks
    assert {"panstart", "pan", "panend"} <= set(events)
    assert "box.data" in events["pan"][0].code


def test_no_vertical_coordinate_is_ever_read(song, config):
    # the drag only ever computes from cb_obj.x, so y cannot move
    editor = _editor(song, config)
    events = editor.figure.js_event_callbacks
    assert "cb_obj.y" not in events["pan"][0].code
    assert "cb_obj.y" not in events["panstart"][0].code


def test_the_figure_has_no_pan_tool_to_fight_the_drag(song, config):
    editor = _editor(song, config)
    assert not any(type(t).__name__ == "PanTool" for t in editor.figure.tools)


def test_no_warning_when_the_box_clears_its_neighbours(song, config):
    editor = _editor(song, config, neighbours=[_row(song, 0.5, 0.8, "B")])
    assert editor.warning.object == ""


def test_crossing_a_neighbour_says_it_will_be_trimmed(song, config):
    editor = _editor(song, config, neighbours=[_row(song, 1.7, 2.0, "B")])
    editor._refresh_readout(1.5, 1.9)
    assert "Overlaps B" in editor.warning.object
    assert "takes precedence" in editor.warning.object


def test_the_editor_is_laid_out_as_a_modal(song, config):
    editor = _editor(song, config)
    assert "bounds-modal" in editor.layout.css_classes


def test_a_crossing_is_warned_about_but_not_refused(song, config):
    captured = []
    editor = BoundsEditor(
        _row(song, 1.5, 1.8), "k", on_validate=lambda k, b: captured.append(b),
        on_cancel=lambda: None, config=config,
        neighbours=[_row(song, 1.7, 2.0, "B")],
    )
    _drag(editor, 1.5, 1.9)

    editor._validate(None)

    assert captured and captured[0] == pytest.approx((1.5, 1.9), abs=1e-6)


def test_an_inverted_box_is_refused(song, config):
    captured = []
    editor = BoundsEditor(
        _row(song, 1.5, 1.8), "k", on_validate=lambda k, b: captured.append(b),
        on_cancel=lambda: None, config=config,
    )
    _drag(editor, 1.65, 1.65)

    editor._validate(None)

    assert captured == []
    assert "cannot end before it starts" in editor.warning.object


def test_the_readout_shows_the_times(song, config):
    editor = _editor(song, config)
    assert "1.500" in editor.readout.text and "1.800" in editor.readout.text


def test_the_view_opens_on_the_default_margin(song, config):
    editor = _editor(song, config)
    t0, t1 = editor.window
    assert (1.5 - t0, t1 - 1.8) == pytest.approx((0.5, 0.5))


def test_widening_shows_more_on_both_sides(song, config):
    editor = _editor(song, config)
    editor._set_margin(editor.margin_s * 2)
    t0, t1 = editor.window
    assert (1.5 - t0, t1 - 1.8) == pytest.approx((1.0, 1.0))


def test_the_wider_button_is_wired_to_the_view(song, config):
    editor = _editor(song, config)
    before = editor.margin_s
    editor.wider_btn.clicks += 1
    assert editor.margin_s > before


def test_the_window_stops_at_the_edges_of_the_recording(song, config):
    # the file is 3 s long, so a large margin cannot be honoured on either side
    editor = _editor(song, config)
    editor._set_margin(8.0)
    t0, t1 = editor.window
    assert (t0, t1) == pytest.approx((0.0, 3.0), abs=1e-3)


def test_widening_leaves_the_boundaries_alone(song, config):
    editor = _editor(song, config)
    _drag(editor, 1.42, 1.91)
    editor._set_margin(4.0)
    assert editor.bounds() == pytest.approx((1.42, 1.91), abs=1e-6)


def test_the_spectrogram_and_the_axis_follow_the_new_window(song, config):
    editor = _editor(song, config)
    editor._set_margin(2.0)
    t0, t1 = editor.window
    assert editor.spectrogram.data["x"][0] == pytest.approx(t0)
    assert editor.spectrogram.data["dw"][0] == pytest.approx(t1 - t0)
    assert (editor.figure.x_range.start, editor.figure.x_range.end) == pytest.approx((t0, t1))


def test_the_drag_callbacks_see_the_new_window(song, config):
    # the JS reads the window from a source, so widening reaches it without
    # rebuilding the callbacks
    editor = _editor(song, config)
    editor._set_margin(2.0)
    assert editor.window_src.data["t0"][0] == pytest.approx(editor.window[0])
    assert editor.window_src.data["t1"][0] == pytest.approx(editor.window[1])


def test_the_view_cannot_be_widened_past_the_cap(song, config):
    from dashboard.view.bounds_editor import MARGIN_MAX_S
    editor = _editor(song, config)
    for _ in range(12):
        editor._set_margin(editor.margin_s * 2)
    assert editor.margin_s == pytest.approx(MARGIN_MAX_S)
    assert editor.wider_btn.disabled


def test_the_view_cannot_be_narrowed_past_the_floor(song, config):
    from dashboard.view.bounds_editor import MARGIN_MIN_S
    editor = _editor(song, config)
    for _ in range(12):
        editor._set_margin(editor.margin_s / 2)
    assert editor.margin_s == pytest.approx(MARGIN_MIN_S)
    assert editor.narrower_btn.disabled


def test_the_neighbours_are_one_renderer_drawn_where_they_are(song, config):
    editor = _editor(song, config, neighbours=[
        _row(song, 0.2, 0.4, "B"), _row(song, 2.5, 2.7, "C"),
    ])
    rects = [r for r in editor.figure.renderers
             if getattr(r.glyph, "__class__", None).__name__ == "Rect"]
    widths = [len(r.data_source.data["x"]) for r in rects]
    assert 2 in widths

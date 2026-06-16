"""
tests/test_drs2p.py
===================
Unit tests for the drs2p image-processing pipeline.

Covered modules
---------------
roi_detection  – RoiMeasurer.roi_to_mask, roi_to_mask_freehand, run()
helpers        – transpose, transpose_autoballance, filter_list, frame_to_sec,
                 sec_to_frame, flatten_array
traces         – find_time_index, data_normalize, csv_transform, csv_cutter

External I/O (Fiji, real TIFF movies, real ROI zip files) is fully mocked
or replaced with synthetic in-memory data so the tests run without any
experiment files on disk.
"""

import csv
import io
import os
import sys
import tempfile
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Make the project's src/ importable without installing the package
# ---------------------------------------------------------------------------
SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


# ---------------------------------------------------------------------------
# Helpers — thin stand-in for the config dataclasses the real code uses
# ---------------------------------------------------------------------------

def _movie_config(spf=0.1, n_frames=100, duration=10.0, events=None):
    cfg = MagicMock()
    cfg.seconds_per_frame_adjusted = spf
    cfg.fps_adjusted = 1.0 / spf
    cfg.movie_duration_adjusted = duration
    cfg.n_frames = n_frames
    cfg.events = events or [[0, 5.0]]
    return cfg


def _trigger_config(
    time_before=2.0,
    time_after=3.0,
    baseline=1.0,
    relative=True,
    mean_col=2,
    cols_per_roi=4,
):
    cfg = MagicMock()
    cfg.time_before_trig = time_before
    cfg.time_after_trig = time_after
    cfg.baseline_duraton = baseline
    cfg.relative_values = relative
    cfg.mean_col_order = mean_col
    cfg.cols_per_roi = cols_per_roi
    return cfg


# ===========================================================================
# roi_detection.py
# NOT DEBUGGED YET
# ===========================================================================

# class TestRoiToMask:
#     """RoiMeasurer.roi_to_mask — all ROI types."""

#     def _measurer(self):
#         from roi_detection import RoiMeasurer
#         return RoiMeasurer(
#             dir="/tmp", tiff="dummy.tif", roi_zip="dummy.zip",
#             output_csv="/tmp/out.csv", seconds_per_frame=0.1,
#         )

#     # --- oval (type 2) ---

#     def test_oval_mask_shape(self):
#         m = self._measurer()
#         roi = MagicMock()
#         roi.roitype = 2
#         roi.left, roi.top, roi.right, roi.bottom = 10, 10, 30, 30
#         mask = m.roi_to_mask(roi, (50, 50))
#         assert mask.shape == (50, 50)
#         assert mask.dtype == bool

#     def test_oval_mask_pixels_inside_ellipse(self):
#         """All True pixels must satisfy the ellipse equation."""
#         m = self._measurer()
#         roi = MagicMock()
#         roi.roitype = 2
#         roi.left, roi.top, roi.right, roi.bottom = 0, 0, 20, 10
#         mask = m.roi_to_mask(roi, (20, 20))
#         xr, yr = 10.0, 5.0          # semi-axes
#         cx, cy = 10.0, 5.0          # centre
#         ys, xs = np.where(mask)
#         dx = (xs - cx) / xr
#         dy = (ys - cy) / yr
#         # Allow a tiny tolerance for boundary rounding
#         assert np.all(dx**2 + dy**2 <= 1.05)

#     def test_oval_mask_has_nonzero_pixels(self):
#         m = self._measurer()
#         roi = MagicMock()
#         roi.roitype = 2
#         roi.left, roi.top, roi.right, roi.bottom = 5, 5, 25, 25
#         mask = m.roi_to_mask(roi, (50, 50))
#         assert mask.sum() > 0

#     def test_oval_mask_outside_roi_is_false(self):
#         """Pixels well outside the bounding box must stay False."""
#         m = self._measurer()
#         roi = MagicMock()
#         roi.roitype = 2
#         roi.left, roi.top, roi.right, roi.bottom = 10, 10, 20, 20
#         mask = m.roi_to_mask(roi, (50, 50))
#         assert not mask[:5, :].any()
#         assert not mask[:, :5].any()
#         assert not mask[25:, :].any()
#         assert not mask[:, 25:].any()

#     def test_oval_small_even(self):
#         """2×2 oval should produce at least 1 pixel."""
#         m = self._measurer()
#         roi = MagicMock()
#         roi.roitype = 2
#         roi.left, roi.top, roi.right, roi.bottom = 5, 5, 7, 7
#         mask = m.roi_to_mask(roi, (20, 20))
#         assert mask.sum() >= 1

#     # --- rect (type 1) ---

#     def test_rect_mask_exact_region(self):
#         m = self._measurer()
#         roi = MagicMock()
#         roi.roitype = 1
#         roi.left, roi.top, roi.right, roi.bottom = 3, 2, 8, 6
#         mask = m.roi_to_mask(roi, (20, 20))
#         assert mask[2:6, 3:8].all()
#         assert mask.sum() == 4 * 5

#     def test_rect_mask_no_overflow(self):
#         m = self._measurer()
#         roi = MagicMock()
#         roi.roitype = 1
#         roi.left, roi.top, roi.right, roi.bottom = 0, 0, 5, 5
#         mask = m.roi_to_mask(roi, (10, 10))
#         assert mask.sum() == 25
#         assert not mask[5:, :].any()

#     # --- unsupported type ---

#     def test_unsupported_roi_type_returns_zero_mask(self, capsys):
#         m = self._measurer()
#         roi = MagicMock()
#         roi.roitype = 99
#         roi.name = "test"
#         mask = m.roi_to_mask(roi, (20, 20))
#         assert mask.shape == (20, 20)
#         assert not mask.any()

#     # --- freehand / polygon (types 0, 7, 8) ---

#     def test_freehand_square_polygon(self):
#         """A square polygon should fill the interior."""
#         m = self._measurer()
#         roi = MagicMock()
#         roi.roitype = 7
#         coords = np.array([[5, 5], [5, 15], [15, 15], [15, 5]], dtype=float)
#         roi.coordinates.return_value = coords
#         mask = m.roi_to_mask(roi, (20, 20))
#         assert mask.sum() > 0
#         # corners outside the square must be False
#         assert not mask[0, 0]
#         assert not mask[19, 19]

#     def test_freehand_empty_coords_returns_zero_mask(self):
#         m = self._measurer()
#         roi = MagicMock()
#         roi.roitype = 7
#         roi.coordinates.return_value = np.array([])
#         mask = m.roi_to_mask(roi, (20, 20))
#         assert not mask.any()

#     def test_polygon_type0(self):
#         """ROI type 0 (polygon) uses the same freehand path."""
#         m = self._measurer()
#         roi = MagicMock()
#         roi.roitype = 0
#         coords = np.array([[2, 2], [2, 8], [8, 8], [8, 2]], dtype=float)
#         roi.coordinates.return_value = coords
#         mask = m.roi_to_mask(roi, (15, 15))
#         assert mask.sum() > 0


# class TestRoiMeasurerRun:
#     """RoiMeasurer.run() — end-to-end with a synthetic TIFF and mock ROIs."""

#     def _make_tiff(self, path, n_frames=5, shape=(20, 20), value=100):
#         """Write a minimal multi-page grayscale TIFF with constant pixel values."""
#         import tifffile
#         # Write each frame as a separate page using photometric='minisblack'
#         # to avoid tifffile interpreting shape (N, H, W) uint16 as RGB.
#         stack = np.full((n_frames, *shape), value, dtype=np.uint16)
#         tifffile.imwrite(path, stack, photometric="minisblack")

#     def test_run_produces_csv_with_correct_shape(self, tmp_path):
#         from roi_detection import RoiMeasurer

#         tiff_path = str(tmp_path / "movie.tif")
#         csv_path  = str(tmp_path / "out.csv")
#         self._make_tiff(tiff_path, n_frames=3, shape=(30, 30), value=200)

#         # Two oval ROIs
#         roi1 = MagicMock(); roi1.roitype = 2; roi1.name = "roi1"
#         roi1.left, roi1.top, roi1.right, roi1.bottom = 2, 2, 12, 12
#         roi2 = MagicMock(); roi2.roitype = 2; roi2.name = "roi2"
#         roi2.left, roi2.top, roi2.right, roi2.bottom = 14, 14, 24, 24

#         meas = RoiMeasurer(
#             dir=str(tmp_path), tiff="movie.tif", roi_zip="dummy.zip",
#             output_csv=csv_path, seconds_per_frame=0.1,
#         )
#         with patch.object(meas, "load_rois", return_value=[roi1, roi2]):
#             meas.run()

#         assert os.path.exists(csv_path)
#         with open(csv_path) as f:
#             rows = list(csv.reader(f))

#         # 1 header + 3 data rows
#         assert len(rows) == 4
#         # header: 1 empty + 4 columns per ROI × 2 ROIs
#         assert len(rows[0]) == 1 + 4 * 2

#     def test_run_mean_equals_frame_value(self, tmp_path):
#         """With a flat-value frame, every ROI mean must equal that value."""
#         from roi_detection import RoiMeasurer

#         VALUE = 512
#         tiff_path = str(tmp_path / "flat.tif")
#         csv_path  = str(tmp_path / "out.csv")
#         self._make_tiff(tiff_path, n_frames=2, shape=(20, 20), value=VALUE)

#         roi = MagicMock(); roi.roitype = 2; roi.name = "r"
#         roi.left, roi.top, roi.right, roi.bottom = 2, 2, 10, 10

#         meas = RoiMeasurer(
#             dir=str(tmp_path), tiff="flat.tif", roi_zip="x.zip",
#             output_csv=csv_path, seconds_per_frame=0.1,
#         )
#         with patch.object(meas, "load_rois", return_value=[roi]):
#             meas.run()

#         with open(csv_path) as f:
#             rows = list(csv.reader(f))

#         # Column layout: ["", "Area(r)", "Mean(r)", "Min(r)", "Max(r)"]
#         # Mean is at index 2
#         for row in rows[1:]:           # skip header
#             assert float(row[2]) == float(VALUE)

#     def test_run_masks_cached_not_recomputed(self, tmp_path):
#         """roi_to_mask must be called once per ROI, not once per frame."""
#         from roi_detection import RoiMeasurer

#         tiff_path = str(tmp_path / "m.tif")
#         csv_path  = str(tmp_path / "o.csv")
#         self._make_tiff(tiff_path, n_frames=5, shape=(20, 20))

#         roi = MagicMock(); roi.roitype = 1; roi.name = "r"
#         roi.left, roi.top, roi.right, roi.bottom = 1, 1, 5, 5

#         meas = RoiMeasurer(
#             dir=str(tmp_path), tiff="m.tif", roi_zip="x.zip",
#             output_csv=csv_path, seconds_per_frame=0.1,
#         )
#         with patch.object(meas, "load_rois", return_value=[roi]):
#             with patch.object(meas, "roi_to_mask", wraps=meas.roi_to_mask) as spy:
#                 meas.run()
#                 # 5 frames, 1 ROI → mask built exactly once
#                 assert spy.call_count == 1

#     def test_run_empty_mask_roi_outputs_zeros(self, tmp_path):
#         """A ROI that produces an empty mask should emit [0,0,0,0]."""
#         from roi_detection import RoiMeasurer

#         tiff_path = str(tmp_path / "m.tif")
#         csv_path  = str(tmp_path / "o.csv")
#         self._make_tiff(tiff_path, n_frames=2, shape=(5, 5))

#         roi = MagicMock(); roi.roitype = 2; roi.name = "empty"
#         # ROI entirely outside the frame
#         roi.left, roi.top, roi.right, roi.bottom = 100, 100, 200, 200

#         meas = RoiMeasurer(
#             dir=str(tmp_path), tiff="m.tif", roi_zip="x.zip",
#             output_csv=csv_path, seconds_per_frame=0.1,
#         )
#         with patch.object(meas, "load_rois", return_value=[roi]):
#             meas.run()

#         with open(csv_path) as f:
#             rows = list(csv.reader(f))
#         for row in rows[1:]:
#             assert row[1:] == ["0", "0", "0", "0"]


# ===========================================================================
# helpers.py
# ===========================================================================

class TestHelpersTranspose:

    def _make_helpers(self):
        from helpers import Helpers
        h = Helpers()
        # attach a minimal movie_config so frame/sec helpers don't crash
        h.movie_config = _movie_config()
        return h

    def test_transpose_numpy_array(self):
        h = self._make_helpers()
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        result = h.transpose(arr)
        np.testing.assert_array_equal(result, arr.T)

    def test_transpose_list_of_lists(self):
        h = self._make_helpers()
        matrix = [[1, 2, 3], [4, 5, 6]]
        result = h.transpose(matrix)
        assert result == [[1, 4], [2, 5], [3, 6]]

    def test_transpose_single_row(self):
        h = self._make_helpers()
        assert h.transpose([[1, 2, 3]]) == [[1], [2], [3]]

    def test_transpose_single_column(self):
        h = self._make_helpers()
        assert h.transpose([[1], [2], [3]]) == [[1, 2, 3]]

    def test_transpose_ndarray_is_view_not_copy(self):
        h = self._make_helpers()
        arr = np.array([[1, 2], [3, 4]])
        result = h.transpose(arr)
        # .T is a view — modifying it changes the original
        result[0, 1] = 99
        assert arr[1, 0] == 99

    def test_transpose_autoballance_ragged(self):
        h = self._make_helpers()
        data = [[1, 2, 3], [4, 5]]
        result = h.transpose_autoballance(data)
        # Should be 3 rows (max length), 2 cols
        assert len(result) == 3
        assert result[0] == (1, 4)
        assert result[1] == (2, 5)
        assert result[2][0] == 3
        assert result[2][1] is None   # padding

    def test_transpose_autoballance_uniform(self):
        h = self._make_helpers()
        data = [[1, 2], [3, 4]]
        result = h.transpose_autoballance(data)
        assert result == ((1, 3), (2, 4))


class TestHelpersFilterList:

    def _h(self):
        from helpers import Helpers
        h = Helpers()
        h.movie_config = _movie_config()
        return h

    def test_filter_replace_true(self):
        h = self._h()
        data = [10, 20, 30, 40]
        mask = [True, False, True, False]
        result = h.filter_list(data, mask, replace=True, replace_with=0)
        assert result == [10, 0, 30, 0]

    def test_filter_replace_false(self):
        h = self._h()
        data = [10, 20, 30, 40]
        mask = [True, False, True, False]
        result = h.filter_list(data, mask, replace=False)
        assert result == [10, 30]

    def test_filter_all_true(self):
        h = self._h()
        data = [1, 2, 3]
        result = h.filter_list(data, [True]*3, replace=False)
        assert result == [1, 2, 3]

    def test_filter_all_false(self):
        h = self._h()
        data = [1, 2, 3]
        result = h.filter_list(data, [False]*3, replace=True, replace_with=None)
        assert result == [None, None, None]

    def test_filter_replace_with_string(self):
        h = self._h()
        result = h.filter_list(["a", "b", "c"], [True, False, True],
                               replace=True, replace_with="x")
        assert result == ["a", "x", "c"]


class TestHelpersFrameSec:

    def _h(self, spf=0.5, n_frames=20, duration=10.0):
        from helpers import Helpers
        h = Helpers()
        h.movie_config = _movie_config(spf=spf, n_frames=n_frames, duration=duration)
        return h

    def test_frame_to_sec_zero(self):
        h = self._h(spf=0.5)
        assert h.frame_to_sec(0) == pytest.approx(0.0)

    def test_frame_to_sec_midpoint(self):
        h = self._h(spf=0.5)
        # frame 10 at 2 fps → 5.0 s
        assert h.frame_to_sec(10) == pytest.approx(5.0)

    def test_sec_to_frame_zero(self):
        h = self._h(spf=0.5)
        assert h.sec_to_frame(0.0) == 0

    def test_sec_to_frame_midpoint(self):
        h = self._h(spf=0.5)
        # 5 s at 2 fps → frame 10
        assert h.sec_to_frame(5.0) == 10

    def test_frame_to_sec_beyond_clamps(self):
        """frame > n_frames should return movie_duration_adjusted."""
        h = self._h(spf=0.5, n_frames=20, duration=10.0)
        assert h.frame_to_sec(9999) == pytest.approx(10.0)

    def test_sec_to_frame_beyond_clamps(self):
        h = self._h(spf=0.5, n_frames=20, duration=10.0)
        assert h.sec_to_frame(9999.0) == 20

    def test_roundtrip(self):
        h = self._h(spf=0.1)
        for frame in range(10):
            assert h.sec_to_frame(h.frame_to_sec(frame)) == frame


class TestHelpersFlatten:

    def _h(self):
        from helpers import Helpers
        h = Helpers()
        h.movie_config = _movie_config()
        return h

    def test_flatten_nested(self):
        h = self._h()
        assert list(h.flatten_array([1, [2, [3, 4]], 5])) == [1, 2, 3, 4, 5]

    def test_flatten_flat(self):
        h = self._h()
        assert list(h.flatten_array([1, 2, 3])) == [1, 2, 3]

    def test_flatten_empty(self):
        h = self._h()
        assert list(h.flatten_array([])) == []


# ===========================================================================
# traces.py  — pure computation methods (no filesystem I/O)
# ===========================================================================

def _make_traces_obj(movie_cfg=None, trigger_cfg=None):
    """Return a TracesCalc instance with minimal mocked dependencies."""
    # TracesCalc inherits from Helpers and uses self.movie_config / trigger_config
    # We build a minimal stand-in that has only what the tested methods need.
    from traces import TracesCalc
    mc = movie_cfg or _movie_config(spf=0.1)
    tc = trigger_cfg or _trigger_config()

    obj = TracesCalc.__new__(TracesCalc)
    obj.movie_config   = mc
    obj.trigger_config = tc
    obj.run_config     = MagicMock()
    obj.s_trig_time    = 5.0
    obj.log            = ""
    obj.path           = "/tmp"
    obj.file           = "dummy"
    obj.output_suffix  = "test"
    obj.filename_suffix = ""
    obj.file_nosuffix  = "dummy"
    obj.s_movie_duration = mc.movie_duration_adjusted
    obj.n_frames       = mc.n_frames
    return obj


class TestFindTimeIndex:

    def test_exact_match(self):
        t = _make_traces_obj()
        # content as list-of-rows where column 0 is time
        content = [[i * 0.5, 0.0] for i in range(10)]
        assert t.find_time_index(content, 2.0) == 4   # 4 * 0.5 == 2.0

    def test_nearest_match(self):
        t = _make_traces_obj()
        content = [[0.0, 0], [1.0, 0], [2.0, 0], [3.0, 0]]
        # 1.4 → nearest is 1.0 (index 1)
        assert t.find_time_index(content, 1.4) == 1

    def test_nearest_match_upper(self):
        t = _make_traces_obj()
        content = [[0.0, 0], [1.0, 0], [2.0, 0]]
        # 1.6 → nearest is 2.0 (index 2)
        assert t.find_time_index(content, 1.6) == 2

    def test_single_row(self):
        t = _make_traces_obj()
        content = [[7.77, 99]]
        assert t.find_time_index(content, 0.0) == 0

    def test_returns_int(self):
        t = _make_traces_obj()
        content = [[0.0, 0], [1.0, 0]]
        idx = t.find_time_index(content, 0.5)
        assert isinstance(idx, int)


class TestDataNormalize:

    def test_zero_baseline_returns_zeros(self):
        t = _make_traces_obj()
        # 3 ROIs, 5 time points; baseline is frames 0:2 which have value 0
        data = [[0.0, 0.0, 1.0, 2.0, 3.0],
                [0.0, 0.0, 2.0, 4.0, 6.0],
                [0.0, 0.0, 0.5, 1.0, 1.5]]
        result = t.data_normalize(data, start=0, zero=2)
        # mean == 0 → result should be 0.0 everywhere
        for row in result:
            assert all(v == pytest.approx(0.0) for v in row)

    def test_flat_signal_normalizes_to_zero(self):
        """A signal equal to its baseline should give ΔF/F₀ = 0 everywhere."""
        t = _make_traces_obj()
        data = [[5.0, 5.0, 5.0, 5.0, 5.0]]
        result = t.data_normalize(data, start=0, zero=3)
        assert all(v == pytest.approx(0.0) for v in result[0])

    def test_doubling_signal_gives_one(self):
        """A signal that is 2× the baseline should give ΔF/F₀ = 1.0."""
        t = _make_traces_obj()
        data = [[4.0, 4.0, 8.0, 8.0]]   # baseline mean = 4, signal = 8
        result = t.data_normalize(data, start=0, zero=2)
        assert result[0][2] == pytest.approx(1.0)
        assert result[0][3] == pytest.approx(1.0)

    def test_output_shape_preserved(self):
        t = _make_traces_obj()
        data = [[1.0, 2.0, 3.0, 4.0],
                [2.0, 2.0, 4.0, 6.0]]
        result = t.data_normalize(data, start=0, zero=2)
        assert len(result) == 2
        assert all(len(row) == 4 for row in result)

    def test_returns_list_of_lists(self):
        t = _make_traces_obj()
        data = [[1.0, 2.0, 3.0]]
        result = t.data_normalize(data, start=0, zero=1)
        assert isinstance(result, list)
        assert isinstance(result[0], list)

    def test_multiple_rois_independent(self):
        """Each ROI must be normalised against its own baseline, not others."""
        t = _make_traces_obj()
        # roi0 baseline=2, roi1 baseline=10
        data = [[2.0, 2.0, 4.0],    # ΔF/F₀ at t=2 → (4-2)/2 = 1.0
                [10.0, 10.0, 20.0]] # ΔF/F₀ at t=2 → (20-10)/10 = 1.0
        result = t.data_normalize(data, start=0, zero=2)
        assert result[0][2] == pytest.approx(1.0)
        assert result[1][2] == pytest.approx(1.0)


class TestCsvTransform:
    """csv_transform — converts raw ImageJ CSV to (time, mean) table."""

    def _raw_csv(self, n_frames=5, n_rois=2, mean_col=2, cols_per_roi=4):
        """
        Synthetic content_raw mimicking ImageJ measurement export.

        Real ImageJ column layout per ROI (cols_per_roi=4, mean_col=2):
            col 0 : frame index (1-based)
            col 1 : Area
            col 2 : Mean      ← mean_col_order=2 targets this
            col 3 : Min
            col 4 : Max
            (next ROI repeats the pattern)

        mean values are set to float(10 + roi_index + frame_index) so each
        ROI×frame combination is uniquely identifiable in assertions.
        """
        header = [""]
        for r in range(n_rois):
            header += [f"Area{r}", f"Mean{r}", f"Min{r}", f"Max{r}"]

        rows = [header]
        for f in range(n_frames):
            row = [str(f + 1)]          # 1-based frame index
            for r in range(n_rois):
                area = 100
                mean = float(10 + r + f)
                mn   = 5.0
                mx   = 15.0
                row += [str(area), str(mean), str(mn), str(mx)]
            rows.append(row)

        return tuple(tuple(r) for r in rows)

    def test_output_is_list_of_rows(self):
        mc = _movie_config(spf=0.1)
        tc = _trigger_config(mean_col=2, cols_per_roi=4)
        t  = _make_traces_obj(mc, tc)
        raw = self._raw_csv(n_frames=5, n_rois=2)
        result = t.csv_transform(raw)
        assert isinstance(result, list)
        assert all(isinstance(row, list) for row in result)

    def test_row_count_equals_n_frames(self):
        mc = _movie_config(spf=0.1)
        tc = _trigger_config(mean_col=2, cols_per_roi=4)
        t  = _make_traces_obj(mc, tc)
        raw = self._raw_csv(n_frames=7, n_rois=3)
        result = t.csv_transform(raw)
        assert len(result) == 7

    def test_first_column_is_time(self):
        """
        Each frame's timestamp marks the END of its acquisition window, not
        the start. Frame 0 (1-indexed frame 1 in the raw CSV) finishes
        acquiring at t = spf, frame 1 at t = 2*spf, and so on. This matches
        the acquisition hardware's actual timestamp semantics.
        """
        mc = _movie_config(spf=0.2)
        tc = _trigger_config(mean_col=2, cols_per_roi=4)
        t  = _make_traces_obj(mc, tc)
        raw = self._raw_csv(n_frames=5, n_rois=1)
        result = t.csv_transform(raw)
        times = [row[0] for row in result]
        expected = [(i + 1) * 0.2 for i in range(5)]
        for got, exp in zip(times, expected):
            assert got == pytest.approx(exp, rel=1e-6)

    def test_col_count_is_1_plus_n_rois(self):
        mc = _movie_config(spf=0.1)
        tc = _trigger_config(mean_col=2, cols_per_roi=4)
        t  = _make_traces_obj(mc, tc)
        raw = self._raw_csv(n_frames=4, n_rois=3)
        result = t.csv_transform(raw)
        # 1 time col + 3 roi mean cols
        assert all(len(row) == 4 for row in result)

    def test_mean_values_extracted_correctly(self):
        """The Mean column values must survive the transform unchanged."""
        mc = _movie_config(spf=0.1)
        tc = _trigger_config(mean_col=2, cols_per_roi=4)
        t  = _make_traces_obj(mc, tc)
        raw = self._raw_csv(n_frames=3, n_rois=2)
        result = t.csv_transform(raw)
        # frame 0, roi 0: mean = 10 + 0 + 0 = 10.0
        assert result[0][1] == pytest.approx(10.0)
        # frame 1, roi 1: mean = 10 + 1 + 1 = 12.0
        assert result[1][2] == pytest.approx(12.0)


class TestCsvCutter:
    """csv_cutter — slices and optionally normalises a time-series block."""

    def _content(self, n_frames=50, n_rois=2, spf=0.1, baseline_val=5.0, signal_val=10.0, trig_frame=20):
        """
        Returns list-of-rows: [time, roi0, roi1, …]
        Baseline = baseline_val for frames 0..trig_frame-1
        Signal   = signal_val  for frames trig_frame..end
        """
        rows = []
        for i in range(n_frames):
            t = i * spf
            val = baseline_val if i < trig_frame else signal_val
            rows.append([t] + [val] * n_rois)
        return rows

    def test_output_is_ndarray(self):
        mc = _movie_config(spf=0.1, n_frames=50, duration=5.0)
        tc = _trigger_config(time_before=1.0, time_after=2.0, baseline=1.0, relative=False)
        t  = _make_traces_obj(mc, tc)
        content = self._content(n_frames=50)
        result = t.csv_cutter(content)
        assert isinstance(result, np.ndarray)

    def test_time_column_rezeroed(self):
        """After cutting, time at the trigger index should be ≈ 0."""
        mc = _movie_config(spf=0.1, n_frames=60, duration=6.0)
        tc = _trigger_config(time_before=1.0, time_after=2.0, baseline=0.5, relative=False)
        t  = _make_traces_obj(mc, tc)
        t.s_trig_time = 3.0
        content = self._content(n_frames=60, spf=0.1, trig_frame=30)
        result = t.csv_cutter(content)
        time_col = result[:, 0]
        # The trigger must be the zero crossing
        closest_to_zero_idx = np.argmin(np.abs(time_col))
        assert time_col[closest_to_zero_idx] == pytest.approx(0.0, abs=0.15)

    def test_output_window_length(self):
        """Output rows ≈ (time_before + time_after) / spf."""
        mc = _movie_config(spf=0.1, n_frames=100, duration=10.0)
        tc = _trigger_config(time_before=2.0, time_after=3.0, baseline=1.0, relative=False)
        t  = _make_traces_obj(mc, tc)
        t.s_trig_time = 5.0
        content = self._content(n_frames=100, spf=0.1, trig_frame=50)
        result = t.csv_cutter(content)
        expected_rows = int((2.0 + 3.0) / 0.1)
        assert abs(len(result) - expected_rows) <= 3   # ±3 frames tolerance

    def test_relative_normalization_applied(self):
        """With relative_values=True the baseline columns should be ≈ 0."""
        mc = _movie_config(spf=0.1, n_frames=60, duration=6.0)
        tc = _trigger_config(time_before=1.0, time_after=2.0, baseline=1.0, relative=True)
        t  = _make_traces_obj(mc, tc)
        t.s_trig_time = 3.0
        content = self._content(n_frames=60, spf=0.1, trig_frame=30,
                                baseline_val=5.0, signal_val=5.0)
        result = t.csv_cutter(content)
        # Flat signal → ΔF/F₀ = 0 everywhere
        roi_cols = result[:, 1:]
        assert np.allclose(roi_cols, 0.0, atol=1e-6)

    def test_no_relative_keeps_raw_values(self):
        mc = _movie_config(spf=0.1, n_frames=60, duration=6.0)
        tc = _trigger_config(time_before=1.0, time_after=2.0, baseline=1.0, relative=False)
        t  = _make_traces_obj(mc, tc)
        t.s_trig_time = 3.0
        content = self._content(n_frames=60, spf=0.1, trig_frame=30,
                                baseline_val=7.0, signal_val=7.0)
        result = t.csv_cutter(content)
        roi_cols = result[:, 1:]
        assert np.allclose(roi_cols, 7.0, atol=1e-6)

    def test_col_count_preserved(self):
        n_rois = 4
        mc = _movie_config(spf=0.1, n_frames=80, duration=8.0)
        tc = _trigger_config(time_before=1.0, time_after=2.0, baseline=0.5, relative=False)
        t  = _make_traces_obj(mc, tc)
        t.s_trig_time = 4.0
        content = self._content(n_frames=80, n_rois=n_rois, spf=0.1, trig_frame=40)
        result = t.csv_cutter(content)
        assert result.shape[1] == 1 + n_rois


# ===========================================================================
# Integration smoke-test  — wires csv_transform → csv_cutter together
# ===========================================================================

class TestTransformCutterIntegration:

    def test_pipeline_produces_correct_deltaf(self):
        """
        Synthetic pipeline: build a raw CSV where baseline=5, signal=10
        after the trigger.  After transform + cut + normalisation the
        signal portion should be ΔF/F₀ ≈ 1.0.
        """
        spf      = 0.1
        n_frames = 80
        trig_t   = 4.0               # trigger at frame 40
        trig_f   = int(trig_t / spf) # 40

        # Build raw CSV — real ImageJ layout: frame, Area, Mean, Min, Max
        # mean_col=2 targets the Mean column (index 2 within each ROI group)
        header = ["", "Area0", "Mean0", "Min0", "Max0"]
        rows   = [header]
        for i in range(n_frames):
            mean = 5.0 if i < trig_f else 10.0
            rows.append([str(i+1), "100", str(mean), "2.0", "15.0"])
        content_raw = tuple(tuple(r) for r in rows)

        mc = _movie_config(spf=spf, n_frames=n_frames, duration=n_frames * spf)
        tc = _trigger_config(
            time_before=2.0, time_after=3.0,
            baseline=2.0, relative=True,
            mean_col=2, cols_per_roi=4,
        )
        t = _make_traces_obj(mc, tc)
        t.s_trig_time = trig_t

        content  = t.csv_transform(content_raw)
        result   = t.csv_cutter(content)

        # Rows after the trigger (time_col > 0)
        time_col = result[:, 0]
        roi_col  = result[:, 1]
        post_trig = roi_col[time_col > 0.05]

        assert len(post_trig) > 0
        assert np.allclose(post_trig, 1.0, atol=0.05)



if __name__ == '__main__':
    pytest.main([__file__, '-v'])
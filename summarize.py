"""
Summarize Amplitude / AUC CSV data across several working folders.

Run it pointing at a ROOT directory that contains one or more working folders:

    ROOT/
        <WorkFolder A>/
            <YYYY_MM_DD>[_M#]_Field..._CALCULATIONS_auto_<suffix>/
                outputs_.../
                    _Ampl_A&C_rows-roi_cols-epoch_<suffix>_auto_.csv
                    _Ampl_C_rows-roi_cols-epoch_<suffix>_auto_.csv
                    _AUC_A&C_rows-roi_cols-epoch_<suffix>_auto_.csv
                    _AUC_C_rows-roi_cols-epoch_<suffix>_auto_.csv
                    _Bin_C_rows-roi_cols-epoch_<suffix>_auto_.csv
        <WorkFolder B>/ ...

<suffix> is whatever the pipeline appended after "_CALCULATIONS_auto_", i.e.
the trigger's label -- "_Control", "_Application", or empty when the trigger
had no label. Each suffix is a separate experimental condition, so it is
summarized SEPARATELY: conditions are never pooled, and never end up in the
same paired statistical test.

Two kinds of output are produced (all as .xlsx), one set per suffix:

  * ROOT/summary_for_<workfolder>[_<suffix>].xlsx -- one row per day
    (instance), the grand mean over every surviving cell of that day.
  * <workfolder>/summary_<instance>[_<suffix>].xlsx -- a local, per-day summary
    whose rows are individual ROIs (each value is the mean across that ROI's
    epochs).

An empty suffix keeps the old, unsuffixed file names.

Both share the same column layout:
    <id> | Ampl A&C | Ampl C | Ampl C/A&C | <blank> | AUC A&C | AUC C | AUC C/A&C

Under the data rows every sheet carries its paired Wilcoxon bar plots and,
below them, six histograms -- Ampl A&C, Ampl C, Ampl A&C/C, AUC A&C, AUC C,
AUC A&C/C -- in a 3-wide grid. Each panel is the per-folder figure of
make_histograms.py for one metric: the pooled bouton histogram with one
empirical CDF per day drawn thin behind their bolded mean, so no single day
dominates the curve. A day is one n whatever it holds -- several acquisitions
of the same date (and the same _M number) are pooled into it as repetitions,
the same unit the bar plots above are tested on. The distributions are always
ROI-level, so a ROOT summary shows the boutons of all the days it covers, not
its day means.

Every chart is embedded twice: as a PNG and as the SVG of the same figure,
attached to it the way Excel stores a vector picture. Excel 2019 / Microsoft
365 draw the SVG, so a plot stays sharp at any zoom and can be pulled into a
figure as vector art; older versions fall back to the PNG. Set EMBED_SVG to
False for raster-only workbooks.

When a working folder holds several applications, they also get

  * ROOT/<CondA>_vs_<CondB>[_vs_<CondC>...]_summary_for_<workfolder>.xlsx -- the
    conditions' Ampl A&C/C and AUC A&C/C side by side, one row per day, with
    the same bar plots (now testing condition against condition) and a
    distribution overlay per ratio metric.

Two conditions give one sheet. Three or more give an "all conditions" sheet,
where the bar plots carry every condition at once and the test becomes the
repeated-measures one AutoStatLib picks, followed by one sheet per pair in the
same workbook -- a pair usually shares more days than all the conditions
together do, and those extra days would otherwise be lost.

Only days present in every condition of a sheet take part there, since the test
is paired.

Folders without the expected structure are ignored.

The "instance" identifier is everything before "_Field" in a calculation
folder's name, i.e. the date plus an optional _M1 / _M2 suffix. Several folders
can share the same instance AND suffix (e.g. two _Field acquisitions on the
same date); their ROIs are pooled together for that day and condition.

For every output folder, rows (ROIs) whose TRUE-rate in Bin_C is below THRESHOLD
are dropped from the Ampl / AUC matrices before anything is computed.

When several conditions are present, that ROI filter is taken from the control
condition (BINARIZATION_SOURCE) and applied unchanged to all the others, so
Control and Application report the same ROIs and ROI selection cannot be
biased by the drug response. Folders are paired with their control by
acquisition (same day, same _Field folder), since the ROI indices only line up
within one movie. A folder with no control counterpart, or one whose ROI count
disagrees with it, falls back to its own filter and says so.

What counts as one acquisition needs care. A calculation folder is named
"<movie csv>_CALCULATIONS_auto_<trigger label>", and running a movie with
several triggers leaves one exported CSV per trigger behind, each of which the
next trigger is then calculated over as well. So the same movie reappears as
"..._Control.csv_CALCULATIONS_auto__Drug" beside "..._Drug.csv_CALCULATIONS_
auto__Drug", with identical results, and its CSV name carries a condition
label that says which trigger exported it rather than what was recorded.
resolve_acquisitions() therefore strips that label to get the acquisition's
identity and keeps a single copy per acquisition -- the one whose CSV carries
the condition's own label. Without it a drug condition reports twice the
acquisitions of its control, weighs those boutons twice in the day mean, and
never finds its control counterpart to inherit the ROI filter from, so it
silently falls back to a filter chosen under the drug.

The "% AP success" columns stay per-condition: they measure how often each ROI
actually fired under that condition, so they are never inherited.
"""

import hashlib
import itertools
import math
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: render plots to files, never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.utils import column_index_from_string, get_column_letter

import AutoStatLib

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# ROOT directory containing one or more working folders.
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent

# ROOT-level (per-working-folder) summary file name: "<PREFIX><dirname>.xlsx".
SUMMARY_PREFIX = "summary_for_"
# Local per-day summary file name inside a working folder: "<PREFIX><instance>.xlsx".
LOCAL_PREFIX = "summary_"

# A cell is considered "signal" when its SNR is at or above this value.
SNR_THRESHOLD = 5.0

# Keep a ROI only if at least this fraction of its cells pass the SNR threshold.
ROI_THRESHOLD = 0.29

# When a working folder holds several conditions (suffixes), the ROI filter is
# taken from THIS condition and applied unchanged to all the others, so every
# condition reports the same set of ROIs and the selection cannot be biased by
# the drug response. Matched case-insensitively against the folder suffix;
# set to "" to give every condition its own filter again.
BINARIZATION_SOURCE = "Control"

# Metric -> glob pattern inside each outputs_ folder. Order defines column order.
# The wildcard covers varying suffixes like __auto_ vs __Control_auto_ etc.
METRIC_FILES = {
    "Ampl A&C": "_Ampl_A&C_rows-roi_cols-epoch_*.csv",
    "Ampl C":   "_Ampl_C_rows-roi_cols-epoch_*.csv",
    "AUC A&C":  "_AUC_A&C_rows-roi_cols-epoch_*.csv",
    "AUC C":    "_AUC_C_rows-roi_cols-epoch_*.csv",
}
METRICS = list(METRIC_FILES)
SNR_AC_FILE = "_SNR_A&C_rows-roi_cols-epoch_*.csv"
SNR_C_FILE  = "_SNR_C_rows-roi_cols-epoch_*.csv"

CALC_SUFFIX = "_CALCULATIONS_auto_"

# A valid calculation folder: date (+ optional _M#), then _Field ..., then
# CALCULATIONS_auto_, then the trigger label. Group 1 is the instance (day),
# group 2 the label -- empty for an unlabelled trigger, "_Control" etc.
# otherwise. The leading .* is greedy on purpose so the LAST occurrence of
# CALC_SUFFIX is the separator.
DIR_RE = re.compile(
    r"^(\d{4}_\d{2}_\d{2}(?:_M\d+)?)_Field.*" + re.escape(CALC_SUFFIX) + r"(.*)$"
)

# Paired comparisons to test + plot in every summary.
# (left/right are 0-based indices into an output row; anchor is the Excel column
#  letter under which the plot is dropped -- "B" = cols 2-3, "F" = cols 6-7.)
COMPARISONS = [
    {"left": 1, "right": 2, "names": ["Ampl A&C",        "Ampl C"],        "anchor": "B", "y_label": "Ampl"},
    {"left": 5, "right": 6, "names": ["AUC A&C",         "AUC C"],         "anchor": "F", "y_label": "AUC"},
    {"left": 9, "right": 10, "names": ["% AP A&C", "% AP C"], "anchor": "J", "y_label": "AP success % "},
]

# Column indices of the two ratio metrics in an output row, used to pair two
# conditions against each other in the cross-condition summary.
RATIO_COLUMNS = {"Ampl A&C/C": 3, "AUC A&C/C": 7}

# What those ratios are called on the axes of the cross-condition file. The
# per-condition summaries keep the short column names, since their panels sit
# under the columns they come from.
RATIO_AXIS_LABELS = {
    "Ampl A&C/C": "Presynaptic inhibition rate (amplitude)",
    "AUC A&C/C":  "Presynaptic inhibition rate (AUC)",
}

# Cross-condition file name: "<CondA>_vs_<CondB>_summary_for_<workfolder>.xlsx".
COMPARE_SEP = "_vs_"

# Anchor columns of the two bar plots when exactly two conditions are compared,
# so that file keeps the layout it has always had. With more conditions the bar
# plots are wider and get laid out from FIRST_PLOT_COL by their actual width.
CONDITION_ANCHORS = ["B", "F"]
FIRST_PLOT_COL = 2      # column B
DEFAULT_COL_PX = 64     # default Excel column width in px, to convert widths

# --- histogram panels, dropped under the bar plots --------------------------
# Distributions are always ROI-level (one value per bouton), so a ROOT summary
# shows the pooled boutons of every day it covers rather than its day means.
HIST_METRICS = ["Ampl A&C", "Ampl C", "Ampl A&C/C", "AUC A&C", "AUC C", "AUC A&C/C"]
HIST_BINS_N = 50    # histogram bins
HIST_GRID_N = 512   # x-resolution of the empirical CDFs
HIST_SIZE = (5, 3.2)  # inches, per panel

# A ratio explodes when its denominator sits near zero, and a couple of such
# boutons are enough to squash every bar of a panel into a single bin. When the
# bulk of the values covers less than HIST_CLIP_SHARE of the full range, the
# view is clipped to the percentile window below and the title says how many
# boutons fell outside it. Nothing is dropped from the data itself: the CDF is
# still computed over every value, so a clipped panel starts above 0.
HIST_CLIP_PCT = 1.0     # percentile window kept in view: 1st .. 99th
HIST_CLIP_SHARE = 0.25  # clip only when that window is this much of the range
HIST_CLIP_MIN_N = 10    # ... and only with enough boutons for percentiles
HIST_DPI = 150

# Where the panels land: 3 per band, left to right, then the next band below.
# The anchors are 6 columns apart so half-scale 750 px figures do not overlap.
HIST_COLS = ["B", "H", "N"]
IMG_SCALE = 2       # embedded images are shown at 1/IMG_SCALE of their pixel size
PX_PER_ROW = 20     # default Excel row height in px, to convert heights to rows

# Every chart goes into the workbook twice: as the PNG openpyxl can embed, and
# as the SVG of the same figure attached to it through the Office "svgBlip"
# extension -- the structure Excel itself writes for a vector picture. Excel
# 2016 and newer draw the SVG, so a plot stays sharp at any zoom and can be
# copied into a figure; anything older ignores the extension and shows the PNG.
# Set to False to go back to raster-only workbooks.
EMBED_SVG = True
SVG_EXT_URI = "{96DAC541-7B7A-43D3-8B79-37D633B846F1}"
SVG_NS = "http://schemas.microsoft.com/office/drawing/2016/SVG/main"
IMAGE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"


def read_matrix(path: Path) -> pd.DataFrame:
    """Read a head-less, index-less numeric matrix (rows=ROI, cols=epoch)."""
    return pd.read_csv(path, header=None)


def snr_to_bin(path: Path) -> pd.DataFrame:
    """Read the SNR matrix and binarize: True where SNR >= SNR_THRESHOLD."""
    snr = pd.read_csv(path, header=None, dtype=float)
    return snr >= SNR_THRESHOLD


def load_matrices_for_folder(out_dir: Path):
    """Return {metric: UNFILTERED DataFrame} plus the folder's own ROI mask.

    Nothing is dropped here. Which ROIs survive is decided later, in
    apply_binarization(), because when a working folder holds several
    conditions the mask comes from the control condition rather than from each
    folder on its own.

    Returns None (and warns) if a required file is missing or shapes mismatch.
    """
    def resolve(pattern):
        """Return the first file matching a glob pattern, or None."""
        return next(out_dir.glob(pattern), None)

    snr_c_path = resolve(SNR_C_FILE)
    if snr_c_path is None:
        print(f"  ! missing {SNR_C_FILE} in {out_dir.name}; skipping folder")
        return None

    snr_ac_path = resolve(SNR_AC_FILE)
    if snr_ac_path is None:
        print(f"  ! missing {SNR_AC_FILE} in {out_dir.name}; skipping folder")
        return None

    bin_c  = snr_to_bin(snr_c_path)
    bin_ac = snr_to_bin(snr_ac_path)

    # Combine the two binarized matrices: a cell is "signal" if either SNR is above threshold.
    binarization = bin_c | bin_ac

    # Per-ROW (per-ROI) fraction of cells above SNR threshold; keep rows at/above ROI_THRESHOLD.
    own_mask = binarization.mean(axis=1) >= ROI_THRESHOLD

    result = {}
    for metric, pattern in METRIC_FILES.items():
        fpath = resolve(pattern)
        if fpath is None:
            print(f"  ! missing {pattern} in {out_dir.name}; skipping folder")
            return None

        mat = read_matrix(fpath)
        if mat.shape[0] != bin_c.shape[0]:
            print(f"  ! shape mismatch for {pattern} in {out_dir.name}; skipping folder")
            return None

        result[metric] = mat                 # unfiltered; apply_binarization() cuts it

    # Per-ROI AP success rate (fraction of epochs above SNR threshold) for ALL ROIs (no filter).
    # These stay per-condition: they measure how often each ROI actually fired
    # under THAT condition, so they must not be inherited from the control.
    result["AP A&C"]  = bin_ac.mean(axis=1)
    result["AP C"]    = bin_c.mean(axis=1)
    result["own_mask"] = own_mask            # bool Series over all ROI indices
    return result


def find_outputs_dir(calc_dir: Path):
    """Return the single outputs_ subfolder inside a calculation folder."""
    for sub in calc_dir.iterdir():
        if sub.is_dir() and sub.name.startswith("outputs_"):
            return sub
    return None


def source_token(calc_dir_name: str, instance: str, suffix: str) -> str:
    """Short, traceable tag for an acquisition folder (e.g. 'Field_1_0001...')."""
    token = calc_dir_name
    tail = CALC_SUFFIX + suffix
    if token.endswith(tail):
        token = token[: -len(tail)]
    if token.startswith(instance + "_"):
        token = token[len(instance) + 1:]
    return token


def movie_label(token: str, labels) -> str:
    """The condition label the source CSV of an acquisition is named after.

    Empty when it carries none of them.
    """
    stem = token[:-4] if token.lower().endswith(".csv") else token
    for label in sorted(labels, key=len, reverse=True):  # CNQX_20uM before CNQX
        if label and stem.lower().endswith("_" + label.lower()):
            return label
    return ""


def acquisition_key(token: str, labels) -> str:
    """What identifies the acquisition itself, without the condition label.

    A calculation folder is "<movie csv>_CALCULATIONS_auto_<trigger label>",
    and for a movie with several triggers the pipeline exports one CSV per
    trigger, so the CSV name usually ends with a condition label as well. That
    label says which trigger wrote the CSV, not which movie was recorded, so
    dropping it is what makes the same field line up across conditions.
    """
    stem = token[:-4] if token.lower().endswith(".csv") else token
    label = movie_label(token, labels)
    return stem[: -(len(label) + 1)] if label else stem


def same_results(left, right) -> bool:
    """True when two calculation folders hold identical metric matrices."""
    return all(left[metric].equals(right[metric]) for metric in METRICS)


def resolve_acquisitions(conditions):
    """Key every folder by its acquisition and drop the pipeline's duplicates.

    Running a movie with several triggers leaves one CSV per trigger on disk,
    and each trigger is then calculated over every CSV exported so far. One
    acquisition therefore shows up two or three times under the same
    condition, with byte-identical results. Pooled as they are, they inflate
    the ROI count and weigh those boutons several times in the day's mean --
    which is why a drug condition could report twice the acquisitions of its
    control.

    The copy whose CSV carries the condition's own label is the one kept. If
    the copies disagree they are not duplicates, so all of them are kept and
    said so; that is not a call this function can make.
    """
    labels = {clean_suffix(suffix) for suffix in conditions}
    labels.discard("")

    for suffix, instances in conditions.items():
        condition = condition_label(suffix)
        for instance, folders in instances.items():
            groups = {}
            for token, mats in folders:
                groups.setdefault(acquisition_key(token, labels), []).append((token, mats))

            resolved = []
            for key, found in groups.items():
                if len(found) == 1:
                    resolved.append((key, found[0][1]))
                    continue

                if not all(same_results(found[0][1], other[1]) for other in found[1:]):
                    print(f"  ! {instance} {key} [{condition}]: {len(found)} runs of the "
                          f"same acquisition with DIFFERENT results "
                          f"({', '.join(token for token, _ in found)}); keeping all of them")
                    resolved += found
                    continue

                keep = next((f for f in found
                             if movie_label(f[0], labels) == clean_suffix(suffix)), found[0])
                print(f"  ! {instance} {key} [{condition}]: {len(found)} identical runs of "
                      f"the same acquisition; keeping {keep[0]}")
                resolved.append((key, keep[1]))

            instances[instance] = resolved
    return conditions


def clean_suffix(suffix: str) -> str:
    """Normalize a trigger label for use in a file name.

    Labels come from the YAML and usually already carry a leading underscore
    ("_Control"), but they are free text, so strip the separators off and let
    summary_name() put exactly one back. Characters that are illegal in a file
    name are replaced rather than allowed to blow up the write.
    """
    cleaned = re.sub(r'[<>:"/\\|?*]', "-", suffix).strip().strip("_")
    return cleaned


def summary_name(prefix: str, stem: str, suffix: str, ext: str = ".xlsx") -> str:
    """'<prefix><stem>[_<suffix>]<ext>' -- unsuffixed name when no label."""
    cleaned = clean_suffix(suffix)
    return f"{prefix}{stem}_{cleaned}{ext}" if cleaned else f"{prefix}{stem}{ext}"


def condition_label(suffix: str) -> str:
    """Human-readable condition name for log lines and file names."""
    return clean_suffix(suffix) or "no label"


def stacked_title(*parts) -> str:
    """Title lines, skipping the empty ones.

    An unlabelled trigger has no condition name, and printing a stand-in for
    it ("no label") on a plot only adds noise -- the line is left out instead.
    """
    return "\n".join(line for line in
                      (" ".join(str(word) for word in
                                (part if isinstance(part, (list, tuple)) else [part])
                                if str(word).strip()).strip()
                       for part in parts) if line)


def is_binarization_source(suffix: str) -> bool:
    """True if this suffix is the condition the ROI filter is taken from."""
    wanted = clean_suffix(BINARIZATION_SOURCE)
    return bool(wanted) and clean_suffix(suffix).lower() == wanted.lower()


def apply_binarization(conditions):
    """Decide each folder's surviving ROIs, then cut the metric matrices.

    With several conditions present, the ROI filter is taken from the
    BINARIZATION_SOURCE condition and applied unchanged to the others. ROI
    indices line up because the conditions are different triggers over the
    same movie and the same ROI set, so a folder is matched to its control by
    (instance, source_token) - i.e. the same acquisition.

    A folder keeps its own filter when there is no control to inherit from, or
    when the two disagree about the number of ROIs.
    """
    # mask from the control condition, per acquisition
    donors = {}
    for suffix, instances in conditions.items():
        if not is_binarization_source(suffix):
            continue
        for instance, folders in instances.items():
            for token, raw in folders:
                donors[(instance, token)] = raw["own_mask"]

    for suffix, instances in conditions.items():
        inherits = donors and not is_binarization_source(suffix)
        for instance, folders in instances.items():
            for token, raw in folders:
                mask = raw["own_mask"]
                if inherits:
                    donor = donors.get((instance, token))
                    if donor is None:
                        print(f"  ! {instance} {token} [{condition_label(suffix)}]: "
                              f"no {clean_suffix(BINARIZATION_SOURCE)} counterpart; "
                              f"keeping its own binarization")
                    elif len(donor) != len(mask):
                        print(f"  ! {instance} {token} [{condition_label(suffix)}]: "
                              f"{len(mask)} ROIs vs {len(donor)} in "
                              f"{clean_suffix(BINARIZATION_SOURCE)}; "
                              f"keeping its own binarization")
                    else:
                        mask = donor

                raw["keep_mask"] = mask
                for metric in METRICS:
                    raw[metric] = raw[metric][mask.values]
    return conditions


def collect_conditions(work_folder: Path):
    """Map suffix -> instance -> [(source_token, {metric: filtered DataFrame})].

    Calculation folders are grouped first by their suffix (the trigger label,
    i.e. the experimental condition) and only then by day, so that two
    conditions recorded on the same date stay completely separate.

    Empty if the working folder has no valid calculation folder.
    """
    conditions = {}
    for calc_dir in sorted(work_folder.iterdir()):
        if not calc_dir.is_dir():
            continue
        m = DIR_RE.match(calc_dir.name)
        if not m:
            continue  # ignore non-matching directories
        instance, suffix = m.group(1), m.group(2)

        out_dir = find_outputs_dir(calc_dir)
        if out_dir is None:
            print(f"  ! no outputs_ folder in {calc_dir.name}; skipping")
            continue

        print(f"  processing {calc_dir.name}"
              f"  ->  instance '{instance}', condition '{condition_label(suffix)}'")
        mats = load_matrices_for_folder(out_dir)
        if mats is None:
            continue

        conditions.setdefault(suffix, {}).setdefault(instance, []).append(
            (source_token(calc_dir.name, instance, suffix), mats)
        )
    return conditions


def make_row(label, vals):
    """Build one output row (with ratios and blank separator columns)."""
    ampl_ac, ampl_c = vals["Ampl A&C"], vals["Ampl C"]
    auc_ac, auc_c   = vals["AUC A&C"],  vals["AUC C"]
    ampl_ratio = ampl_ac / ampl_c if ampl_c else ""
    auc_ratio  = auc_ac  / auc_c  if auc_c  else ""
    return [label, ampl_ac, ampl_c, ampl_ratio, "", auc_ac, auc_c, auc_ratio,
            "", vals["AP A&C"], vals["AP C"]]


def day_rows(instances):
    """One row per day (instance): grand mean over every surviving cell."""
    rows = []
    for instance in sorted(instances):
        vals = {}
        for metric in METRICS:
            cells = [mats[metric].to_numpy().ravel() for _, mats in instances[instance]]
            pooled = np.concatenate(cells) if cells else np.array([])
            vals[metric] = np.nanmean(pooled) if pooled.size else float("nan")
        # AP success rate: mean of per-ROI rates across all folders, scaled to %.
        for ap_key in ("AP A&C", "AP C"):
            rates = [mats[ap_key].to_numpy() for _, mats in instances[instance]]
            pooled = np.concatenate(rates) if rates else np.array([])
            vals[ap_key] = np.nanmean(pooled) * 100 if pooled.size else float("nan")
        rows.append(make_row(instance, vals))
    return rows


def roi_rows(folders):
    """One row per ROI (all ROIs, not just surviving ones).

    Surviving ROIs get Ampl, AUC and AP values.
    Filtered-out ROIs get a label and AP values only; Ampl/AUC cells are empty.
    """
    rows = []
    for token, mats in folders:
        keep_mask = mats["keep_mask"]
        surviving_means = {metric: mats[metric].mean(axis=1) for metric in METRICS}
        ap_ac = mats["AP A&C"]
        ap_c  = mats["AP C"]

        for roi_idx in ap_ac.index:  # all ROIs
            label      = f"{token}_roi{roi_idx}"
            ap_ac_val  = float(ap_ac.loc[roi_idx]) * 100
            ap_c_val   = float(ap_c.loc[roi_idx])  * 100

            if keep_mask.loc[roi_idx]:
                vals = {metric: surviving_means[metric].loc[roi_idx] for metric in METRICS}
                vals["AP A&C"] = ap_ac_val
                vals["AP C"]   = ap_c_val
                rows.append(make_row(label, vals))
            else:
                # Filtered ROI: empty Ampl/AUC columns, AP columns still filled.
                rows.append([label, "", "", "", "", "", "", "", "", ap_ac_val, ap_c_val])
    return rows


def per_roi_values(folders):
    """{metric: {day: array of per-ROI values}} over the surviving ROIs.

    `folders` is a list of (group_key, mats). The values are exactly the ones a
    local per-ROI summary puts in its rows -- the mean across a ROI's epochs,
    and the ratio of those two means -- so the histograms describe the same
    numbers as the sheet above them.

    Folders sharing a key are pooled, and the key is the day: several
    acquisitions of the same day are one n with its repetitions inside it.
    That is what the CDF overlay averages over, so a day with two fields
    weighs the same as a day with one, exactly as it does in the bar plots.
    """
    values = {metric: {} for metric in HIST_METRICS}
    for key, mats in folders:
        means = {metric: mats[metric].mean(axis=1) for metric in METRICS}
        per_metric = dict(means)
        # Ratio of the per-ROI means, as in make_row(); a zero denominator is
        # dropped rather than turned into an infinity.
        per_metric["Ampl A&C/C"] = means["Ampl A&C"] / means["Ampl C"].replace(0, np.nan)
        per_metric["AUC A&C/C"] = means["AUC A&C"] / means["AUC C"].replace(0, np.nan)
        for metric in HIST_METRICS:
            arr = np.asarray(per_metric[metric], dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                pooled = values[metric].get(key)
                values[metric][key] = arr if pooled is None else np.concatenate([pooled, arr])
    return values


def roi_values_for(instances, only=None):
    """per_roi_values() keyed by day, so one day is one n.

    `only` keeps just those days, which the cross-condition file needs so its
    histograms cover exactly the days its paired rows do.
    """
    folders = [(instance, mats)
               for instance in sorted(instances)
               if only is None or instance in only
               for _, mats in instances[instance]]
    return per_roi_values(folders)


def save_figure(fig, png_path: Path):
    """Write a matplotlib figure as PNG and, for Excel, as its SVG twin."""
    fig.savefig(png_path, dpi=HIST_DPI)
    if EMBED_SVG:
        fig.savefig(png_path.with_suffix(".svg"))


def mean_cdf(arrays, grid):
    """Mean of each array's empirical CDF on a shared x-grid.

    Every CDF runs 0->1, so averaging them weights each day equally,
    independent of bouton count. None when there is nothing to average.
    """
    cdfs = [np.searchsorted(np.sort(v), grid, side="right") / v.size
            for v in arrays if v.size]
    return np.mean(cdfs, axis=0) if cdfs else None


def hist_range(arrays):
    """Shared (pooled values, lo, hi, n_outside) for a set of arrays.

    Full range by default; the percentile window when outliers would make the
    panel unreadable (see HIST_CLIP_SHARE). n_outside counts the values left
    out of the view, not out of the statistics.
    """
    pooled = np.concatenate([v for v in arrays if v.size])
    lo, hi = float(pooled.min()), float(pooled.max())
    outside = 0

    if pooled.size >= HIST_CLIP_MIN_N:
        p_lo, p_hi = np.percentile(pooled, [HIST_CLIP_PCT, 100 - HIST_CLIP_PCT])
        if p_hi > p_lo and (p_hi - p_lo) < HIST_CLIP_SHARE * (hi - lo):
            lo, hi = float(p_lo), float(p_hi)
            outside = int(np.count_nonzero((pooled < lo) | (pooled > hi)))

    if hi <= lo:  # a single distinct value would give zero-width bins
        lo, hi = lo - 0.5, hi + 0.5
    return pooled, lo, hi, outside


def histogram_png(groups, metric, png_path: Path, title: str = "") -> bool:
    """Pooled histogram of one metric + one CDF per day, their mean bolded.

    The per-folder figure of make_histograms.py, as a single panel. False when
    there is no data to draw.
    """
    arrays = [v for v in groups.get(metric, {}).values() if v.size]
    if not arrays:
        return False

    pooled, lo, hi, outside = hist_range(arrays)
    edges = np.linspace(lo, hi, HIST_BINS_N + 1)
    grid = np.linspace(lo, hi, HIST_GRID_N)

    fig, ax = plt.subplots(figsize=HIST_SIZE)
    ax.hist(pooled, bins=edges, color="C0", edgecolor="black", linewidth=0.4)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(metric)
    ax.set_ylabel("bouton count")
    ax.set_title(f"{title}\n{metric}: {pooled.size} boutons, n={len(arrays)}"
                 + (f", {outside} off-scale" if outside else ""), fontsize=9)

    # CDFs on a twin 0-1 axis: one thin line per day, their mean bold.
    axc = ax.twinx()
    for v in arrays:
        axc.plot(grid, np.searchsorted(np.sort(v), grid, side="right") / v.size,
                 color="C1", lw=0.8, alpha=0.3)
    curve = mean_cdf(arrays, grid)
    if curve is not None:
        axc.plot(grid, curve, color="C1", lw=2.6, label="mean cumulative")
        axc.legend(loc="lower right", fontsize=7)
    axc.set_ylim(0, 1)
    axc.set_ylabel("cumulative (norm. to 1)")

    if metric in RATIO_COLUMNS:  # a ratio of 1 means no effect
        ax.axvline(1.0, color="red", linestyle="--", lw=1)

    fig.tight_layout()
    save_figure(fig, png_path)
    plt.close(fig)
    return True


def overlay_png(per_condition, metric, png_path: Path, title: str = "") -> bool:
    """Condition-vs-condition distribution of one metric.

    Step histograms behind, one bold mean CDF per condition in front -- the
    folder-comparison figure of make_histograms.py. `per_condition` maps a
    condition label to its per_roi_values() result. False when nothing to draw.
    """
    present = [(label, [v for v in values.get(metric, {}).values() if v.size])
               for label, values in per_condition.items()]
    present = [(label, arrays) for label, arrays in present if arrays]
    if not present:
        return False

    _, lo, hi, outside = hist_range([v for _, arrays in present for v in arrays])
    edges = np.linspace(lo, hi, HIST_BINS_N + 1)
    grid = np.linspace(lo, hi, HIST_GRID_N)

    fig, ax = plt.subplots(figsize=HIST_SIZE)
    axh = ax.twinx()  # counts behind, cumulative in front
    axh.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)

    for i, (label, arrays) in enumerate(present):
        color = f"C{i}"
        axh.hist(np.concatenate(arrays), bins=edges, histtype="stepfilled",
                 color=color, lw=1.0, alpha=0.3)
        # one thin curve per day behind the condition's mean, as the
        # per-condition histograms draw them
        for values in arrays:
            ax.plot(grid, np.searchsorted(np.sort(values), grid, side="right") / values.size,
                    color=color, lw=0.8, alpha=0.3)
        curve = mean_cdf(arrays, grid)
        if curve is not None:
            ax.plot(grid, curve, color=color, lw=2.4,
                    label=f"{label} (n={len(arrays)})")

    axh.set_ylabel("bouton count")
    ax.set_xlabel(RATIO_AXIS_LABELS.get(metric, metric))
    ax.set_ylabel("cumulative (norm. to 1)")
    ax.set_ylim(0, 1)
    ax.set_xlim(lo, hi)
    ax.set_title(f"{title}\n{metric}" + (f" ({outside} off-scale)" if outside else ""),
                 fontsize=9)
    ax.legend(loc="lower right", fontsize=7)
    if metric in RATIO_COLUMNS:
        ax.axvline(1.0, color="red", linestyle="--", lw=1)

    fig.tight_layout()
    save_figure(fig, png_path)
    plt.close(fig)
    return True


def rows_for(px: int) -> int:
    """How many default-height Excel rows an image of `px` pixels covers."""
    return int(math.ceil(px / PX_PER_ROW))


def place_image_grid(ws, pngs, first_row: int, cols=HIST_COLS) -> int:
    """Anchor rendered PNGs in a grid, left to right then down.

    Images are embedded at 1/IMG_SCALE so they stay readable next to the data.
    Returns the first free row below the grid.
    """
    row, band = first_row, 0
    for i, png in enumerate(pngs):
        col = i % len(cols)
        if col == 0 and i:
            row += band + 1
            band = 0
        img = XLImage(str(png))
        img.width = img.width // IMG_SCALE
        img.height = img.height // IMG_SCALE
        band = max(band, rows_for(img.height))
        ws.add_image(img, f"{cols[col]}{row}")
    return row + band + 1


def add_histogram_band(ws, tmp: Path, first_row: int, render, metrics=HIST_METRICS,
                       prefix: str = "") -> int:
    """Render one panel per metric and place them in a grid under the bar plots.

    `render(metric, png_path)` draws a single panel and returns False when that
    metric has no data; those panels are left out instead of being blank.
    """
    pngs = []
    for i, metric in enumerate(metrics):
        png = tmp / f"{prefix}hist_{i}.png"
        if render(metric, png):
            pngs.append(png)
    if not pngs:
        return first_row
    return place_image_grid(ws, pngs, first_row)


def group_barplot(groups, names, png_path: Path, y_label: str = "", plot_title: str = ""):
    """Run the paired test AutoStatLib picks for N groups and save a bar plot.

    Two groups give a Wilcoxon / paired t-test as before; three or more give a
    repeated-measures ANOVA (or its non-parametric counterpart) plus the
    post-hoc matrix, which the plot annotates.

    Returns the exact p-value (or None if the test could not run).
    """
    analysis = AutoStatLib.StatisticalAnalysis(
        [list(group) for group in groups],
        paired=True,
        groups_name=list(names),
        verbose=False,
    )
    analysis.RunAuto()
    res = analysis.GetResult() or {}
    p_exact = res.get("p_value_exact")

    plot = AutoStatLib.BarStatPlot(
        data_groups=res['Samples'],
        plot_title=plot_title,
        y_label=y_label,
        figure_scale_factor=1.0,
        **res,
    )
    plot.plot()
    plot.save(str(png_path), format="png", dpi=150, transparent=False)
    if EMBED_SVG:
        plot.save(str(Path(png_path).with_suffix(".svg")), format="svg",
                  dpi=150, transparent=False)
    plot.close()
    return p_exact


def wilcoxon_barplot(group_left, group_right, names, png_path: Path,
                     y_label: str = "", plot_title: str = ""):
    """Two-group shorthand for group_barplot()."""
    return group_barplot([group_left, group_right], names, png_path,
                         y_label=y_label, plot_title=plot_title)


def cols_for(px: int) -> int:
    """How many default-width Excel columns an image of `px` pixels covers."""
    return int(math.ceil(px / DEFAULT_COL_PX))


def condition_comparisons(n_conditions: int):
    """Bar plot spec per ratio metric: which row columns hold the N groups.

    The cross-condition rows are
        instance | <ratio 1> per condition | blank | <ratio 2> per condition
    so each metric's groups sit in one contiguous run.
    """
    specs = []
    for i, metric in enumerate(RATIO_COLUMNS):
        start = 1 + i * (n_conditions + 1)
        spec = {"cols": list(range(start, start + n_conditions)),
                "y_label": RATIO_AXIS_LABELS.get(metric, metric)}
        if n_conditions == 2:
            spec["anchor"] = CONDITION_ANCHORS[i]
        specs.append(spec)
    return specs


def sheet_name(text: str, used) -> str:
    """An Excel-legal, unique sheet title (31 chars, no []:*?/\\)."""
    name = re.sub(r"[\[\]:*?/\\]", "-", text).strip()[:31] or "sheet"
    base, n = name, 2
    while name in used:
        tail = f"~{n}"
        name, n = base[:31 - len(tail)] + tail, n + 1
    used.add(name)
    return name


def add_barplot_band(ws, rows, comparisons, tmp: Path, first_row: int,
                     plot_title: str = "", prefix: str = "") -> int:
    """Run each paired test and embed its bar plot side by side.

    A comparison names its groups either as "left"/"right" or, for three or
    more, as a list of row columns in "cols". Without an explicit "anchor" the
    plots are laid out left to right by their rendered width.

    Returns the first free row below the band, so the histograms know where to
    start no matter how tall the plots came out.
    """
    band, next_col = 0, FIRST_PLOT_COL
    for i, cmp in enumerate(comparisons):
        cols = cmp.get("cols") or [cmp["left"], cmp["right"]]
        # Keep only rows where every group is numeric (drops the empty cells of
        # filtered ROIs, and days a condition never saw).
        kept = [r for r in rows
                if all(isinstance(r[col], (int, float)) for col in cols)]
        groups = [[r[col] for r in kept] for col in cols]
        png = tmp / f"{prefix}plot_{i}.png"
        p = group_barplot(groups, cmp["names"], png,
                          y_label=cmp["y_label"], plot_title=plot_title)
        img = XLImage(str(png))
        img.width  = img.width  // IMG_SCALE
        img.height = img.height // IMG_SCALE
        band = max(band, rows_for(img.height))

        col = column_index_from_string(cmp["anchor"]) if cmp.get("anchor") else next_col
        ws.add_image(img, f"{get_column_letter(col)}{first_row}")
        next_col = col + cols_for(img.width) + 1

        label = " vs ".join(cmp["names"])
        print(f"      {label}: paired test p={p:.4g}" if p is not None else
              f"      {label}: paired test skipped")
    return first_row + band + 1


def inject_svg_blip(drawing_xml: str, png_rid: str, svg_rid: str) -> str:
    """Hang an <asvg:svgBlip> off the picture that embeds `png_rid`."""
    pattern = re.compile(r'<a:blip([^>]*?)r:embed="%s"([^>]*?)/>' % re.escape(png_rid))

    def attach(m):
        return (f'<a:blip{m.group(1)}r:embed="{png_rid}"{m.group(2)}>'
                f'<a:extLst><a:ext uri="{SVG_EXT_URI}">'
                f'<asvg:svgBlip xmlns:asvg="{SVG_NS}" r:embed="{svg_rid}"/>'
                f'</a:ext></a:extLst></a:blip>')

    return pattern.sub(attach, drawing_xml, count=1)


def media_part(target: str) -> str:
    """Package path a drawing relationship target points at.

    openpyxl writes "/xl/media/image1.png"; the format also allows
    "../media/image1.png" relative to xl/drawings/.
    """
    if target.startswith("/"):
        return target.lstrip("/")
    return "xl/" + target.replace("../", "")


def attach_svgs(xlsx_path: Path, tmp: Path) -> int:
    """Give every chart of a saved workbook its vector twin.

    openpyxl can embed raster images only, so the charts went in as PNGs. Each
    PNG is matched back to the SVG rendered beside it (by content, so the match
    survives however openpyxl numbered the media parts), the SVG is added to
    the package, and the picture's blip gets the svgBlip extension pointing at
    it. Excel 2016+ then draws the vector and older versions the PNG.

    Returns how many charts got their twin.
    """
    twins = {}
    for svg in tmp.glob("*.svg"):
        png = svg.with_suffix(".png")
        if png.exists():
            twins[hashlib.sha1(png.read_bytes()).hexdigest()] = svg
    if not twins:
        return 0

    with zipfile.ZipFile(xlsx_path) as zin:
        parts = {name: zin.read(name) for name in zin.namelist()}

    svg_for_media = {}
    for name, data in parts.items():
        if name.startswith("xl/media/") and name.endswith(".png"):
            twin = twins.get(hashlib.sha1(data).hexdigest())
            if twin is not None:
                svg_for_media[name] = twin
    if not svg_for_media:
        return 0

    attached = 0
    rels_names = [n for n in parts
                  if re.fullmatch(r"xl/drawings/_rels/drawing\d+\.xml\.rels", n)]
    for rels_name in rels_names:
        rels = parts[rels_name].decode("utf-8")
        drawing_name = "xl/drawings/" + Path(rels_name).name[: -len(".rels")]
        if drawing_name not in parts:
            continue
        drawing = parts[drawing_name].decode("utf-8")

        next_id = max((int(n) for n in re.findall(r'Id="rId(\d+)"', rels)), default=0) + 1
        added = []
        # attribute order is openpyxl's business, so read each one separately
        for rel in re.findall(r"<Relationship\b[^>]*?/>", rels):
            rid = re.search(r'Id="([^"]+)"', rel)
            target = re.search(r'Target="([^"]+)"', rel)
            if not (rid and target):
                continue
            twin = svg_for_media.get(media_part(target.group(1)))
            if twin is None:
                continue
            svg_target = target.group(1)[: -len(".png")] + ".svg"
            parts[media_part(svg_target)] = twin.read_bytes()
            svg_rid = f"rId{next_id}"
            next_id += 1
            added.append((svg_rid, svg_target))  # same style as the PNG's target
            drawing = inject_svg_blip(drawing, rid.group(1), svg_rid)
            attached += 1

        if added:
            parts[rels_name] = rels.replace(
                "</Relationships>",
                "".join(f'<Relationship Id="{rid}" Type="{IMAGE_REL}" Target="{target}"/>'
                        for rid, target in added) + "</Relationships>",
            ).encode("utf-8")
            parts[drawing_name] = drawing.encode("utf-8")

    if not attached:
        return 0

    # the package has to declare what an .svg part is
    types = parts["[Content_Types].xml"].decode("utf-8")
    if 'Extension="svg"' not in types:
        types = re.sub(r"(<Types[^>]*>)",
                       r'\1<Default Extension="svg" ContentType="image/svg+xml"/>',
                       types, count=1)
        parts["[Content_Types].xml"] = types.encode("utf-8")

    # rewrite the package, content types first as Excel expects
    order = ["[Content_Types].xml"] + [n for n in parts if n != "[Content_Types].xml"]
    with zipfile.ZipFile(xlsx_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for name in order:
            zout.writestr(name, parts[name])
    return attached


def write_xlsx(id_label, rows, out_path: Path, plot_title: str = "", hist_groups=None,
               hist_title: str = ""):
    """Write one summary sheet as .xlsx, then run Wilcoxon tests on the
    Ampl (cols 2-3) and AUC (cols 6-7) pairs and embed the bar plots just
    below the last data row, under their respective columns.

    `hist_groups` is a per_roi_values() mapping; when given, the six metric
    histograms are dropped underneath the bar plots in a 3-wide grid. They
    describe boutons rather than the rows above, hence their own title.
    """
    header = [id_label, "Ampl A&C", "Ampl C", "Ampl A&C/C", "", "AUC A&C", "AUC C", "AUC A&C/C",
              "", "% AP success A&C", "% AP success C"]
    wb = Workbook()
    ws = wb.active
    ws.title = "summary"
    ws.append(header)
    for row in rows:
        ws.append(row)

    # Plots are anchored two rows below the last data row.
    anchor_row = ws.max_row + 2

    # Temp PNGs must survive until wb.save() reads them into the workbook.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        next_row = add_barplot_band(ws, rows, COMPARISONS, tmp, anchor_row, plot_title)
        if hist_groups:
            add_histogram_band(
                ws, tmp, next_row + 1,
                lambda metric, png: histogram_png(hist_groups, metric, png,
                                                  hist_title or plot_title),
            )
        wb.save(out_path)
        if EMBED_SVG:
            attach_svgs(out_path, tmp)


def fill_comparison_sheet(ws, block, tmp: Path, prefix: str = ""):
    """Lay one comparison out on a sheet: rows, bar plots, then the overlays."""
    labels = block["labels"]
    header = ["instance"] + [f"Ampl A&C/C {label}" for label in labels] + [""] \
        + [f"AUC A&C/C {label}" for label in labels]
    ws.append(header)
    for row in block["rows"]:
        ws.append(row)

    anchor_row = ws.max_row + 2
    # The group names are the conditions here; the metric is in the y label.
    comparisons = [dict(cmp, names=labels)
                   for cmp in condition_comparisons(len(labels))]

    next_row = add_barplot_band(ws, block["rows"], comparisons, tmp, anchor_row,
                                block.get("plot_title", ""), prefix=prefix)
    if block.get("hist_conditions"):
        add_histogram_band(
            ws, tmp, next_row + 1,
            lambda metric, png: overlay_png(block["hist_conditions"], metric, png,
                                            block.get("hist_title") or block.get("plot_title", "")),
            metrics=list(RATIO_COLUMNS),
            prefix=prefix,
        )


def write_comparison_xlsx(blocks, out_path: Path):
    """Write the cross-condition workbook: one sheet per comparison.

    With two conditions that is the single "comparison" sheet this file has
    always had. With three or more the first sheet holds every condition at
    once and one sheet per pair follows, so the pairwise view keeps the days
    that an all-conditions pairing has to drop.
    """
    wb = Workbook()
    used = set()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, block in enumerate(blocks):
            ws = wb.active if i == 0 else wb.create_sheet()
            ws.title = sheet_name(block["sheet"], used)
            print(f"    sheet '{ws.title}': {len(block['rows'])} paired day(s)")
            fill_comparison_sheet(ws, block, tmp, prefix=f"s{i}_")
        wb.save(out_path)
        if EMBED_SVG:
            attach_svgs(out_path, tmp)


def ordered_conditions(conditions):
    """Condition suffixes, the ROI-filter donor (the control) first."""
    return sorted(conditions,
                  key=lambda s: (not is_binarization_source(s), clean_suffix(s).lower()))


def comparison_rows(day_by_instance, subset):
    """Rows for one comparison: the days every condition in `subset` recorded.

    Returns (rows, shared days, days left out). A day only one condition saw
    has nothing to pair with, so it cannot take part.
    """
    shared = sorted(set.intersection(*(set(day_by_instance[s]) for s in subset)))
    seen = set().union(*(set(day_by_instance[s]) for s in subset))
    rows, used = [], []
    for instance in shared:
        # ratio by ratio, condition by condition -- the row layout of the sheet
        values = [day_by_instance[cond][instance][col]
                  for col in RATIO_COLUMNS.values()
                  for cond in subset]
        if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in values):
            continue
        n = len(subset)
        rows.append([instance] + values[:n] + [""] + values[n:])
        used.append(instance)
    return rows, used, sorted(seen - set(used))


def write_condition_comparisons(work_folder: Path, conditions):
    """One extra ROOT-level file comparing the applications of a folder.

    "<CondA>_vs_<CondB>[_vs_<CondC>...]_summary_for_<workfolder>.xlsx" puts the
    conditions' Ampl A&C/C and AUC A&C/C next to each other, one row per day,
    and tests them against each other. The numbers are the ones the
    per-condition summaries report for that day.

    Two conditions give the single sheet this file has always had. Three or
    more give an "all conditions" sheet -- a repeated-measures test over every
    condition at once -- followed by one sheet per pair, because a pair usually
    shares more days than all the conditions together do.

    Only days present in every condition of a sheet take part there, since the
    test is paired.
    """
    order = ordered_conditions(conditions)
    labels = {suffix: condition_label(suffix) for suffix in order}
    day_by_instance = {suffix: {row[0]: row for row in day_rows(conditions[suffix])}
                       for suffix in order}

    out_path = ROOT / (COMPARE_SEP.join(labels[suffix] for suffix in order)
                       + "_" + summary_name(SUMMARY_PREFIX, work_folder.name, ""))
    if out_path.exists():
        print(f"Skipping {out_path.name} (already exists)")
        return

    # every condition at once, then -- beyond two -- each pair on its own sheet
    subsets = [order]
    if len(order) > 2:
        subsets += [list(pair) for pair in itertools.combinations(order, 2)]

    blocks = []
    for subset in subsets:
        names = [labels[suffix] for suffix in subset]
        pair = " vs ".join(names)
        rows, used, dropped = comparison_rows(day_by_instance, subset)
        if not rows:
            print(f"  ! {pair}: no day with data in every condition; sheet left out")
            continue
        if dropped:
            print(f"  ! {pair}: {len(dropped)} day(s) not shared by all of them "
                  f"({', '.join(dropped)}); left out of the pairing")
        blocks.append({
            "sheet": "comparison" if len(order) == 2 else
                     ("all conditions" if subset is order else pair),
            "labels": names,
            "rows": rows,
            # the bar plot figure is narrow, so the folder name goes on the
            # wider histogram panels only
            "plot_title": f"Statistics on mice\n{pair}",
            "hist_title": f"{work_folder.name}\n{pair}",
            "hist_conditions": {labels[suffix]: roi_values_for(conditions[suffix], only=used)
                                for suffix in subset},
        })

    if not blocks:
        print(f"  ! {work_folder.name}: nothing to compare across conditions")
        return

    write_comparison_xlsx(blocks, out_path)
    print(f"Wrote {len(blocks)} comparison sheet(s) "
          f"[{' vs '.join(labels[suffix] for suffix in order)}] to {out_path}")


def main():
    if not ROOT.is_dir():
        sys.exit(f"Root folder not found: {ROOT}")

    candidates = [d for d in sorted(ROOT.iterdir()) if d.is_dir()]
    processed = 0

    for work_folder in candidates:
        conditions = collect_conditions(work_folder)
        if not conditions:
            # No valid structure inside -> ignore this folder.
            continue
        # one entry per acquisition before anything is pooled or paired
        resolve_acquisitions(conditions)
        apply_binarization(conditions)

        # Each suffix (condition) is summarized on its own, start to finish.
        for suffix in sorted(conditions):
            instances = conditions[suffix]
            condition = condition_label(suffix)   # for the log lines
            label = clean_suffix(suffix)          # empty when the trigger had none

            # ROOT-level summary: one row per day, for this condition only.
            root_path = ROOT / summary_name(SUMMARY_PREFIX, work_folder.name, suffix)
            if root_path.exists():
                print(f"Skipping {root_path.name} (already exists)")
            else:
                # Histograms pool the boutons of every day this file covers.
                write_xlsx("instance", day_rows(instances), root_path,
                           plot_title=stacked_title("Statistics on mice", label),
                           hist_groups=roi_values_for(instances),
                           hist_title=stacked_title(work_folder.name, label))
                print(f"Wrote {len(instances)} day(s) [{condition}] to {root_path}")

            # Local per-day summaries (rows = ROIs) inside the working folder.
            for instance in sorted(instances):
                local_path = work_folder / summary_name(LOCAL_PREFIX, instance, suffix)
                if local_path.exists():
                    print(f"    Skipping {local_path.name} (already exists)")
                    continue
                rows = roi_rows(instances[instance])
                write_xlsx("ROI", rows, local_path,
                           plot_title=stacked_title("Statistic on rois", [instance, label]),
                           hist_groups=roi_values_for({instance: instances[instance]}),
                           hist_title=stacked_title([instance, label]))
                print(f"    local: {len(rows)} ROI(s) [{condition}] -> {local_path}")

        # Several applications in one experiment: compare them against each other.
        if len(conditions) > 1:
            write_condition_comparisons(work_folder, conditions)

        processed += 1
        print()

    if processed == 0:
        print(f"No working folders with the expected structure found under {ROOT}")
    else:
        print(f"Done: processed {processed} working folder(s).")


if __name__ == "__main__":
    main()

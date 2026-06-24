"""
Summarize Amplitude / AUC CSV data across several working folders.

Run it pointing at a ROOT directory that contains one or more working folders:

    ROOT/
        <WorkFolder A>/
            <YYYY_MM_DD>[_M#]_Field..._CALCULATIONS_auto_/
                outputs_.../
                    _Ampl_A&C_rows-roi_cols-epoch__auto_.csv
                    _Ampl_C_rows-roi_cols-epoch__auto_.csv
                    _AUC_A&C_rows-roi_cols-epoch__auto_.csv
                    _AUC_C_rows-roi_cols-epoch__auto_.csv
                    _Bin_C_rows-roi_cols-epoch__auto_.csv
        <WorkFolder B>/ ...

Two kinds of output are produced (all as .xlsx):

  * ROOT/summary_for_<workfolder>.xlsx  -- one row per day (instance), the
    grand mean over every surviving cell of that day.
  * <workfolder>/summary_<instance>.xlsx -- a local, per-day summary whose rows
    are individual ROIs (each value is the mean across that ROI's epochs).

Both share the same column layout:
    <id> | Ampl A&C | Ampl C | Ampl C/A&C | <blank> | AUC A&C | AUC C | AUC C/A&C

Folders without the expected structure are ignored.

The "instance" identifier is everything before "_Field" in a calculation
folder's name, i.e. the date plus an optional _M1 / _M2 suffix. Several folders
can share the same instance (e.g. two _Field acquisitions on the same date);
their ROIs are pooled together for that day.

For every output folder, rows (ROIs) whose TRUE-rate in Bin_C is below THRESHOLD
are dropped from the Ampl / AUC matrices before anything is computed.
"""

import re
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: render plots to files, never open a window

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage

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
ROI_THRESHOLD = 0.37

# Metric -> csv file name inside each outputs_ folder. Order defines column order.
METRIC_FILES = {
    "Ampl A&C": "_Ampl_A&C_rows-roi_cols-epoch__auto_.csv",
    "Ampl C":   "_Ampl_C_rows-roi_cols-epoch__auto_.csv",
    "AUC A&C":  "_AUC_A&C_rows-roi_cols-epoch__auto_.csv",
    "AUC C":    "_AUC_C_rows-roi_cols-epoch__auto_.csv",
}
METRICS = list(METRIC_FILES)
SNR_FILE = "_SNR_C_rows-roi_cols-epoch__auto_.csv"

# A valid calculation folder: date (+ optional _M#) followed by _Field ... CALCULATIONS_auto_
DIR_RE = re.compile(r"^(\d{4}_\d{2}_\d{2}(?:_M\d+)?)_Field.*CALCULATIONS_auto_$")

CALC_SUFFIX = "_CALCULATIONS_auto_"

# Paired comparisons to test + plot in every summary.
# (left/right are 0-based indices into an output row; anchor is the Excel column
#  letter under which the plot is dropped -- "B" = cols 2-3, "F" = cols 6-7.)
COMPARISONS = [
    {"left": 1, "right": 2, "names": ["Ampl A&C", "Ampl C"], "anchor": "B"},
    {"left": 5, "right": 6, "names": ["AUC A&C", "AUC C"],   "anchor": "F"},
]


def read_matrix(path: Path) -> pd.DataFrame:
    """Read a head-less, index-less numeric matrix (rows=ROI, cols=epoch)."""
    return pd.read_csv(path, header=None)


def snr_to_bin(path: Path) -> pd.DataFrame:
    """Read the SNR matrix and binarize: True where SNR >= SNR_THRESHOLD."""
    snr = pd.read_csv(path, header=None, dtype=float)
    return snr >= SNR_THRESHOLD


def filtered_matrices_for_folder(out_dir: Path):
    """Return {metric: filtered DataFrame} for one outputs_ folder.

    ROIs (rows) below the Bin_C TRUE-rate threshold are dropped; the surviving
    rows keep their original index so each ROI stays identifiable.

    Returns None (and warns) if a required file is missing or shapes mismatch.
    """
    snr_path = out_dir / SNR_FILE
    if not snr_path.exists():
        print(f"  ! missing {SNR_FILE} in {out_dir.name}; skipping folder")
        return None

    bin_df = snr_to_bin(snr_path)
    # Per-ROW (per-ROI) fraction of cells above SNR threshold; keep rows at/above ROI_THRESHOLD.
    keep_mask = bin_df.mean(axis=1) >= ROI_THRESHOLD

    result = {}
    for metric, fname in METRIC_FILES.items():
        fpath = out_dir / fname
        if not fpath.exists():
            print(f"  ! missing {fname} in {out_dir.name}; skipping folder")
            return None

        mat = read_matrix(fpath)
        if mat.shape[0] != bin_df.shape[0]:
            print(f"  ! shape mismatch for {fname} in {out_dir.name}; skipping folder")
            return None

        result[metric] = mat[keep_mask.values]
    return result


def find_outputs_dir(calc_dir: Path):
    """Return the single outputs_ subfolder inside a calculation folder."""
    for sub in calc_dir.iterdir():
        if sub.is_dir() and sub.name.startswith("outputs_"):
            return sub
    return None


def source_token(calc_dir_name: str, instance: str) -> str:
    """Short, traceable tag for an acquisition folder (e.g. 'Field_1_0001...')."""
    token = calc_dir_name
    if token.endswith(CALC_SUFFIX):
        token = token[: -len(CALC_SUFFIX)]
    if token.startswith(instance + "_"):
        token = token[len(instance) + 1:]
    return token


def collect_instances(work_folder: Path):
    """Map instance -> list of (source_token, {metric: filtered DataFrame}).

    Empty if the working folder has no valid instance.
    """
    instances = {}
    for calc_dir in sorted(work_folder.iterdir()):
        if not calc_dir.is_dir():
            continue
        m = DIR_RE.match(calc_dir.name)
        if not m:
            continue  # ignore non-matching directories
        instance = m.group(1)

        out_dir = find_outputs_dir(calc_dir)
        if out_dir is None:
            print(f"  ! no outputs_ folder in {calc_dir.name}; skipping")
            continue

        print(f"  processing {calc_dir.name}  ->  instance '{instance}'")
        mats = filtered_matrices_for_folder(out_dir)
        if mats is None:
            continue

        instances.setdefault(instance, []).append(
            (source_token(calc_dir.name, instance), mats)
        )
    return instances


def make_row(label, vals):
    """Build one output row (with ratios and the blank separator column)."""
    ampl_ac, ampl_c = vals["Ampl A&C"], vals["Ampl C"]
    auc_ac, auc_c = vals["AUC A&C"], vals["AUC C"]
    ampl_ratio = ampl_ac / ampl_c if ampl_c else ""
    auc_ratio = auc_ac / auc_c if auc_c else ""
    return [label, ampl_ac, ampl_c, ampl_ratio, "", auc_ac, auc_c, auc_ratio]


def day_rows(instances):
    """One row per day (instance): grand mean over every surviving cell."""
    rows = []
    for instance in sorted(instances):
        vals = {}
        for metric in METRICS:
            cells = [mats[metric].to_numpy().ravel() for _, mats in instances[instance]]
            pooled = np.concatenate(cells) if cells else np.array([])
            vals[metric] = np.nanmean(pooled) if pooled.size else float("nan")
        rows.append(make_row(instance, vals))
    return rows


def roi_rows(folders):
    """One row per surviving ROI: value = mean across that ROI's epochs."""
    rows = []
    for token, mats in folders:
        means = {metric: mats[metric].mean(axis=1) for metric in METRICS}
        for roi_idx in means["Ampl A&C"].index:
            vals = {metric: means[metric].loc[roi_idx] for metric in METRICS}
            rows.append(make_row(f"{token}_roi{roi_idx}", vals))
    return rows


def wilcoxon_barplot(group_left, group_right, names, png_path: Path):
    """Run a paired Wilcoxon test on two columns and save a bar plot.

    Returns the exact p-value (or None if the test could not run).
    """
    analysis = AutoStatLib.StatisticalAnalysis(
        [list(group_left), list(group_right)],
        paired=True,
        groups_name=names,
        verbose=False,
    )
    analysis.RunWilcoxon()
    res = analysis.GetResult() or {}
    p_exact = res.get("p_value_exact")

    plot = AutoStatLib.BarStatPlot(
        data_groups=[list(group_left), list(group_right)],
        p_value_exact=p_exact,
        Test_Name=res.get("Test_Name", "Wilcoxon signed-rank test"),
        Paired_Test_Applied=True,
        Groups_Name=names,
        y_label=names[0].split()[0],  # "Ampl" / "AUC"
        figure_scale_factor=1.0,
    )
    plot.plot()
    plot.save(str(png_path), format="png", dpi=150, transparent=False)
    plot.close()
    return p_exact


def write_xlsx(id_label, rows, out_path: Path):
    """Write one summary sheet as .xlsx, then run Wilcoxon tests on the
    Ampl (cols 2-3) and AUC (cols 6-7) pairs and embed the bar plots just
    below the last data row, under their respective columns."""
    header = [id_label, "Ampl A&C", "Ampl C", "Ampl A&C/C", "", "AUC A&C", "AUC C", "AUC A&C/C"]
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
        for i, cmp in enumerate(COMPARISONS):
            left = [r[cmp["left"]] for r in rows]
            right = [r[cmp["right"]] for r in rows]
            png = tmp / f"plot_{i}.png"
            p = wilcoxon_barplot(left, right, cmp["names"], png)
            img = XLImage(str(png))
            img.width  = img.width  // 2
            img.height = img.height // 2
            ws.add_image(img, f"{cmp['anchor']}{anchor_row}")
            print(f"      {cmp['names'][0]} vs {cmp['names'][1]}: "
                  f"Wilcoxon p={p:.4g}" if p is not None else
                  f"      {cmp['names'][0]} vs {cmp['names'][1]}: Wilcoxon (no p)")
        wb.save(out_path)


def main():
    if not ROOT.is_dir():
        sys.exit(f"Root folder not found: {ROOT}")

    candidates = [d for d in sorted(ROOT.iterdir()) if d.is_dir()]
    processed = 0

    for work_folder in candidates:
        instances = collect_instances(work_folder)
        if not instances:
            # No valid structure inside -> ignore this folder.
            continue

        # ROOT-level summary: one row per day.
        root_path = ROOT / f"{SUMMARY_PREFIX}{work_folder.name}.xlsx"
        write_xlsx("instance", day_rows(instances), root_path)
        print(f"Wrote {len(instances)} day(s) to {root_path}")

        # Local per-day summaries (rows = ROIs) inside the working folder.
        for instance in sorted(instances):
            rows = roi_rows(instances[instance])
            local_path = work_folder / f"{LOCAL_PREFIX}{instance}.xlsx"
            write_xlsx("ROI", rows, local_path)
            print(f"    local: {len(rows)} ROI(s) -> {local_path}")

        processed += 1
        print()

    if processed == 0:
        print(f"No working folders with the expected structure found under {ROOT}")
    else:
        print(f"Done: processed {processed} working folder(s).")


if __name__ == "__main__":
    main()

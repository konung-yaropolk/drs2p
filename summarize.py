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

# A valid calculation folder: date (+ optional _M#) followed by _Field ... CALCULATIONS_auto_
DIR_RE = re.compile(r"^(\d{4}_\d{2}_\d{2}(?:_M\d+)?)_Field.*CALCULATIONS_auto_$")

CALC_SUFFIX = "_CALCULATIONS_auto_"

# Paired comparisons to test + plot in every summary.
# (left/right are 0-based indices into an output row; anchor is the Excel column
#  letter under which the plot is dropped -- "B" = cols 2-3, "F" = cols 6-7.)
COMPARISONS = [
    {"left": 1, "right": 2, "names": ["Ampl A&C",        "Ampl C"],        "anchor": "B", "y_label": "Ampl"},
    {"left": 5, "right": 6, "names": ["AUC A&C",         "AUC C"],         "anchor": "F", "y_label": "AUC"},
    {"left": 9, "right": 10, "names": ["% AP A&C", "% AP C"], "anchor": "J", "y_label": "AP success % "},
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

    # Per-ROW (per-ROI) fraction of cells above SNR threshold; keep rows at/above ROI_THRESHOLD.
    keep_mask = bin_c.mean(axis=1) >= ROI_THRESHOLD

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

        result[metric] = mat[keep_mask.values]

    # Per-ROI AP success rate (fraction of epochs above SNR threshold) for ALL ROIs (no filter).
    result["AP A&C"]    = bin_ac.mean(axis=1)
    result["AP C"]      = bin_c.mean(axis=1)
    result["keep_mask"] = keep_mask          # bool Series over all ROI indices
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


def wilcoxon_barplot(group_left, group_right, names, png_path: Path, y_label: str = "", plot_title: str = ""):
    """Run a paired Wilcoxon test on two columns and save a bar plot.

    Returns the exact p-value (or None if the test could not run).
    """
    analysis = AutoStatLib.StatisticalAnalysis(
        [list(group_left), list(group_right)],
        paired=True,
        groups_name=names,
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
    plot.close()
    return p_exact


def write_xlsx(id_label, rows, out_path: Path, plot_title: str = ""):
    """Write one summary sheet as .xlsx, then run Wilcoxon tests on the
    Ampl (cols 2-3) and AUC (cols 6-7) pairs and embed the bar plots just
    below the last data row, under their respective columns."""
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
        for i, cmp in enumerate(COMPARISONS):
            # Keep only rows where both values are numeric (drops empty cells from filtered ROIs).
            pairs = [(r[cmp["left"]], r[cmp["right"]]) for r in rows
                     if isinstance(r[cmp["left"]], (int, float))
                     and isinstance(r[cmp["right"]], (int, float))]
            left, right = (list(x) for x in zip(*pairs)) if pairs else ([], [])
            png = tmp / f"plot_{i}.png"
            p = wilcoxon_barplot(left, right, cmp["names"], png, y_label=cmp["y_label"], plot_title=plot_title)
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
        if root_path.exists():
            print(f"Skipping {root_path.name} (already exists)")
        else:
            write_xlsx("instance", day_rows(instances), root_path, plot_title="Statistics on mice")
            print(f"Wrote {len(instances)} day(s) to {root_path}")

        # Local per-day summaries (rows = ROIs) inside the working folder.
        for instance in sorted(instances):
            local_path = work_folder / f"{LOCAL_PREFIX}{instance}.xlsx"
            if local_path.exists():
                print(f"    Skipping {local_path.name} (already exists)")
                continue
            rows = roi_rows(instances[instance])
            write_xlsx("ROI", rows, local_path, plot_title=f"Statistic on rois\n{instance}")
            print(f"    local: {len(rows)} ROI(s) -> {local_path}")

        processed += 1
        print()

    if processed == 0:
        print(f"No working folders with the expected structure found under {ROOT}")
    else:
        print(f"Done: processed {processed} working folder(s).")


if __name__ == "__main__":
    main()

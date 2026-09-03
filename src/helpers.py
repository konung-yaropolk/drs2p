import os
import re
import csv
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import platform
import ssd_guard
from pathlib import Path, PurePosixPath, PureWindowsPath


class Helpers:

    def normalize_path(self, path_str: str, target: str = "auto") -> str:
        """
        Normalize a mixed-separator path to a safe PurePath for the target platform.

        Args:
            path_str: Path with mixed or inconsistent separators/drive letters.
            target:   "auto" (current platform), "windows", or "posix".

        Returns:
            PureWindowsPath or PurePosixPath with correct separators.
        """
        if target == "auto":
            target = "windows" if platform.system() == "Windows" else "posix"

        # Extract and normalize drive letter (e.g. c:folder → C:/)
        match = re.match(r"^([a-zA-Z]):[/\\]?", path_str)
        drive = f"{match.group(1).upper()}:/" if match else ""
        rest  = path_str[match.end():] if match else path_str

        # Unify separators, drop empty parts, rejoin as a single clean string
        parts = [p for p in rest.replace("\\", "/").split("/") if p]
        normalized = drive + "/".join(parts)

        return str(PureWindowsPath(normalized)) if target == "windows" else str(PurePosixPath(normalized))
        
    def calculate_suffix_and_nosuffix(self, file_full_path):
        # Get the directory from the given file's full path

        file_full_path = os.path.abspath(
            os.path.normpath(os.path.splitext(file_full_path)[0])
        )

        dir_path = os.path.dirname(file_full_path)
        # Get the given file's name
        given_file = os.path.basename(file_full_path)

        # List all .txt files in the directory
        txt_files = [
            os.path.abspath(os.path.normpath(os.path.join(dir_path, f)))
            for f in os.listdir(dir_path)
            if f.endswith(".txt")
        ]

        assert (
            txt_files
        ), f"!!! Error: No .txt metadata file found in directory {dir_path}."

        # Find the longest common prefix among the given file and txt files
        common_prefixes = [
            os.path.commonprefix([file_full_path, os.path.splitext(txt_file)[0]]) for txt_file in txt_files
        ]

        file_nosuffix_with_path = max(common_prefixes, key=len)

        # Remove the directory path from the common prefix
        file_nosuffix = os.path.basename(file_nosuffix_with_path)

        # Determine the suffix from the given file
        filename_suffix = os.path.basename(
            file_full_path[len(file_nosuffix_with_path) :]
        )

        return filename_suffix, file_nosuffix

    def frame_to_sec(self, frame: int) -> float:
        """Convert frame to timestamp (start of frame)."""
        out = (frame / self.movie_config.fps_adjusted
               if frame <= self.movie_config.n_frames
               else self.movie_config.movie_duration_adjusted)
        return out

    def sec_to_frame(self, timestamp: float) -> int:
        """Convert timestamp to frame (floor rounding)."""
        # if timestamp < 0:
        #     raise ValueError("Timestamp must be non-negative")

        out = (
            int((timestamp * self.movie_config.fps_adjusted) // 1)
            if timestamp <= self.movie_config.movie_duration_adjusted
            else self.movie_config.n_frames
        )

        return out

    def flatten_array(self, nested_list):
        for item in nested_list:
            if isinstance(item, list):
                yield from self.flatten_array(item)
            else:
                yield item

    def save_tiff(self, output_path, data, metadata={}):

        # output = Image.fromarray(data)
        # output.save(output_path, save_all=True,
        #             compression="tiff_deflate",
        #             tiffinfo=metadata)

        # does not work for some reasons:
        # if data.ndim == 3 else np.array([data]).astype(np.float32)
        img = data.astype(np.float32)
        tifffile.imwrite(
            output_path, img, imagej=True, compression="zlib", metadata=metadata
        )

    def transpose(self, matrix):
        """Transpose a matrix.

        Fast path: if already an ndarray, use .T (zero-copy view).
        Slow path: list-of-lists — convert to ndarray, transpose, return as list.
        The caller receives the same data type it passes in, so existing code
        that iterates with `for row in transpose(...)` keeps working.
        """
        if isinstance(matrix, np.ndarray):
            return matrix.T

        # Convert to ndarray for the transpose, then back to list of lists.
        # This replaces the old O(rows*cols) Python nested loop.
        arr = np.array(matrix, dtype=object)
        return arr.T.tolist()

    def transpose_autoballance(self, data):
        """Transpose a ragged (non-rectangular) list, padding with None."""
        # Determine max row length and pad all rows to that length
        max_len = max(len(row) for row in data)
        # Use np.empty(object) to avoid numeric coercion on mixed types
        arr = np.empty((len(data), max_len), dtype=object)
        for i, row in enumerate(data):
            arr[i, :len(row)] = row
        return tuple(map(tuple, arr.T.tolist()))
    def csv_write(self, data, csv_path, csv_file, filename_suffix, subdir=False):

        if subdir:
            os.makedirs(csv_path + csv_file + filename_suffix + "/", exist_ok=True)
            path = "{0}{1}{2}/{2}.csv".format(
                csv_path,
                csv_file,
                filename_suffix,
            )

        else:
            path = "{0}/{2}.csv".format(
                csv_path,
                csv_file,
                filename_suffix,
            )

        with ssd_guard.guarded_open(path, "w") as f:

            writer = csv.writer(
                f,
                delimiter=",",
                lineterminator="\r",
            )
            for row in data:
                writer.writerow(row)

    def filter_list(self, data, bin_mask, replace=True, replace_with=None):
        """Filter or mask a list/array by a boolean mask.

        replace=True  → keep element where True, substitute replace_with where False.
        replace=False → return only elements where True (no None padding).

        Uses np.where / boolean indexing so no Python loop is needed.
        """
        arr  = np.asarray(data,     dtype=object)
        mask = np.asarray(bin_mask, dtype=bool)

        if replace:
            # np.where on object arrays needs explicit fill scalar
            out = np.where(mask, arr, replace_with)
            return out.tolist()
        else:
            return arr[mask].tolist()

    def plot_traces(
        self,
        x,
        cols,
        events,
        savename,
        show=False,
        save=True,
        average=True,
        offset=0,
        figsize=(15, 5),
        alpha=None,
        dpi=200,
        linewidth=0.5,
        linecolor="k",
        fillcolor="g",
        fillalpha=1,
        avg_linecolor="r",
        event_linecolor="g",
        event_linestyle=":",
    ):

        x = np.asarray(x)

        plt.figure(figsize=figsize, dpi=dpi)
        # plt.style.use("ggplot")

        # set alpha based on number of columns,
        # so that the more columns,
        # the more transparent each line
        if not alpha:
            n_cols = len(cols)
            alpha = min(3 / n_cols, 1)
        elif alpha > 1:
            alpha = 1

        # cols is typically a 2-D ndarray (n_traces × n_samples) after the
        # refactor in traces.py; fall back gracefully if it is a list.
        cols_arr = np.asarray(cols)

        for col in cols_arr:
            plt.plot(x, col, color=linecolor, linewidth=linewidth, alpha=alpha)

        # avg line plot
        if average:
            plt.plot(
                x,
                cols_arr.mean(axis=0),
                color=avg_linecolor,
                linewidth=linewidth * 3,
                alpha=1,
            )

        cols_max = cols_arr.max()
        cols_min = cols_arr.min()

        for event in events:
            if isinstance(event, (int, float)):
                plt.axvline(
                    event,
                    color=event_linecolor,
                    linestyle=event_linestyle,
                    linewidth=linewidth * 0.5,
                    alpha=1,
                    zorder=1,
                )
            elif isinstance(event, (list, tuple)):
                plt.fill_between(
                    x,
                    y1=cols_max,
                    y2=cols_min,
                    where=(x >= event[0]) & (x <= event[-1]),
                    color=fillcolor,
                    edgecolor="none",
                    alpha=fillalpha,
                    zorder=0,
                )
            else:
                pass

        # plt.suptitle(TITLE)
        # plt.xlabel('Time, s')
        # plt.ylabel("Amplitude + Offset")

        plt.tight_layout()

        # Save the combined figure

        if save:
            if isinstance(savename, str):
                plt.savefig(savename, transparent=False)
            elif isinstance(savename, (list, tuple)):
                for name in savename:
                    plt.savefig(name, transparent=False)
            else:
                self.logging("!!!    Fail: invalid savename type        ", type(savename))

        if show:
            plt.show()

        plt.close()

import os
import csv
import numpy as np
import tifffile
import matplotlib.pyplot as plt



class Helpers:
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
            os.path.commonprefix([file_full_path, txt_file]) for txt_file in txt_files
        ]

        file_nosuffix_with_path = max(common_prefixes, key=len).rstrip("_")

        # Remove the directory path from the common prefix
        file_nosuffix = os.path.basename(file_nosuffix_with_path)

        # Determine the suffix from the given file
        filename_suffix = os.path.basename(
            file_full_path[len(file_nosuffix_with_path) :]
        )

        return filename_suffix, file_nosuffix

    def frame_to_sec(self, frame: int) -> float:
        """Convert frame to timestamp (start of frame)."""
        out = frame / self.fps if frame <= self.n_frames else self.s_movie_duration
        return out

    def sec_to_frame(self, timestamp: float) -> int:
        """Convert timestamp to frame (floor rounding)."""
        # if timestamp < 0:
        #     raise ValueError("Timestamp must be non-negative")

        out = (
            int((timestamp * self.fps) // 1)
            if timestamp <= self.s_movie_duration
            else self.n_frames
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
        rows = len(matrix)
        cols = max(len(row) for row in matrix)

        # use internal numpy method if ndarray for optimization
        if isinstance(matrix, np.ndarray):
            transposed = matrix.T
        else:
            transposed = [[None] * rows for _ in range(cols)]

            for i in range(rows):
                for j in range(len(matrix[i])):
                    transposed[j][i] = matrix[i][j]

        return transposed

    def transpose_autoballance(self, data):
        # Determine the maximum length of any row
        max_len = max(len(row) for row in data)
        # Fill shorter rows with None to make all rows equal in length
        balanced_data = tuple(list(row) + [None] * (max_len - len(row)) for row in data)
        # Transpose the matrix
        data_t = tuple(
            tuple(balanced_data[j][i] for j in range(len(balanced_data)))
            for i in range(max_len)
        )
        return data_t

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

        with open(path, "w") as f:

            writer = csv.writer(
                f,
                delimiter=",",
                lineterminator="\r",
            )
            for row in data:
                writer.writerow(row)

    def filter_list(self, list, bin, replace=True, replace_with=None):
        if replace == True:
            output = [value if bin[i] else replace_with for i, value in enumerate(list)]
        else:
            output = [value for value, keep in zip(list, bin) if keep]

        return output

    def plot_traces(
        self,
        x,
        cols,
        events,
        savename,
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

        x = np.array(x)

        plt.figure(figsize=figsize, dpi=dpi)
        # plt.style.use("ggplot")

        # set alpha based on number of columns,
        # so that the more columns,
        # the more transparent each line
        if not alpha:
            n_cols = len(cols)
            alpha = 3 / n_cols

        if alpha > 1:
            alpha = 1

        # Plot each trace
        for i, col in enumerate(cols):
            plt.plot(x, col, color=linecolor, linewidth=linewidth, alpha=alpha)

        # avg line plot
        if average:
            plt.plot(
                x,
                np.mean(cols, axis=0),
                color=avg_linecolor,
                linewidth=linewidth * 3,
                alpha=1,
            )

        for event in events:
            if isinstance(event, int) or isinstance(event, float):
                plt.axvline(
                    event,
                    color=event_linecolor,
                    linestyle=event_linestyle,
                    linewidth=linewidth * 0.5,
                    alpha=1,
                    zorder=1,
                )
            elif isinstance(event, list) or isinstance(event, tuple):
                plt.fill_between(
                    x,
                    y1=np.max(cols),
                    y2=np.min(cols),
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

        if isinstance(savename, str):
            plt.savefig(savename, transparent=False)
        elif isinstance(savename, list) or isinstance(savename, tuple):
            for name in savename:
                plt.savefig(name, transparent=False)
        else:
            self.logging("!!!    Fail: invalid savename type        ", type(savename))

        plt.close()

import re
import os
import csv
import AutoStatLib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages

from debug import Debug
from helpers import Helpers

CALCULATIONS_SUBFOLDER_NAME = "_CALCULATIONS_auto_"
BINARIZATION_RESP_THRESHOLD = 0.29
PLOT_STATS_FOR_EACH_ROI = False    # significantly slows down the traces calculation when massive amount of rois
LETTER_FIGSIZE = (8.5, 11)    # printable letter format, portrait; use (11, 8.5) for landscape
RATIO_BAR_UNIT = 0.3    # constant scale of the ratio bars: ratio of 1 takes 80% of the vertical shift
DEBUG = True


class TracesCalc(Helpers, Debug):
    def __init__(self, 
                 run_config,
                 movie_config,
                 trigger_config):
        self.run_config = run_config
        self.movie_config = movie_config
        self.trigger_config = trigger_config
        self.log = ' \n'

        self.file_path=self.movie_config.file_path
        self.path = self.movie_config.path
        self.file = self.movie_config.filename
        self.output_suffix = self.trigger_config.label
        self.filename_suffix = self.movie_config.filename_suffix
        self.file_nosuffix = self.movie_config.file_nosuffix    

        self.s_trig_time = self.movie_config.events[self.trigger_config.trig_number - 1][1]
        self.s_epoch_duration = self.trigger_config.step_duration * len(self.trigger_config.drs_pattern[0])
        self.vertical_shift = trigger_config.vertical_shift
        self.n_steps_per_epoch = len(self.trigger_config.drs_pattern[0])

        #work in progress
        self.v_shifts =  {}
        self.filters =  {}
        self.v_shifts_return = {}
        self.filters_return = {}
        self.ampls_return = {}
        self.aucs_return = {}

    def logging(self, *args, **kwargs):
        message = ' '.join(map(str, args))
        self.log += '\n' + message

    def file_finder(self, pattern, nonrecursive=False):
        files_list = []  # To store the paths of .txt files

        # Walk through the directory and its subdirectories
        for root, _, files in os.walk(self.path):
            for filename in files:
                if re.search(pattern, filename):
                    files_list.append(
                        [root if root[-1] == "/" else root + "/", filename[:-4]]
                    )

            if nonrecursive:
                break

        return files_list

    def file_lister(self, pattern, nonrecursive=False):
        files = []

        if os.path.isdir(self.path):
            files.extend(self.file_finder(pattern, nonrecursive))
        else:
            self.logging("!!!    Fail: invalid path        ", self.path)

        return files

    def find_time_index(self, content, time):
        # content is a list-of-rows; column 0 is the time axis.
        # Extract it as a float array and use np.argmin — one vectorized call
        # instead of a generator + Python list comprehension + linear scan.
        time_col = np.asarray([float(row[0]) for row in content])
        return int(np.argmin(np.abs(time_col - time)))

    def data_normalize(self, content, start, zero):
        # content: list of columns (each column is a sequence of numeric-string or float values).
        # Convert once to a 2-D float array: shape (n_rois, n_frames).
        arr = np.array(content, dtype=float)   # single allocation + type conversion

        # Baseline slice for every ROI simultaneously
        baseline = arr[:, start:zero]          # shape (n_rois, baseline_len)
        means = baseline.mean(axis=1, keepdims=True)  # shape (n_rois, 1)

        # Avoid division by zero — same semantics as the original (result = 0 when mean=0)
        safe_means = np.where(means == 0, 1.0, means)
        normalized = np.where(means == 0, 0.0, (arr - means) / safe_means)  # ΔF/F₀

        # Return as list-of-lists to keep downstream code unchanged
        return normalized.tolist()

    def csv_cutter(self, content):
        # content: list of rows (each row = [time, roi1, roi2, ...]) as strings.
        # Convert the whole block to a float array once: shape (n_frames, n_cols).
        arr = np.array(content, dtype=float)
        time_col = arr[:, 0]                   # shape (n_frames,)

        # All index lookups now operate on the pre-converted float array.
        def _nearest(t):
            return int(np.argmin(np.abs(time_col - t)))

        start = (
            _nearest(self.s_trig_time - self.trigger_config.time_before_trig)
            if self.trigger_config.time_before_trig
            else None
        )
        start_bl = (
            _nearest(self.s_trig_time - self.trigger_config.baseline_duraton)
            if self.trigger_config.baseline_duraton
            else start
        )
        zero  = _nearest(self.s_trig_time)
        end   = (
            _nearest(self.s_trig_time + self.trigger_config.time_after_trig)
            if self.trigger_config.time_after_trig
            else None
        )

        # Build the output block: first column is time re-zeroed to trigger.
        # Remaining columns are ROI traces, optionally normalized.
        timeline_zero = time_col - self.s_trig_time   # shape (n_frames,)
        roi_data = arr[:, 1:].T                        # shape (n_rois, n_frames)

        if self.trigger_config.relative_values:
            roi_data = np.array(
                self.data_normalize(roi_data.tolist(), start_bl, zero),
                dtype=float,
            )

        # Reassemble as (n_frames, n_cols) then slice the time window
        out = np.column_stack([timeline_zero, roi_data.T])  # (n_frames, 1+n_rois)
        return out[start:end]

    def csv_transform(self, content_raw):
        mean_col = self.trigger_config.mean_col_order   # "Mean" column index
        n_cols   = self.trigger_config.cols_per_roi     # measurements per ROI

        # Parse the full CSV block into a float array once.
        # content_raw is a tuple-of-tuples of strings; skip the header row (row 0).
        # Shape after skip: (n_frames, total_cols).
        raw_arr = np.array(content_raw[1:], dtype=float)

        # Extract only the "Mean" columns for each ROI: columns mean_col, mean_col+n_cols, ...
        roi_means = raw_arr[:, mean_col::n_cols]  # shape (n_frames, n_rois)

        # Build the time axis as a float array.
        # Each frame's timestamp is the END of its acquisition window, not the
        # start — frame i (0-indexed) finishes acquiring at time (i+1)*spf.
        # So frame 0 -> spf, frame 1 -> 2*spf, etc. (matches original semantics).
        n_frames   = roi_means.shape[0]
        time_col   = np.arange(1, n_frames + 1) * self.movie_config.seconds_per_frame_adjusted

        # Return list-of-rows: each row = [time, roi1_mean, roi2_mean, ...]
        # so that csv_cutter receives the same structure it always did.
        out = np.column_stack([time_col, roi_means])  # (n_frames, 1+n_rois)
        return out.tolist()

    def csv_read(self, csv_path, csv_file):

        with open(csv_path + csv_file + ".csv", "r") as file:
            reader = csv.reader(file, delimiter=",")
            content_raw = tuple(reader)

        return content_raw

    def calculate_ampl_auc_bin(self, start_bl, end_bl, start, end):

        matrix = np.array(self.transpose(self.csv_matrix))

        # Extract time vector and data traces
        x = matrix[0]
        traces = matrix[1:]
        
        # Indices for baseline and signal periods
        bl_indices = np.where((x >= start_bl) & (x <= end_bl))[0]
        sig_indices = np.where((x >= start) & (x <= end))[0]
        whole_step_indices = np.where((x >= start_bl) & (x <= end))[0]

        # Lists to store peak amplitudes and AUCs for each trace
        ampl_list = []
        auc_list = []
        bin_list = []
        snr_list = [] # saignal-to-noise ratio list, in ampl/std of baseline
        raw_line_list = [x[whole_step_indices] - start]

        for i, trace in enumerate(traces):

            # Calculate baseline
            baseline = np.mean(trace[bl_indices])
            # Baseline correction
            baselined_trace = trace - baseline
            # Peak amplitude in signal period
            ampl = np.max(baselined_trace[sig_indices])
            ampl_list.append(ampl)
            # AUC in signal period
            auc = np.trapezoid(baselined_trace[sig_indices], x[sig_indices])
            auc_list.append(auc)
            # Signal-to-noise ratio
            snr_list.append(ampl /  np.std(trace[bl_indices]))
            # Binarization
            bin_list.append(ampl > self.trigger_config.sigmas_treshold * np.std(trace[bl_indices]))

            raw_line_list.append(baselined_trace[whole_step_indices])

            # # Debug responce binarization 
            # # needed only during dev
            # print(f"Signal/Noise {ampl /  np.std(trace[bl_indices]):.2f} sigmas; Current responce considered: {ampl > self.trigger_config.sigmas_treshold * 
            #                             np.std(trace[bl_indices])}")
            # self.plot_traces(
            #     whole_step_indices,
            #     [baselined_trace[whole_step_indices]],
            #     [[bl_indices[0],bl_indices[-1]], [sig_indices[0],sig_indices[-1]]],
            #     ".png",
            #     save=False,
            #     show=True,
            #     average=False,
            #     linewidth=1.5,
            #     linecolor="darkcyan",
            #     fillcolor="violet",
            #     event_linecolor="orchid",
            #     event_linestyle="-",
            #     avg_linecolor="darkcyan",
            #     alpha=0.5,
            #     fillalpha=0.5,
            #     dpi=100,
            #     figsize=(5, 5),
            # )

        # Calculate mean amplitude and AUC across all traces
        ampl_mean_of_rois = np.mean(ampl_list)
        auc_mean_of_rois = np.mean(auc_list)

        return (
            ampl_mean_of_rois,
            ampl_list,
            auc_mean_of_rois,
            auc_list,
            bin_list,
            snr_list,
            raw_line_list,
        )

    def calc_traces_sequence(self, i):

        delay = self.trigger_config.step_duration * i
        (
            ampl_mean_of_rois_by_epoch,
            ampl_list_each_by_roi,
            auc_mean_of_rois_by_epoch,
            auc_list_each_by_roi,
            bin_list_each_by_roi,
            snr_list_each_by_roi,
            raw_line_list,
        ) = [
            [
                self.calculate_ampl_auc_bin(
                    (i * self.s_epoch_duration) + delay - self.trigger_config.step_duration / 2,
                    (i * self.s_epoch_duration) + delay,
                    (i * self.s_epoch_duration) + delay,
                    (i * self.s_epoch_duration) + delay + self.trigger_config.step_duration / 2,
                )[j]
                for i in range(self.trigger_config.start_from_epoch, self.trigger_config.start_from_epoch + self.trigger_config.n_epochs)
            ]
            for j in range(7)
        ]

        ampl_list_each_by_epoch = self.transpose(ampl_list_each_by_roi)
        auc_list_each_by_epoch = self.transpose(auc_list_each_by_roi)
        bin_list_each_by_epoch = self.transpose(bin_list_each_by_roi)
        snr_list_each_by_epoch = self.transpose(snr_list_each_by_roi)
        ampl_mean_of_epochs_by_rois = [
            np.mean(epoch) for epoch in ampl_list_each_by_epoch
        ]
        auc_mean_of_epochs_by_rois = [
            np.mean(epoch) for epoch in auc_list_each_by_epoch
        ]

        return (
            ampl_mean_of_rois_by_epoch,
            ampl_mean_of_epochs_by_rois,
            ampl_list_each_by_roi,
            ampl_list_each_by_epoch,
            auc_mean_of_rois_by_epoch,
            auc_mean_of_epochs_by_rois,
            auc_list_each_by_roi,
            auc_list_each_by_epoch,
            bin_list_each_by_epoch,
            snr_list_each_by_epoch,
            raw_line_list,
        )

    def detailed_stats(self, csv_path, csv_file, output_dir):
        st1_ampl_mean_of_epochs_by_rois = []
        st2_ampl_mean_of_epochs_by_rois = []
        st1_auc_mean_of_epochs_by_rois = []
        st2_auc_mean_of_epochs_by_rois = []
        st1_bin_summary_by_rois = []
        st2_bin_summary_by_rois = []
        ampl_st2_to_st1_ratio_mean_of_epochs_by_rois = np.array([0.001])
        ampl_st2_to_st1_ratio_rois_by_epoch = np.array([[0.001]])
        auc_st2_to_st1_ratio_mean_of_epochs_by_rois = np.array([0.001])
        auc_st2_to_st1_ratio_rois_by_epoch = np.array([[0.001]])
        s1s2_ampl_list_each_by_epoch = np.array([[0.001]])
        s1s2_auc_list_each_by_epoch = np.array([[0.001]])
        s1s2_bin_list_each_by_epoch = []
        s1_bin_list_each_by_epoch = []
        s2_bin_list_each_by_epoch = []
        s2_ampl_list_each_by_roi = []
        s2_ampl_list_each_by_epoch = []
        s2_auc_list_each_by_epoch = []
        self.s1s2_delay = 0
        self.s1_delay = 0
        self.s2_delay = 0
        # check if we have the expected pattern
        # detailed_stats only works when s1s2 or s1 alone is present
        # skip if pattern is s1 and s2 alternating (no s1s2)
        has_s1s2 = any(a == 1 and c == 1 for a, c in zip(
            self.trigger_config.drs_pattern[0], 
            self.trigger_config.drs_pattern[1]))
        has_s1_only = any(a == 1 and c == 0 for a, c in zip(
            self.trigger_config.drs_pattern[0],
            self.trigger_config.drs_pattern[1]))
        has_s2_only = any(a == 0 and c == 1 for a, c in zip(
            self.trigger_config.drs_pattern[0],
            self.trigger_config.drs_pattern[1]))

        if not has_s1s2 and has_s1_only and has_s2_only:
            self.logging("Skipping detailed_stats: alternating s1/s2 pattern not supported yet")
            return
    
        

        # create unique id for each calculation unit (trigger)
        # unit_id = self.file_path + '%' + str(self.trigger_config.trig_number-1)
        unit_id = (
            csv_path + csv_file + "$trig:" + str(self.trigger_config.trig_number) + "$" + self.output_suffix
        )

        # get the name of directory conraining original tif and csv file to add to the output files name
        base_dir = os.path.basename(os.path.normpath(csv_path))

        s1s2 = False
        s1 = False
        s2 = False
        self.group_names = []
        for i, (sp1, sp2) in enumerate(zip(self.trigger_config.drs_pattern[0], self.trigger_config.drs_pattern[1])):
            match (sp1, sp2):
                case (1, 1):
                    (
                        s1s2_ampl_mean_of_rois_by_epoch,
                        s1s2_ampl_mean_of_epochs_by_rois,
                        s1s2_ampl_list_each_by_roi,
                        s1s2_ampl_list_each_by_epoch,
                        s1s2_auc_mean_of_rois_by_epoch,
                        s1s2_auc_mean_of_epochs_by_rois,
                        s1s2_auc_list_each_by_roi,
                        s1s2_auc_list_each_by_epoch,
                        s1s2_bin_list_each_by_epoch,
                        s1s2_snr_list_each_by_epoch,
                        s1s2_raw_line_list,
                    ) = self.calc_traces_sequence(i)
                    self.s1s2_delay = i * self.trigger_config.step_duration
                    s1s2 = True
                    s1s2_order = i
                    self.group_names.append(self.trigger_config.stim_1_name + "&" + self.trigger_config.stim_2_name)
                case (1, 0):
                    (
                        s1_ampl_mean_of_rois_by_epoch,
                        s1_ampl_mean_of_epochs_by_rois,
                        s1_ampl_list_each_by_roi,
                        s1_ampl_list_each_by_epoch,
                        s1_auc_mean_of_rois_by_epoch,
                        s1_auc_mean_of_epochs_by_rois,
                        s1_auc_list_each_by_roi,
                        s1_auc_list_each_by_epoch,
                        s1_bin_list_each_by_epoch,
                        s1_snr_list_each_by_epoch,
                        s1_raw_line_list,
                    ) = self.calc_traces_sequence(i)
                    self.s1_delay = i * self.trigger_config.step_duration
                    s1 = True
                    s1_order = i
                    self.group_names.append(self.trigger_config.stim_1_name)
                case (0, 1):
                    (
                        s2_ampl_mean_of_rois_by_epoch,
                        s2_ampl_mean_of_epochs_by_rois,
                        s2_ampl_list_each_by_roi,
                        s2_ampl_list_each_by_epoch,
                        s2_auc_mean_of_rois_by_epoch,
                        s2_auc_mean_of_epochs_by_rois,
                        s2_auc_list_each_by_roi,
                        s2_auc_list_each_by_epoch,
                        s2_bin_list_each_by_epoch,
                        s2_snr_list_each_by_epoch,
                        s2_raw_line_list,
                    ) = self.calc_traces_sequence(i)
                    self.s2_delay = i * self.trigger_config.step_duration
                    s2 = True
                    s2_order = i
                    self.group_names.append(self.trigger_config.stim_2_name)
                case (0, 0):
                    pass
                case (None, None):
                    pass
                # responses_each_by_roi, responses_each_by_epoch = self.calc_traces_sequence(i)

        # Check is there both stim or only one to avoid errs
        # Огидна конструкція, потім переробити
        st1_ampl_mean_of_epochs_by_rois = []
        st2_ampl_mean_of_epochs_by_rois = []
        st1_auc_mean_of_epochs_by_rois = []
        st2_auc_mean_of_epochs_by_rois = []
        ampl_st2_to_st1_ratio_mean_of_epochs_by_rois = np.array([0.001])
        ampl_st2_to_st1_ratio_rois_by_epoch = np.array([[0.001]])
        auc_st2_to_st1_ratio_mean_of_epochs_by_rois = np.array([0.001])
        auc_st2_to_st1_ratio_rois_by_epoch = np.array([[0.001]])

        if s1s2:
            st1_ampl_mean_of_epochs_by_rois = s1s2_ampl_mean_of_epochs_by_rois
            st2_ampl_mean_of_epochs_by_rois = s2_ampl_mean_of_epochs_by_rois
            st1_auc_mean_of_epochs_by_rois = s1s2_auc_mean_of_epochs_by_rois
            st2_auc_mean_of_epochs_by_rois = s2_auc_mean_of_epochs_by_rois

        if not s1s2 and not s1:
            ampl_s2_to_s1s2_ratio_mean_of_epochs_by_rois = np.array(
                [0.001] * len(s2_ampl_mean_of_epochs_by_rois)
            )
            ampl_s2_to_s1s2_ratio_rois_by_epoch = np.array(
                [[0.001] for _ in range(len(s2_ampl_list_each_by_epoch))]
            )
            s1s2_ampl_mean_of_epochs_by_rois = np.array(
                [0.001] * len(s2_auc_mean_of_epochs_by_rois)
            )
            s1s2_ampl_list_each_by_epoch = np.array([[0.001] * self.trigger_config.n_epochs])

            auc_s2_to_s1s2_ratio_mean_of_epochs_by_rois = np.array(
                [0.001] * len(s2_auc_mean_of_epochs_by_rois)
            )
            auc_s2_to_s1s2_ratio_rois_by_epoch = np.array(
                [[0.001] for _ in range(len(s2_auc_list_each_by_epoch))]
            )
            s1s2_auc_mean_of_epochs_by_rois = np.array(
                [0.001] * len(s2_auc_mean_of_epochs_by_rois)
            )
            s1s2_auc_list_each_by_epoch = np.array([[0.001] * self.trigger_config.n_epochs])

        if s1s2 and not s1:
            st1_ampl_mean_of_epochs_by_rois = s1s2_ampl_mean_of_epochs_by_rois
            st2_ampl_mean_of_epochs_by_rois = s2_ampl_mean_of_epochs_by_rois

            ampl_st2_to_st1_ratio_mean_of_epochs_by_rois = np.array(
                s2_ampl_mean_of_epochs_by_rois
            ) / np.array(s1s2_ampl_mean_of_epochs_by_rois)
            ampl_st2_to_st1_ratio_rois_by_epoch = np.array(
                s2_ampl_list_each_by_epoch
            ) / np.array(s1s2_ampl_list_each_by_epoch)

            st1_auc_mean_of_epochs_by_rois = s1s2_auc_mean_of_epochs_by_rois
            st2_auc_mean_of_epochs_by_rois = s2_auc_mean_of_epochs_by_rois

            auc_st2_to_st1_ratio_mean_of_epochs_by_rois = np.array(
                s2_auc_mean_of_epochs_by_rois
            ) / np.array(s1s2_auc_mean_of_epochs_by_rois)
            auc_st2_to_st1_ratio_rois_by_epoch = np.array(
                s2_auc_list_each_by_epoch
            ) / np.array(s1s2_auc_list_each_by_epoch)

        if s1 and not s1s2:
            st1_ampl_mean_of_epochs_by_rois = s1_ampl_mean_of_epochs_by_rois
            st2_ampl_mean_of_epochs_by_rois = s2_ampl_mean_of_epochs_by_rois

            ampl_st2_to_st1_ratio_mean_of_epochs_by_rois = np.array(
                s2_ampl_mean_of_epochs_by_rois
            ) / np.array(s1_ampl_mean_of_epochs_by_rois)
            ampl_st2_to_st1_ratio_rois_by_epoch = np.array(
                s2_ampl_list_each_by_epoch
            ) / np.array(s1_ampl_list_each_by_epoch)

            st1_auc_mean_of_epochs_by_rois = s1_auc_mean_of_epochs_by_rois
            st2_auc_mean_of_epochs_by_rois = s2_auc_mean_of_epochs_by_rois

            auc_st2_to_st1_ratio_mean_of_epochs_by_rois = np.array(
                s2_auc_mean_of_epochs_by_rois
            ) / np.array(s1_auc_mean_of_epochs_by_rois)
            auc_st2_to_st1_ratio_rois_by_epoch = np.array(
                s2_auc_list_each_by_epoch
            ) / np.array(s1_auc_list_each_by_epoch)

        # Як рахуємо коли в нас є і перший і другий стимули окремо і обидва разом
        if s1 and s1s2:
            # тут можна помінять якщо треба інший варіант порівнянь
            st1_ampl_mean_of_epochs_by_rois = s1s2_ampl_mean_of_epochs_by_rois
            st2_ampl_mean_of_epochs_by_rois = s2_ampl_mean_of_epochs_by_rois

            ampl_st2_to_st1_ratio_mean_of_epochs_by_rois = np.array(
                s2_ampl_mean_of_epochs_by_rois
            ) / np.array(s1s2_ampl_mean_of_epochs_by_rois)
            ampl_st2_to_st1_ratio_rois_by_epoch = np.array(
                s2_ampl_list_each_by_epoch
            ) / np.array(s1s2_ampl_list_each_by_epoch)

            st1_auc_mean_of_epochs_by_rois = s1s2_auc_mean_of_epochs_by_rois
            st2_auc_mean_of_epochs_by_rois = s2_auc_mean_of_epochs_by_rois

            auc_st2_to_st1_ratio_mean_of_epochs_by_rois = np.array(
                s2_auc_mean_of_epochs_by_rois
            ) / np.array(s1s2_auc_mean_of_epochs_by_rois)
            auc_st2_to_st1_ratio_rois_by_epoch = np.array(
                s2_auc_list_each_by_epoch
            ) / np.array(s1s2_auc_list_each_by_epoch)
        

        # Binarization:

        if (not s1s2) and (not s1):
            s1s2_bin_list_each_by_epoch = s2_bin_list_each_by_epoch
            s1_bin_list_each_by_epoch = s2_bin_list_each_by_epoch
        if (not s1s2) and s1:
            s1s2_bin_list_each_by_epoch = s1_bin_list_each_by_epoch
        if (not s1) and s1s2:
            s1_bin_list_each_by_epoch = s1s2_bin_list_each_by_epoch
        if s1 and s1s2:
            s1_bin_list_each_by_epoch = s1s2_bin_list_each_by_epoch

        if len(self.group_names) == 1:
            self.group_names.insert(0, "_")



        if not s1 and s1s2:
            st1_bin_summary_by_rois = (
                np.array(s1s2_bin_list_each_by_epoch, dtype=float).mean(axis=1)
                > BINARIZATION_RESP_THRESHOLD
            ).tolist()
        if not s1s2 and s1:
            st1_bin_summary_by_rois = (
                np.array(s1_bin_list_each_by_epoch, dtype=float).mean(axis=1)
                > BINARIZATION_RESP_THRESHOLD
            ).tolist()
        # fix for case with 'long' stim patterns when s1, s1s2, and s2 are presented all
        # ugly construction, but it works, and I don't have time to refactor it now
        if s1 and s1s2:
            st1_bin_summary_by_rois = (
                np.array(s1s2_bin_list_each_by_epoch, dtype=float).mean(axis=1)
                > BINARIZATION_RESP_THRESHOLD
            ).tolist()
        st2_bin_summary_by_rois = (
            np.array(s2_bin_list_each_by_epoch, dtype=float).mean(axis=1)
            > BINARIZATION_RESP_THRESHOLD
        ).tolist()


        s1s2_OR_s2_bin_list_each_by_epoch = np.logical_or(s1s2_bin_list_each_by_epoch, s2_bin_list_each_by_epoch).tolist()
        s1_OR_s2_bin_list_each_by_epoch = np.logical_or(s1_bin_list_each_by_epoch, s2_bin_list_each_by_epoch).tolist()

        st1_OR_st2_bin_summary_by_rois = np.logical_or(st1_bin_summary_by_rois, st2_bin_summary_by_rois).tolist()


        # save binarization for the next calculations
        load_unitid = (
            csv_path
            + csv_file
            + "$"
            + str(self.trigger_config.SD_filter_of_trig - 1)
            + "$"
            + "MASK FOR ANY ENDING HERE"
        )
        current_filter = [
            st1_bin_summary_by_rois,
            st1_bin_summary_by_rois,
            st2_bin_summary_by_rois,
            st1_OR_st2_bin_summary_by_rois
        ]

        if self.trigger_config.SD_filter_of_trig and load_unitid in self.filters:
            filter = self.filters[load_unitid]
        else:
            filter = current_filter

        self.filters_return |= {unit_id: current_filter}

        # save responses Ampl and AUC for the next calculations
        amps = [
            st1_ampl_mean_of_epochs_by_rois,
            st1_ampl_mean_of_epochs_by_rois,
            st2_ampl_mean_of_epochs_by_rois,
        ]

        aucs = [
            st1_auc_mean_of_epochs_by_rois,
            st1_auc_mean_of_epochs_by_rois,
            st2_auc_mean_of_epochs_by_rois,
        ]

        self.ampls_return |= {unit_id: amps}
        self.aucs_return |= {unit_id: aucs}

        self.plot_s2_to_s1s2_ratio_rois_by_epoch(
            1 / ampl_st2_to_st1_ratio_rois_by_epoch,
            "{0}{1}/_rois_by_epoch_{3}_to_{2}_{4}_ratio_auto_.png".format(
                csv_path,
                output_dir,
                self.group_names[0],
                self.group_names[1],
                self.output_suffix,
            ),
        )


        # outputs:
        os.makedirs(f"{csv_path}{output_dir}/outputs_{self.output_suffix}_{output_dir}/", exist_ok=True)

        def write_metrics(stim_label, snr_data, ampl_data, auc_data, bin_data):
            for metric, data in [("SNR", snr_data), ("Ampl", ampl_data), ("AUC", auc_data), ("Bin", bin_data)]:
                self.csv_write(
                    data,
                    csv_path + output_dir,
                    output_dir,
                    f"outputs_{self.output_suffix}_{output_dir}/_{metric}_{stim_label}_rows-roi_cols-epoch_{self.output_suffix}_auto_",
                )

        s1_name = self.trigger_config.stim_1_name
        s2_name = self.trigger_config.stim_2_name


        # Write metrics for data binarized by only one (A+C or C) responses in pair
        if s1s2:
            write_metrics(f"{s1_name}&{s2_name}", s1s2_snr_list_each_by_epoch, s1s2_ampl_list_each_by_epoch, s1s2_auc_list_each_by_epoch, s1s2_bin_list_each_by_epoch)
        if s1:
            write_metrics(s1_name, s1_snr_list_each_by_epoch, s1_ampl_list_each_by_epoch, s1_auc_list_each_by_epoch, s1_bin_list_each_by_epoch)
        if s2:
            write_metrics(s2_name, s2_snr_list_each_by_epoch, s2_ampl_list_each_by_epoch, s2_auc_list_each_by_epoch, s2_bin_list_each_by_epoch)



        # csv file of #1#2 and #2 amplitudes by rois epochs average
        header = [self.group_names[0], self.group_names[1], "ratio col1/col2"]

        # CSV summary Amplitude

        self.csv_write(
            [
                [
                    "Unfiltered",
                    "",
                    "",
                    "",
                    "",
                    "Filtered by {} SD of {}".format(
                        self.trigger_config.sigmas_treshold, self.group_names[1]
                    ),
                    "",
                    "",
                    "",
                    "",
                    "Filtered by {} SD of {}".format(
                        self.trigger_config.sigmas_treshold, self.group_names[0]
                    ),
                ],
                header + [""] * 2 + header + [""] * 2 + header,
                *self.transpose(
                    [
                        st1_ampl_mean_of_epochs_by_rois,
                        st2_ampl_mean_of_epochs_by_rois,
                        1 / ampl_st2_to_st1_ratio_mean_of_epochs_by_rois,
                        "",
                        "",
                        self.filter_list(st1_ampl_mean_of_epochs_by_rois, filter[3]),
                        self.filter_list(st2_ampl_mean_of_epochs_by_rois, filter[3]),
                        self.filter_list(
                            1 / ampl_st2_to_st1_ratio_mean_of_epochs_by_rois, filter[3]
                        ),
                        "",
                        "",
                        self.filter_list(st1_ampl_mean_of_epochs_by_rois, filter[2]),
                        self.filter_list(st2_ampl_mean_of_epochs_by_rois, filter[2]),
                        self.filter_list(
                            1 / ampl_st2_to_st1_ratio_mean_of_epochs_by_rois, filter[2]
                        ),
                    ]
                ),
            ],
            csv_path + output_dir,
            output_dir,
            "_by_rois_mean_of_epochs_{0}_and_{1}_ampl_{2}_auto_".format(
                self.group_names[0], self.group_names[1], self.output_suffix
            ),
        )

        # CSV summary AUC
        self.csv_write(
            [
                [
                    "Unfiltered",
                    "",
                    "",
                    "",
                    "",
                    "Filtered by {} SD of {}".format(
                        self.trigger_config.sigmas_treshold, self.group_names[1]
                    ),
                    "",
                    "",
                    "",
                    "",
                    "Filtered by {} SD of {}".format(
                        self.trigger_config.sigmas_treshold, self.group_names[0]
                    ),
                ],
                header + [""] * 2 + header + [""] * 2 + header,
                *self.transpose(
                    [
                        st1_auc_mean_of_epochs_by_rois,
                        st2_auc_mean_of_epochs_by_rois,
                        1 / auc_st2_to_st1_ratio_mean_of_epochs_by_rois,
                        "",
                        "",
                        self.filter_list(st1_auc_mean_of_epochs_by_rois, filter[3]),
                        self.filter_list(st2_auc_mean_of_epochs_by_rois, filter[3]),
                        self.filter_list(
                            1 / auc_st2_to_st1_ratio_mean_of_epochs_by_rois, filter[3]
                        ),
                        "",
                        "",
                        self.filter_list(st1_auc_mean_of_epochs_by_rois, filter[2]),
                        self.filter_list(st2_auc_mean_of_epochs_by_rois, filter[2]),
                        self.filter_list(
                            1 / auc_st2_to_st1_ratio_mean_of_epochs_by_rois, filter[2]
                        ),
                    ]
                ),
            ],
            csv_path + output_dir,
            output_dir,
            "_by_rois_mean_of_epochs_{0}_and_{1}_auc_{2}_auto_".format(
                self.group_names[0], self.group_names[1], self.output_suffix
            ),
        )

        # plot_s1s2_s2_roi_stats AUC for all rois
        self.plot_s1s2_s2_roi_stats(
            self.filter_list(st1_auc_mean_of_epochs_by_rois, filter[3], replace=False),
            self.filter_list(st2_auc_mean_of_epochs_by_rois, filter[3], replace=False),
            "{0}{1}/_by_rois_{2}_{3}{4}_{4}_auc_auto_.png".format(
                csv_path,
                output_dir,
                self.trigger_config.stim_1_name,
                self.trigger_config.stim_2_name,
                self.output_suffix,
            ),
            paired=True,
            y_label="AUC",
            Groups_Name=[
                        "{}+{}".format(self.trigger_config.stim_1_name, self.trigger_config.stim_2_name),
                        self.trigger_config.stim_2_name,
                    ],
        )

        # plot_s1s2_s2_roi_stats Ampl for all rois
        self.plot_s1s2_s2_roi_stats(
            self.filter_list(st1_ampl_mean_of_epochs_by_rois, filter[3], replace=False),
            self.filter_list(st2_ampl_mean_of_epochs_by_rois, filter[3], replace=False),
            "{0}{1}/_by_rois_{2}_{3}{4}_{4}_ampl_auto_.png".format(
                csv_path,
                output_dir,
                self.trigger_config.stim_1_name,
                self.trigger_config.stim_2_name,
                self.output_suffix,
            ),
            paired=True,
            y_label="ΔF/F₀",
            Groups_Name=[
                        "{}+{}".format(self.trigger_config.stim_1_name, self.trigger_config.stim_2_name),
                        self.trigger_config.stim_2_name,
                    ],
        )

        # plot_s1s2_s2_roi_stats for each roi during timeline
        if s1s2 and s2 and PLOT_STATS_FOR_EACH_ROI:
            for i in range(len(s1s2_ampl_list_each_by_epoch)):
                self.plot_s1s2_s2_roi_stats(
                    self.filter_list(s1s2_ampl_list_each_by_epoch[i], filter[3], replace=False),
                    self.filter_list(s2_ampl_list_each_by_epoch[i], filter[3], replace=False),
                    "{0}{1}/_roi{2}_{3}{4}_{4}_{5}_ampl_auto_.png".format(
                        csv_path,
                        output_dir,
                        i + 1,
                        self.trigger_config.stim_1_name,
                        self.trigger_config.stim_2_name,
                        self.output_suffix,
                    ),
                    paired=True,
                    y_label=f"ΔF/F₀        ROI {i+1}",
                    Groups_Name=[
                        "{}+{}".format(self.trigger_config.stim_1_name, self.trigger_config.stim_2_name),
                        self.trigger_config.stim_2_name,
                    ],
                )

        # save vertical shift for the next calculations
        load_vshift = (
            csv_path
            + csv_file
            + "$"
            + str(self.trigger_config.vertical_shift_of_trig - 1)
            + "$"
            + "MASK FOR ANY ENDING HERE"
        )
        if self.trigger_config.vertical_shift_of_trig and load_vshift in self.v_shifts:
            self.vertical_shift = self.v_shifts[load_vshift]
        if not self.vertical_shift or self.vertical_shift == 0:
            vertical_shift = np.amax(s2_ampl_list_each_by_roi)
        else:
            vertical_shift = self.vertical_shift

        self.v_shifts_return |= {unit_id: vertical_shift}

        # CSV all traces in timeframe
        matrix = self.csv_matrix[
            int(
                (
                    self.trigger_config.start_from_epoch
                    * self.trigger_config.step_duration
                    * self.n_steps_per_epoch
                )
                / self.movie_config.seconds_per_frame_adjusted
            ) : int(
                (
                    (self.trigger_config.start_from_epoch + self.trigger_config.n_epochs + 1)
                    * self.trigger_config.step_duration
                    * self.n_steps_per_epoch
                )
                / self.movie_config.seconds_per_frame_adjusted
            )
        ]
        matrix_T = self.transpose(matrix)

        # save them to CSV
        self.csv_write(
            matrix,
            csv_path + output_dir,
            output_dir,
            "_full_traces_raw_{0}_and_{1}_ampl_{2}_auto_".format(
                self.group_names[0], self.group_names[1], self.output_suffix
            ),
        )

        # plot them all (slows script down)
        self.plot_traces(
            matrix_T[0],
            matrix_T[1:],
            [],
            csv_path
            + output_dir
            + "/"
            + "_full_traces_raw_{0}_and_{1}_ampl_{2}_auto_.png".format(
                self.group_names[0], self.group_names[1], self.output_suffix
            ),
            linewidth=0.5,
            dpi=400,
        )

        # plot debug graph to check time sync
        if DEBUG:
            self.debug_sync_during_trace_calculation(csv_path, output_dir)

        # st1(s1s2) to st2(s2) ampl ratio by epoch,
        # plotted as small bars on the right of each stacked trace
        if (s1s2 or s1) and s2:
            with np.errstate(divide="ignore", invalid="ignore"):
                ampl_st1_to_st2_ratio_rois_by_epoch = 1 / np.array(
                    ampl_st2_to_st1_ratio_rois_by_epoch, dtype=float
                )
        else:
            ampl_st1_to_st2_ratio_rois_by_epoch = None

        # stim timings within an epoch, shaded behind the traces
        stim_timings = []
        if s1s2:
            stim_timings.append((self.s1s2_delay, "magenta"))
        elif s1:
            stim_timings.append((self.s1_delay, "magenta"))
        if s2:
            stim_timings.append((self.s2_delay, "green"))

            # plot_stacked_traces all togather
        for i in self.group_names:
            os.makedirs(
                csv_path
                + output_dir
                + "/_by_rois_traces_bin_{0}_{1}_auto_".format(i, self.output_suffix),
                exist_ok=True,
            )

        # one multipage PDF per group, collecting every stacked traces page
        stacked_pdfs = [
            PdfPages(
                "{0}{1}/_by_rois_traces_bin_{2}_{3}_auto_/{4}_{2}_traces_stacked.pdf".format(
                    csv_path, output_dir, group, self.output_suffix, self.file_nosuffix
                )
            )
            for group in self.group_names
        ]

        # printable letter format: all the traces stacked
        self.plot_stacked_traces(
            np.array(matrix_T[0])
            - (self.trigger_config.start_from_epoch * self.trigger_config.step_duration * self.n_steps_per_epoch),
            matrix_T[:],
            s1s2_OR_s2_bin_list_each_by_epoch,
            st1_OR_st2_bin_summary_by_rois,
            "{0}{1}/_by_rois_traces_bin_{2}_{3}_auto_/_full_traces_stacked_by_rois_auto_.png".format(
                csv_path, output_dir, self.group_names[0], self.output_suffix
            ),
            vertical_shift=vertical_shift,
            delay=self.s2_delay,
            ratio_by_rois=ampl_st1_to_st2_ratio_rois_by_epoch,
            figsize=LETTER_FIGSIZE,
            stim_timings=stim_timings,
            pdf=stacked_pdfs[0],
        )
        # printable letter format: only the responsive (green) traces
        self.plot_stacked_traces(
            np.array(matrix_T[0])
            - (self.trigger_config.start_from_epoch * self.trigger_config.step_duration * self.n_steps_per_epoch),
            matrix_T[:],
            s1s2_OR_s2_bin_list_each_by_epoch,
            st1_OR_st2_bin_summary_by_rois,
            "{0}{1}/_by_rois_traces_bin_{2}_{3}_auto_/_full_traces_stacked_by_rois_responsive_auto_.png".format(
                csv_path, output_dir, self.group_names[0], self.output_suffix
            ),
            vertical_shift=vertical_shift,
            delay=self.s2_delay/2,
            ratio_by_rois=ampl_st1_to_st2_ratio_rois_by_epoch,
            figsize=LETTER_FIGSIZE,
            only_responsive=True,
            stim_timings=stim_timings,
            pdf=stacked_pdfs[0],
        )
        # printable letter format: all the traces stacked
        self.plot_stacked_traces(
            np.array(matrix_T[0])
            - (self.trigger_config.start_from_epoch * self.trigger_config.step_duration * self.n_steps_per_epoch),
            matrix_T[:],
            s1_OR_s2_bin_list_each_by_epoch,
            st1_OR_st2_bin_summary_by_rois,
            "{0}{1}/_by_rois_traces_bin_{2}_{3}_auto_/_full_traces_stacked_by_rois_auto_.png".format(
                csv_path, output_dir, self.group_names[1], self.output_suffix
            ),
            vertical_shift=vertical_shift,
            delay=self.s2_delay/2,
            ratio_by_rois=ampl_st1_to_st2_ratio_rois_by_epoch,
            figsize=LETTER_FIGSIZE,
            stim_timings=stim_timings,
            pdf=stacked_pdfs[1],
        )
        # printable letter format: only the responsive (green) traces
        self.plot_stacked_traces(
            np.array(matrix_T[0])
            - (self.trigger_config.start_from_epoch * self.trigger_config.step_duration * self.n_steps_per_epoch),
            matrix_T[:],
            s1_OR_s2_bin_list_each_by_epoch,
            st1_OR_st2_bin_summary_by_rois,
            "{0}{1}/_by_rois_traces_bin_{2}_{3}_auto_/_full_traces_stacked_by_rois_responsive_auto_.png".format(
                csv_path, output_dir, self.group_names[1], self.output_suffix
            ),
            vertical_shift=vertical_shift,
            delay=self.s2_delay/2,
            ratio_by_rois=ampl_st1_to_st2_ratio_rois_by_epoch,
            figsize=LETTER_FIGSIZE,
            only_responsive=True,
            stim_timings=stim_timings,
            pdf=stacked_pdfs[1],
        )

        # plot_stacked_traces by groups, printable letter format
        chunk_size = 20
        for pos in range(0, len(self.csv_matrix[0]) - 1, chunk_size):
            self.plot_stacked_traces(
                np.array(matrix_T[0])
                - (
                    self.trigger_config.start_from_epoch
                    * self.trigger_config.step_duration
                    * self.n_steps_per_epoch
                ),
                matrix_T[pos : pos + chunk_size + 1],
                s1s2_OR_s2_bin_list_each_by_epoch[pos : pos + chunk_size + 1],
                st1_OR_st2_bin_summary_by_rois[pos : pos + chunk_size + 1],
                "{0}{1}/_by_rois_traces_bin_{2}_{5}_auto_/_full_traces_stacked_by_rois_{3}-{4}_{5}_auto_.png".format(
                    csv_path,
                    output_dir,
                    self.group_names[0],
                    pos + 1,
                    pos + chunk_size,
                    self.output_suffix,
                ),
                vertical_shift=vertical_shift,
                delay=self.s2_delay/2,
                ratio_by_rois=(
                    None
                    if ampl_st1_to_st2_ratio_rois_by_epoch is None
                    else ampl_st1_to_st2_ratio_rois_by_epoch[pos : pos + chunk_size + 1]
                ),
                figsize=LETTER_FIGSIZE,
                stim_timings=stim_timings,
                pdf=stacked_pdfs[0],
            )
        for pos in range(0, len(self.csv_matrix[0]) - 1, chunk_size):
            self.plot_stacked_traces(
                np.array(matrix_T[0])
                - (
                    self.trigger_config.start_from_epoch
                    * self.trigger_config.step_duration
                    * self.n_steps_per_epoch
                ),
                matrix_T[pos : pos + chunk_size + 1],
                s1_OR_s2_bin_list_each_by_epoch[pos : pos + chunk_size + 1],
                st1_OR_st2_bin_summary_by_rois[pos : pos + chunk_size + 1],
                "{0}{1}/_by_rois_traces_bin_{2}_{5}_auto_/_full_traces_stacked_by_rois_{3}-{4}_{5}_auto_.png".format(
                    csv_path,
                    output_dir,
                    self.group_names[1],
                    pos + 1,
                    pos + chunk_size,
                    self.output_suffix,
                ),
                vertical_shift=vertical_shift,
                delay=self.s2_delay/2,
                ratio_by_rois=(
                    None
                    if ampl_st1_to_st2_ratio_rois_by_epoch is None
                    else ampl_st1_to_st2_ratio_rois_by_epoch[pos : pos + chunk_size + 1]
                ),
                figsize=LETTER_FIGSIZE,
                stim_timings=stim_timings,
                pdf=stacked_pdfs[1],
            )

        for stacked_pdf in stacked_pdfs:
            stacked_pdf.close()

        # # plot_traces_by_rois
        # for i in range(len(s1s2_raw_line_list)):
        #     self.plot_traces_by_rois(
        #         s1s2_raw_line_list[i],
        #         s2_raw_line_list[i],
        #         "{0}{1}/_epoch{2}_AC_C_traces_auto_.png".format(
        #             csv_path, output_dir[:], i + self.trigger_config.start_from_epoch
        #         ),
        #     )

        # plot_heatmaps
        # self.plot_heatmap(
        #     matrix_T[:],
        #     "{0}{1}/_by_rois_heatmap_bin_{2}_{3}{4}_in_{5}_auto.png".format(
        #         csv_path, output_dir, self.group_names[0], self.file_nosuffix, self.output_suffix, base_dir
        #     ),
        #     s1s2_OR_s2_bin_list_each_by_epoch,
        #     st1_OR_st2_bin_summary_by_rois,
        #     delay=(self.trigger_config.step_duration * s1s2_order) + (self.trigger_config.start_from_epoch * self.trigger_config.step_duration * self.n_steps_per_epoch),
        # )
        self.plot_heatmap(
            matrix_T[:],
            "{0}{1}/_by_rois_heatmap_bin_{2}_{3}{4}_in_{5}_auto.png".format(
                csv_path, output_dir, self.group_names[1], self.file_nosuffix, self.output_suffix, base_dir
            ),
            s1_OR_s2_bin_list_each_by_epoch,
            st1_OR_st2_bin_summary_by_rois,
            delay=(self.trigger_config.step_duration * s2_order) + (self.trigger_config.start_from_epoch * self.trigger_config.step_duration * self.n_steps_per_epoch),
        )
        self.plot_heatmap(
            matrix_T[:],
            "{0}{1}/_by_rois__heatmap_{2}{3}_in_{4}_auto.png".format(
                csv_path, output_dir, self.file_nosuffix, self.output_suffix, base_dir
            ),
            delay=None,
        )

    def plot_s2_to_s1s2_ratio_rois_by_epoch(self, array, path):

        # Create the plot
        plt.figure(figsize=(15, 10))  # Set the figure size to 10x15 inches
        x = list(range(1, len(array[0]) + 1))

        for roi in array:
            plt.plot(x, roi, marker="o", linestyle="-", color="k")

        plt.title(
            "{1} to {0}+{1} resp amplitude ratio by time".format(
                self.trigger_config.stim_1_name, self.trigger_config.stim_2_name
            )
        )
        plt.xlabel("epoch")
        plt.ylabel(
            "{1} to {0}+{1} resp amplitude ratio".format(
                self.trigger_config.stim_1_name, self.trigger_config.stim_2_name
            )
        )
        plt.savefig(path)
        plt.close()

    def plot_s1s2_s2_roi_stats(
        self, group1, group2, path, paired=True, y_label="", Groups_Name=[]
    ):

        data = [group1, group2]

        # set the parameters:
        paired = True  # is groups dependend or not
        tails = 2  # two-tailed or one-tailed result

        # initiate the analysis
        analysis = AutoStatLib.StatisticalAnalysis(
            data,
            paired=paired,
            tails=tails,
            verbose=False,
            groups_name=Groups_Name,
        )

        analysis.RunWilcoxon()
        results = analysis.GetResult()

        if "p_value_exact" in results:
            plot = AutoStatLib.StatPlots.BarStatPlot(
                data,
                **results,
                y_label=y_label,
                figure_scale_factor=0.8,
                figure_h=4,
                figure_w=0,
            )
        else:
            plot = AutoStatLib.StatPlots.BarStatPlot(data, dependent=True)
        plot.plot()
        plot.save(path)
        plot.close()

    def plot_traces_by_rois(self, array1, array2, path):
        plt.figure()

        # Plot lines from the first array in black
        x = array1[0]
        for y in array1[1:]:
            plt.plot(x, y, "k-", alpha=0.5)

        # Plot lines from the second array in red
        x = array2[0]
        for y in array2[1:]:
            plt.plot(x, y, "r-", alpha=0.5)

        plt.savefig(path)
        plt.close()

    def plot_stacked_traces(
        self,
        x,
        array,
        bin,
        bin_summary_by_rois,
        path,
        vertical_shift=1,
        delay=0,
        ratio_by_rois=None,
        ratio_max=None,    # optional ceiling for the bars, in ratio units
        figsize=(10, 10),
        only_responsive=False,
        stim_timings=None,    # [(delay within an epoch, color), ...] to shade
        pdf=None,
    ):
        fig = plt.figure(figsize=figsize, dpi=300)

        # rois to plot: all of them, or only the responsive (green) ones
        rois = [
            i
            for i in range(len(array) - 1)
            if not only_responsive or bin_summary_by_rois[i]
        ]

        traces_low, traces_high = None, None
        for pos, i in enumerate(rois):
            y = array[i + 1]
            color = "g-" if bin_summary_by_rois[i] else "k-"
            vertical_shifted_y = [val + pos * vertical_shift for val in y]
            plt.plot(x, vertical_shifted_y, color, linewidth=0.7, alpha=1)

            # vertical extent of the traces, to fit the stim boxes in
            traces_low = min(vertical_shifted_y + ([] if traces_low is None else [traces_low]))
            traces_high = max(vertical_shifted_y + ([] if traces_high is None else [traces_high]))

            plt.plot(
                [
                    (
                        j * self.trigger_config.step_duration * self.n_steps_per_epoch + delay
                        if bin[i][j]
                        else None
                    )
                    for j, dot in enumerate(bin[i])
                ],
                [pos * vertical_shift] * len((bin[i])),
                "rx",
            )

        # Set y-tick labels divided by vertical_shift, starting from 1, and rounded to integers
        ax = plt.gca()
        # y_ticks = ax.get_yticks()
        # ax.set_yticks(y_ticks)
        # ax.set_yticklabels([f'{int(round(y / vertical_shift + 1))}' for y in y_ticks])

        # Remove y-axis ticks
        ax.set_yticks([])

        # light shaded box under the traces for each stim event of each epoch
        if stim_timings and traces_low is not None:
            step = self.trigger_config.step_duration
            for timing, box_color in stim_timings:
                for j in range(self.trigger_config.n_epochs):
                    ax.add_patch(
                        plt.Rectangle(
                            (
                                (j * step * self.n_steps_per_epoch) + timing - step * 0.25,
                                traces_low,
                            ),
                            step * 0.75,
                            traces_high - traces_low,
                            color=box_color,
                            alpha=0.1,
                            linewidth=0,
                            zorder=0,
                        )
                    )

        # small bar plot to the right of each trace:
        # one bar per epoch with the st1(s1s2) to st2(s2) ampl ratio
        if ratio_by_rois is not None and len(ratio_by_rois):
            x_span = np.max(x) - np.min(x)
            bars_block_width = x_span * 0.05  # 5% of the trace width
            bars_start = np.max(x) + x_span * 0.01  # small gap after the trace

            ratios = [
                (pos, i, np.asarray(ratio_by_rois[i], dtype=float).ravel())
                for pos, i in enumerate(rois)
                if i < len(ratio_by_rois)
            ]

            positions, widths, heights, bottoms, colors = [], [], [], [], []
            scale_levels = []
            zero_levels = []
            for pos, i, values in ratios:
                if not values.size:
                    continue
                bar_step = bars_block_width / values.size
                for j, value in enumerate(values):
                    if not np.isfinite(value) or value <= 0:
                        continue
                    positions.append(bars_start + (j + 0.5) * bar_step)
                    widths.append(bar_step * 0.8)
                    # constant bars scale, independent of the data
                    heights.append(
                        (min(value, ratio_max) if ratio_max else value)
                        * vertical_shift
                        * RATIO_BAR_UNIT
                    )
                    bottoms.append(pos * vertical_shift)
                    colors.append("tab:purple" if bin_summary_by_rois[i] else "k")
                # level of the ratio = 1 on the bars scale
                scale_levels.append(
                    (pos * vertical_shift) + vertical_shift * RATIO_BAR_UNIT
                )
                zero_levels.append(
                    pos * vertical_shift
                )

            ax.bar(
                positions,
                heights,
                width=widths,
                bottom=bottoms,
                color=colors,
                linewidth=0,
            )

            # dotted line marking the ratio = 1 level of the bars
            ax.hlines(
                scale_levels,
                bars_start,
                bars_start + bars_block_width,
                colors="green",
                linestyles=":",
                linewidth=0.7,
                zorder=4,
            )

            # solid white line on y=0 to separate bars from vertical overlay
            ax.hlines(
                zero_levels,
                bars_start,
                bars_start + bars_block_width,
                colors="white",
                linestyles="-",
                linewidth=0.7,
                zorder=3,
            )


        ax.errorbar(
            -15,
            -0.5,
            yerr=0.5,
            fmt="none",
            capsize=4,
            ecolor="k",
            linewidth=2,
            zorder=3,
        )

        ax.set_rasterization_zorder(6) 

        plt.text(
            -15, -1.0, "1 ΔF/F₀", horizontalalignment="center", verticalalignment="top"
        )

        for pos, i in enumerate(rois):
            plt.text(
                -20,
                (pos * vertical_shift),
                f"{i+1}",
                horizontalalignment="center",
                verticalalignment="bottom",
            )

        # Save the plot as plot.png, and as a page of the multipage PDF
        plt.tight_layout()
        plt.savefig(path, transparent=False)
        if pdf is not None:
            pdf.savefig(fig)
        plt.close()

    def plot_heatmap(self, matrix, path, bin=[], bin_summary_by_rois=[], delay=0, vmin=0, vmax=2.5, norm=None):
        '''
        Plot a heatmap of the given matrix.
        'norm' can be 'linear','log','symlog','asinh','logit','function','functionlog'
        '''

        array = np.array(matrix[1:])  # Exclude the x-axis row
        array = array[::-1]  # reverse matrix along y axis
        x = np.array(matrix[0])  # x-axis values

        # Create the heatmap
        plt.figure(figsize=(14, 10))
        plt.imshow(
            array,
            aspect="auto",
            cmap="gnuplot",
            interpolation="nearest",
            origin="upper",
            extent=[x[0], x[-1], len(array), 0],
            vmin=vmin,
            vmax=vmax,
            norm=norm,   # 'linear','log','symlog','asinh','logit','function','functionlog'
        )
        plt.colorbar(label="ΔF/F₀")

        # Overlay bin events
        if bin and bin_summary_by_rois:
            for i in range(len(array)):
                if bin_summary_by_rois[i]:
                    plt.plot(
                        min(x) - 5, len(array) - i - 0.5, "wo", markeredgecolor="g"
                    )
                for j, dot in enumerate(bin[i]):
                    if dot:
                        event_x = (
                            j * self.trigger_config.step_duration * self.n_steps_per_epoch
                            + delay
                        )
                        plt.plot(
                            event_x, len(array) - i - 0.5, "wx", markeredgecolor="g"
                        )

        # plt.xlabel('Time')
        # plt.ylabel('ROIs')
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def run(self, detailed_stats=True):
        csv_list = []
        csv_list.extend(
            self.file_lister(
                # # this regex will find files like 'file.csv', 'file_something.csv', but not 'file_something.xlsx' or 'file_something.csv_backup'
                r"^" + re.escape(self.file[:-4]) + r".*\.csv$", nonrecursive=True

                # # this regex will find files like 'file.csv', but not 'file_something.csv' or 'file_something.xlsx'
                # r"^" + re.escape(self.file[:-4]) + r".csv$", nonrecursive=True
            )
        )

        if csv_list:

            for i, [csv_path, csv_file] in enumerate(csv_list):
                content_raw = self.csv_read(csv_path, csv_file)
                content = self.csv_transform(content_raw)

                # For multievent movies:
                # for i, event in enumerate(self.events):
                #     csv_output = self.csv_cutter(content, *event)
                #     try:
                #         self.csv_write(csv_output, csv_path, csv_file)
                #     except PermissionError:
                #         self.logging('       File actually opened:')
                #         continue

                self.csv_matrix = self.csv_cutter(content)
                try:
                    self.csv_write(
                        self.csv_matrix,
                        csv_path,
                        csv_file + ".csv",
                        CALCULATIONS_SUBFOLDER_NAME + self.output_suffix,
                        subdir=True,
                    )
                except PermissionError as e:
                    self.logging("       File actually opened:" + repr(e))
                    continue

                if detailed_stats:
                    self.detailed_stats(
                        csv_path,
                        csv_file,
                        csv_file
                        + ".csv"
                        + CALCULATIONS_SUBFOLDER_NAME
                        + self.output_suffix,
                    )

            result = "***    Done: {} csv files for      {}".format(
                len(csv_list), self.file_path
            )

        else:
            result = "---    Skip: no csv files for      {}".format(self.file_path)

        csv_list = None
        self.logging(result)
        return result
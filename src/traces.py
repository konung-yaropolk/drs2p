
import re
import os
import csv
import AutoStatLib
import matplotlib.pyplot as plt
import numpy as np

from debug import Debug
from helpers import Helpers

CALCULATIONS_SUBFOLDER_NAME = "_CALCULATIONS_auto_"
BINARIZATION_RESP_THRESHOLD = 0.29
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
        self.file_path=os.path.join(run_config.working_dir, movie_config.file_name)
        self.filename_suffix, self.file_nosuffix = self.calculate_suffix_and_nosuffix(self.file_path)
        self.s_trig_time = movie_config.events[trigger_config.trig_number - 1][1]
        self.output_suffix = "_"+self.trigger_config.label
        self.path = os.path.dirname(self.file_path)
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
        content = (float(i) - time for i in list(zip(*content))[0])
        diffs = [abs(i) for i in content]
        index = diffs.index(min(diffs))

        return index

    def data_normalize(self, content, start, zero):
        content_normalized = []

        for column in content:
            baseline = column[start:zero]
            baseline_sum = sum((float(cell) for cell in baseline))
            baseline_len = len(baseline)
            mean = baseline_sum / baseline_len if baseline_len and baseline else 0

            column_normalized = [
                (float(cell) - mean) / mean if mean else 0 for cell in column
            ]  # ΔF/F₀
            # column_normalized = [float(cell)/mean if mean else 1 for cell in column]   # ΔF/F

            content_normalized.append(column_normalized)

        return content_normalized

    def csv_cutter(self, content):
        timeline_zero = (float(i) - self.s_trig_time for i in list(zip(*content))[0])

        start = (
            self.find_time_index(content, self.s_trig_time - self.trigger_config.time_before_trig)
            if self.trigger_config.time_before_trig
            else None
        )

        start_bl = (
            self.find_time_index(content, self.s_trig_time - self.trigger_config.baseline_duraton)
            if self.trigger_config.baseline_duraton
            else start
        )

        zero = self.find_time_index(content, self.s_trig_time)

        end = (
            self.find_time_index(content, self.s_trig_time + self.trigger_config.time_after_trig)
            if self.trigger_config.time_after_trig
            else None
        )

        content = list(zip(*content))[1:]
        content[:0] = [timeline_zero]

        if self.trigger_config.relative_values:
            content[1:] = self.data_normalize(content[1:], start_bl, zero)

        csv_output = list(zip(*content))[start:end]
        csv_output_np = np.array(csv_output)

        return csv_output_np

    def csv_transform(
        self,
        content_raw,
    ):

        mean_col = self.trigger_config.mean_col_order  # order of "Mean" col in measurments
        n_cols = self.trigger_config.cols_per_roi  # n of measurments for each ROI

        first_col = (str(i * self.movie_config.seconds_per_frame) for i in range(len(content_raw)))
        content = list(zip(*content_raw))[mean_col::n_cols]
        content[:0] = [first_col]
        content = list(zip(*content))[1:]

        return content

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
        raw_line_list = [x[whole_step_indices] - start]

        for trace in traces:
            # Calculate baseline
            baseline = np.mean(trace[bl_indices])
            # Baseline correction
            corrected_trace = trace - baseline
            # Peak amplitude in signal period
            ampl = np.max(corrected_trace[sig_indices])
            ampl_list.append(ampl)
            # AUC in signal period
            auc = np.trapezoid(corrected_trace[sig_indices], x[sig_indices])
            auc_list.append(auc)
            # biarization
            bin_list.append(ampl > self.trigger_config.sigmas_treshold * np.std(trace[bl_indices]))

            raw_line_list.append(corrected_trace[whole_step_indices])

            # Debug plot
            # self.logging(ampl > self.trigger_config.sigmas_treshold *
            #                 np.std(trace[bl_indices]))
            # plt.plot(corrected_trace[whole_step_indices])
            # plt.show()
            # plt.close()

        # Calculate mean amplitude and AUC across all traces
        ampl_mean_of_rois = np.mean(ampl_list)
        auc_mean_of_rois = np.mean(auc_list)

        return (
            ampl_mean_of_rois,
            ampl_list,
            auc_mean_of_rois,
            auc_list,
            bin_list,
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
            raw_line_list,
        ) = [
            [
                self.calculate_ampl_auc_bin(
                    (i * self.s_epoch_duration) + delay - self.trigger_config.step_duration / 2,
                    (i * self.s_epoch_duration) + delay,
                    (i * self.s_epoch_duration) + delay,
                    (i * self.s_epoch_duration) + delay + self.trigger_config.step_duration / 2,
                )[j]
                for i in range(self.trigger_config.start_from_epoch, self.trigger_config.n_epochs)
            ]
            for j in range(6)
        ]

        ampl_list_each_by_epoch = self.transpose(ampl_list_each_by_roi)
        auc_list_each_by_epoch = self.transpose(auc_list_each_by_roi)
        bin_list_each_by_epoch = self.transpose(bin_list_each_by_roi)
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
        # unit_id = self.file_path + '%' + str(self.trigger_config.trig_number)
        unit_id = (
            csv_path + csv_file + "$" + str(self.trigger_config.trig_number) + "$" + self.output_suffix
        )

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
        pythonst1_ampl_mean_of_epochs_by_rois = []
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

        # Binarization:

        if not s1s2 and not s1:
            s1s2_bin_list_each_by_epoch = s2_bin_list_each_by_epoch
            s1_bin_list_each_by_epoch = s2_bin_list_each_by_epoch
        if not s1s2 and s1:
            s1s2_bin_list_each_by_epoch = s1_bin_list_each_by_epoch
        if not s1 and s1s2:
            s1_bin_list_each_by_epoch = s1s2_bin_list_each_by_epoch

        if len(self.group_names) == 1:
            self.group_names.insert(0, "_")

        if not s1 and s1s2:
            st1_bin_summary_by_rois = [
                sum(i) / len(i) > BINARIZATION_RESP_THRESHOLD
                for i in s1s2_bin_list_each_by_epoch
            ]
        if not s1s2 and s1:
            st1_bin_summary_by_rois = [
                sum(i) / len(i) > BINARIZATION_RESP_THRESHOLD
                for i in s1_bin_list_each_by_epoch
            ]
        st2_bin_summary_by_rois = [
            sum(i) / len(i) > BINARIZATION_RESP_THRESHOLD
            for i in s2_bin_list_each_by_epoch
        ]

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
                        self.filter_list(st1_ampl_mean_of_epochs_by_rois, filter[2]),
                        self.filter_list(st2_ampl_mean_of_epochs_by_rois, filter[2]),
                        self.filter_list(
                            1 / ampl_st2_to_st1_ratio_mean_of_epochs_by_rois, filter[2]
                        ),
                        "",
                        "",
                        self.filter_list(st1_ampl_mean_of_epochs_by_rois, filter[1]),
                        self.filter_list(st2_ampl_mean_of_epochs_by_rois, filter[1]),
                        self.filter_list(
                            1 / ampl_st2_to_st1_ratio_mean_of_epochs_by_rois, filter[1]
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
                        self.filter_list(st1_auc_mean_of_epochs_by_rois, filter[2]),
                        self.filter_list(st2_auc_mean_of_epochs_by_rois, filter[2]),
                        self.filter_list(
                            1 / auc_st2_to_st1_ratio_mean_of_epochs_by_rois, filter[2]
                        ),
                        "",
                        "",
                        self.filter_list(st1_auc_mean_of_epochs_by_rois, filter[1]),
                        self.filter_list(st2_auc_mean_of_epochs_by_rois, filter[1]),
                        self.filter_list(
                            1 / auc_st2_to_st1_ratio_mean_of_epochs_by_rois, filter[1]
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
            self.filter_list(st1_auc_mean_of_epochs_by_rois, filter[2], replace=False),
            self.filter_list(st2_auc_mean_of_epochs_by_rois, filter[2], replace=False),
            "{0}{1}/_by_rois_{2}_{3}_{4}_auc_auto_.png".format(
                csv_path,
                output_dir,
                self.group_names[0],
                self.group_names[1],
                self.output_suffix,
            ),
            paired=True,
            y_label="AUC",
            Groups_Name=[self.group_names[0], self.group_names[1]],
        )

        # plot_s1s2_s2_roi_stats Ampl for all rois
        self.plot_s1s2_s2_roi_stats(
            self.filter_list(st1_ampl_mean_of_epochs_by_rois, filter[2], replace=False),
            self.filter_list(st2_ampl_mean_of_epochs_by_rois, filter[2], replace=False),
            "{0}{1}/_by_rois_{2}_{3}_{4}_ampl_auto_.png".format(
                csv_path,
                output_dir,
                self.group_names[0],
                self.group_names[1],
                self.output_suffix,
            ),
            paired=True,
            y_label="ΔF/F₀",
            Groups_Name=[self.group_names[0], self.group_names[1]],
        )

        # plot_s1s2_s2_roi_stats for each roi during timeline
        if s1s2 and s2:
            for i in range(len(s1s2_ampl_list_each_by_epoch)):
                self.plot_s1s2_s2_roi_stats(
                    s1s2_ampl_list_each_by_epoch[i],
                    s2_ampl_list_each_by_epoch[i],
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
                    (self.trigger_config.start_from_epoch)
                    * self.trigger_config.step_duration
                    * self.n_steps_per_epoch
                )
                / self.movie_config.seconds_per_frame
            ) : int(
                (
                    (self.trigger_config.start_from_epoch + self.trigger_config.n_epochs + 1)
                    * self.trigger_config.step_duration
                    * self.n_steps_per_epoch
                )
                / self.movie_config.seconds_per_frame
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

            # plot_stacked_traces all togather
        for i in self.group_names:
            os.makedirs(
                csv_path
                + output_dir
                + "/_by_rois_traces_bin_{0}_{1}_auto_".format(i, self.output_suffix),
                exist_ok=True,
            )

        self.plot_stacked_traces(
            np.array(matrix_T[0])
            - ((self.trigger_config.start_from_epoch) * self.trigger_config.step_duration * self.n_steps_per_epoch),
            matrix_T[:],
            s1s2_bin_list_each_by_epoch,
            st1_bin_summary_by_rois,
            "{0}{1}/_by_rois_traces_bin_{2}_{3}_auto_/_full_traces_stacked_by_rois_auto_.png".format(
                csv_path, output_dir, self.group_names[0], self.output_suffix
            ),
            vertical_shift=vertical_shift,
            delay=0,
        )
        self.plot_stacked_traces(
            np.array(matrix_T[0])
            - ((self.trigger_config.start_from_epoch) * self.trigger_config.step_duration * self.n_steps_per_epoch),
            matrix_T[:],
            s2_bin_list_each_by_epoch,
            st2_bin_summary_by_rois,
            "{0}{1}/_by_rois_traces_bin_{2}_{3}_auto_/_full_traces_stacked_by_rois_auto_.png".format(
                csv_path, output_dir, self.group_names[1], self.output_suffix
            ),
            vertical_shift=vertical_shift,
            delay=self.s2_delay,
        )

        # plot_stacked_traces by groups
        chunk_size = 20
        for pos in range(0, len(self.csv_matrix[0]) - 1, chunk_size):
            self.plot_stacked_traces(
                np.array(matrix_T[0])
                - (
                    (self.trigger_config.start_from_epoch)
                    * self.trigger_config.step_duration
                    * self.n_steps_per_epoch
                ),
                matrix_T[pos : pos + chunk_size + 1],
                s1s2_bin_list_each_by_epoch[pos : pos + chunk_size + 1],
                st1_bin_summary_by_rois[pos : pos + chunk_size + 1],
                "{0}{1}/_by_rois_traces_bin_{2}_{5}_auto_/_full_traces_stacked_by_rois_{3}-{4}_{5}_auto_.png".format(
                    csv_path,
                    output_dir,
                    self.group_names[0],
                    pos + 1,
                    pos + chunk_size,
                    self.output_suffix,
                ),
                vertical_shift=vertical_shift,
                delay=self.s2_delay,
            )
        for pos in range(0, len(self.csv_matrix[0]) - 1, chunk_size):
            self.plot_stacked_traces(
                np.array(matrix_T[0])
                - (
                    (self.trigger_config.start_from_epoch)
                    * self.trigger_config.step_duration
                    * self.n_steps_per_epoch
                ),
                matrix_T[pos : pos + chunk_size + 1],
                s2_bin_list_each_by_epoch[pos : pos + chunk_size + 1],
                st2_bin_summary_by_rois[pos : pos + chunk_size + 1],
                "{0}{1}/_by_rois_traces_bin_{2}_{5}_auto_/_full_traces_stacked_by_rois_{3}-{4}_{5}_auto_.png".format(
                    csv_path,
                    output_dir,
                    self.group_names[1],
                    pos + 1,
                    pos + chunk_size,
                    self.output_suffix,
                ),
                vertical_shift=vertical_shift,
                delay=self.s2_delay,
            )

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
        self.plot_heatmap(
            matrix_T[:],
            "{0}{1}/_by_rois__heatmap_bin_{2}_{3}_auto_.png".format(
                csv_path, output_dir, self.group_names[0], self.output_suffix
            ),
            s1s2_bin_list_each_by_epoch,
            st1_bin_summary_by_rois,
            delay=0,
        )
        self.plot_heatmap(
            matrix_T[:],
            "{0}{1}/_by_rois__heatmap_bin_{2}_{3}_auto_.png".format(
                csv_path, output_dir, self.group_names[1], self.output_suffix
            ),
            s2_bin_list_each_by_epoch,
            st2_bin_summary_by_rois,
            delay=self.s2_delay,
        )
        self.plot_heatmap(
            matrix_T[:],
            "{0}{1}/_by_rois__heatmap_auto_{2}.png".format(
                csv_path, output_dir, self.output_suffix
            ),
            delay=self.s2_delay,
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
        self, x, array, bin, bin_summary_by_rois, path, vertical_shift=1, delay=0
    ):
        plt.figure(figsize=(10, 10))

        for i, y in enumerate(array[1:]):
            color = "g-" if bin_summary_by_rois[i] else "k-"
            vertical_shifted_y = [val + i * vertical_shift for val in y]
            plt.plot(x, vertical_shifted_y, color, linewidth=0.7, alpha=1)

            plt.plot(
                [
                    (
                        j * self.trigger_config.step_duration * self.n_steps_per_epoch + delay
                        if bin[i][j]
                        else None
                    )
                    for j, dot in enumerate(bin[i])
                ],
                [i * vertical_shift] * len((bin[i])),
                "rx",
            )

        # Set y-tick labels divided by vertical_shift, starting from 1, and rounded to integers
        ax = plt.gca()
        # y_ticks = ax.get_yticks()
        # ax.set_yticks(y_ticks)
        # ax.set_yticklabels([f'{int(round(y / vertical_shift + 1))}' for y in y_ticks])

        # Remove y-axis ticks
        ax.set_yticks([])

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

        plt.text(
            -15, -1.0, "1 ΔF/F₀", horizontalalignment="center", verticalalignment="top"
        )

        for i, y in enumerate(array[1:]):
            plt.text(
                -20,
                (i * vertical_shift),
                f"{i+1}",
                horizontalalignment="center",
                verticalalignment="bottom",
            )

        # Save the plot as plot.png
        plt.tight_layout()
        plt.savefig(path, transparent=False)
        plt.close()

    def plot_heatmap(self, matrix, path, bin=[], bin_summary_by_rois=[], delay=0):
        array = np.array(matrix[1:])  # Exclude the x-axis row
        array = array[::-1]  # reverse matrix along y axis
        x = np.array(matrix[0])  # x-axis values

        # Create the heatmap
        plt.figure(figsize=(14, 10))
        plt.imshow(
            array,
            aspect="auto",
            cmap="magma",
            interpolation="nearest",
            origin="upper",
            extent=[x[0], x[-1], len(array), 0],
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
                            + min(x)
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
        # print(f"TracesCalc.run() called for {self.file_nosuffix}{self.output_suffix}")
        csv_list = []
        csv_list.extend(
            self.file_lister(
                r"^" + re.escape(self.file_nosuffix) + r".*\.csv$", nonrecursive=True
            )
        )
        print(csv_list)

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
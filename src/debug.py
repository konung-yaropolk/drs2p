import numpy as np
import os

DERIVATIVES_SUBFOLDER_NAME = "_DERIVATIVES_auto_"


class Debug:

    def debug_sync_during_derivatives(
        self,
    ):

        x = np.array([i for i in range(self.n_frames)])
        y = np.array([np.mean(np.mean(self.img, axis=1), axis=1)])
        
        n_steps_per_epoch = len(self.trigger_config.drs_pattern[0])
        s_epoch_duration = self.trigger_config.step_duration * n_steps_per_epoch
        last_epoch_time = (
            self.s_trig_time
            + (self.trigger_config.start_from_epoch-1 + self.trigger_config.n_epochs - 1) * s_epoch_duration
        )
        start = self.sec_to_frame(last_epoch_time - self.trigger_config.step_duration)
        end = self.sec_to_frame(last_epoch_time + self.trigger_config.step_duration * 2)
        y2 = np.array([np.mean(np.mean(self.img, axis=1)[start:end], axis=1)])

        lines1 = [
            [i for i in range(*sorted(self.derivatives_frames_log)[j])]
            for j in range(len(self.derivatives_frames_log))
        ]

        lines2 = [
            [i for i in range(*sorted(self.derivatives_frames_log)[-j])]
            for j in range(1, n_steps_per_epoch + 1)
        ]

        self.derivatives_frames_log = [
            (i[0], i[1] - 1) for i in self.derivatives_frames_log
        ]

        lines3 = [[item - sublist[0] for item in sublist] for sublist in lines1]
        events3 = [
            [item - sublist[0] for item in sublist]
            for sublist in self.derivatives_frames_log.copy()
        ]

        x3 = [
            i
            for i in range(
                self.sec_to_frame(-self.trigger_config.step_duration / 4),
                self.sec_to_frame(self.trigger_config.step_duration / 2) - 1,
            )
        ]
        y3 = []

        for frame in self.derivatives_frames_log:
            appendix = []
            begin = frame[0] - self.sec_to_frame(self.trigger_config.step_duration / 4)
            finish = frame[0] + self.sec_to_frame(self.trigger_config.step_duration / 2)

            begin -= 1
            finish -= 1

            appendix = y[0][begin:finish].tolist()
            y3.append(appendix)

        # flatten inhomogeneous array
        lines1 = np.array([*self.flatten_array(lines1)])
        lines2 = np.array([*self.flatten_array(lines2)])
        lines3 = np.array([*self.flatten_array(lines3)])

        events1 = self.derivatives_frames_log.copy()
        events1.extend(lines1.tolist())

        events2 = sorted(self.derivatives_frames_log.copy())[
            -n_steps_per_epoch :
        ].copy()
        events2.extend(lines2.tolist())
        events3.extend(lines3.tolist())

        os.makedirs(
            f"{self.path}/{self.file}{DERIVATIVES_SUBFOLDER_NAME}{self.output_suffix}/debug/",
            exist_ok=True,
        )

        self.plot_traces(
            x,
            y,
            self.derivatives_frames_log,
            f"{self.path}/{self.file}{DERIVATIVES_SUBFOLDER_NAME}{self.output_suffix}/debug/debug_selected_epoch_{self.file_nosuffix}{self.output_suffix}.png",
            linewidth=0.5,
            fillcolor="violet",
            event_linecolor="orchid",
            event_linestyle="-",
            avg_linecolor="darkcyan",
            alpha=1,
            dpi=200,
        )

        self.plot_traces(
            x,
            y,
            events1,
            f"{self.path}/{self.file}{DERIVATIVES_SUBFOLDER_NAME}{self.output_suffix}/debug/debug_selected_epoch_{self.file_nosuffix}{self.output_suffix}.svg",
            linewidth=0.05,
            fillcolor="violet",
            event_linecolor="orchid",
            event_linestyle="-",
            avg_linecolor="darkcyan",
            alpha=1,
            dpi=800,
        )

        self.plot_traces(
            x3,
            y3,
            events3,
            f"{self.path}/{self.file}{DERIVATIVES_SUBFOLDER_NAME}{self.output_suffix}/debug/debug_sync_4all_{self.file_nosuffix}{self.output_suffix}.png",
            average=False,
            linewidth=1.5,
            linecolor="darkcyan",
            fillcolor="violet",
            event_linecolor="orchid",
            event_linestyle="-",
            avg_linecolor="darkcyan",
            alpha=0.5,
            fillalpha=1 / (n_steps_per_epoch * self.trigger_config.n_epochs),
            dpi=100,
            figsize=(10, 5),
        )

        # self.plot_traces(
        #     x[start:end],
        #     y2,
        #     events2,
        #     f"{self.path}/{self.file}{DERIVATIVES_SUBFOLDER_NAME}{self.output_suffix}/debug/debug_sync_{self.file_nosuffix}{self.output_suffix}.png",
        #     linewidth=0.5,
        #     fillcolor="violet",
        #     event_linecolor="orchid",
        #     event_linestyle="-",
        #     avg_linecolor="darkcyan",
        #     alpha=1,
        #     dpi=150,
        # )

    def debug_sync_during_trace_calculation(self, csv_path, output_dir):
        n_steps_per_epoch = len(self.trigger_config.drs_pattern[0])

        start = self.trigger_config.start_from_epoch * self.trigger_config.step_duration * n_steps_per_epoch
        end = start + self.trigger_config.step_duration
        start_frame = int(
            ((self.trigger_config.start_from_epoch) * self.trigger_config.step_duration * self.n_steps_per_epoch)
            / self.movie_config.seconds_per_frame
        )
        end_frame = int(
            (
                (self.trigger_config.start_from_epoch + 1.5)
                * self.trigger_config.step_duration
                * n_steps_per_epoch
            )
            / self.movie_config.seconds_per_frame
        )

        chunk = self.csv_matrix[start_frame:end_frame]
        chunk_T = self.transpose(chunk)
        self.plot_traces(
            chunk_T[0],
            chunk_T[1:],
            [start, end],
            csv_path
            + output_dir
            + "/"
            + "debug_sync_{}{}.png".format(self.file_nosuffix, self.output_suffix),
            linewidth=0.5,
            alpha=0.8,
            event_linecolor="red",
            event_linestyle="-",
            avg_linecolor="darkcyan",
            dpi=400,
        )
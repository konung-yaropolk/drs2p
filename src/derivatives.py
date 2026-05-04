import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from PIL import Image

from helpers import Helpers
from debug import Debug


# gaussian_filter sigma for derivatives calculation, to reduce noise and artifacts
# standard in ImageJ is 1.0
DERIVATIVES_SIGMA = 2.3
DERIVATIVES_SUBFOLDER_NAME = "_DERIVATIVES_auto_"
# save synchronization test plots to check
# sync of stim and recording
# and provide detailed errors traceback
DEBUG = True

class DerivativesCalc(Helpers, Debug):
    def __init__(self, 
                 run_config,
                 movie_config,
                 trigger_config):
        self.run_config = run_config
        self.movie_config = movie_config
        self.trigger_config = trigger_config
        self.s_trig_time = self.movie_config.events[self.trigger_config.trig_number-1][1] #recheck
        self.log = ' \n'
        self.file_path=os.path.join(run_config.working_dir, movie_config.file_name)
        self.path = os.path.dirname(self.file_path)
        self.file = os.path.basename(self.file_path)
        self.output_suffix = "_"+self.trigger_config.label
        self.filename_suffix, self.file_nosuffix = self.calculate_suffix_and_nosuffix(self.file_path)
        #these will be deleted once i deal with the helpers class
        self.s_movie_duration = movie_config.movie_duration
        self.spf = movie_config.seconds_per_frame * (1 + trigger_config.sync_coef)
        self.fps = 1 / self.spf
        self.n_frames = movie_config.n_frames

        self.log += "\nFile: {} \nMovie duration: {} \nn frames: {} \nSampling interval, s: {} \nTrigger time, s: {}".format(
            self.file_path,
            movie_config.movie_duration,
            movie_config.n_frames,
            self.movie_config.seconds_per_frame,
            self.s_trig_time,
        )
    def logging(self, *args, **kwargs):
        message = ' '.join(map(str, args))
        self.log += '\n' + message
    def compute_gaussian_derivatives(self, image_stack, start, end, sigma):

        # Compute derivatives along z-axis
        dz = gaussian_filter(
            image_stack[start:end], sigma=[sigma,  sigma,  sigma], order=[1, 0, 0]
        )

        return dz

    def process_tiff_stack(self, start, end):

        assert (
            end - start >= 2
        ), "!!! Error: {} - Too small interval to differentiate between frames {} and {}. ".format(
            self.file_path, start, end
        )
        assert start > 0, "!!! Error: Set the correct timing"
        assert end < self.n_frames, "!!! Error: Timing is out of movie duration"

        # self.logging("Frames ", start, ":", end)

        # Initialize a list to store the derivative images
        derivatives = self.compute_gaussian_derivatives(
            self.img, start, end, DERIVATIVES_SIGMA
        )

        # Average the derivatives to create a single image
        output_derivative = np.sum(np.maximum(derivatives, 0), axis=0)

        return output_derivative
    def average_sequence_responses(self, s_step_shift):

        sequence_stack = []
        n_steps_per_epoch = len(self.trigger_config.drs_pattern[0])
        s_epoch_duration = self.trigger_config.step_duration * n_steps_per_epoch

        for i in range(self.trigger_config.start_from_epoch-1, self.trigger_config.n_epochs + self.trigger_config.start_from_epoch-1):
            start = self.sec_to_frame(
                self.s_trig_time + (i * s_epoch_duration) + s_step_shift
            )
            end = self.sec_to_frame(
                self.s_trig_time
                + (i * s_epoch_duration)
                + s_step_shift
                + self.trigger_config.resp_duration
            )

            # Adjust for derivatives frame lag
            start += self.trigger_config.frame_lag_derivatives
            end += self.trigger_config.frame_lag_derivatives

            sequence_stack.append(self.process_tiff_stack(start, end))
            self.derivatives_frames_log.append((start, end))

        self.result = np.average(sequence_stack, axis=0)
    def calc_sequence(self, i, filename_ending):

        s_step_shift = self.trigger_config.step_duration * i
        self.average_sequence_responses(s_step_shift)

        metadata = {
            "axes": "YX",
            "min": 0,
            "max": np.max(self.result),
            " DERIVATIVES_SIGMA": DERIVATIVES_SIGMA,
        }

        self.save_tiff(
            os.path.join(
                self.path,
                self.file + DERIVATIVES_SUBFOLDER_NAME + self.output_suffix,
                filename_ending,
            ),
            self.result,
            metadata=metadata,
        )
    def run(self):

        stims_overlap_png = True
        stims_overlap_tif = True
        ratio_heatmap = True
        stims_substracted = True
        stims_substracted_diff = True

        self.derivatives_frames_log = []

        os.makedirs(
            self.file_path + DERIVATIVES_SUBFOLDER_NAME + self.output_suffix + "/",
            exist_ok=True,
        )

        # Open the TIFF image stack
        # rewrite self.n_frames to get actual length of the stack, not just a number from metadata
        self.img = tifffile.imread(self.file_path)

        self.n_frames = len(self.img)

        s1s2_name_ending = "_DERIVATIVES_auto_{}&{}_{}.tif".format(
            self.trigger_config.stim_1_name, self.trigger_config.stim_2_name, self.output_suffix
        )
        s1_name_ending = "_DERIVATIVES_auto_{}_{}.tif".format(
            self.trigger_config.stim_1_name, self.output_suffix
        )
        s2_name_ending = "_DERIVATIVES_auto_{}_{}.tif".format(
            self.trigger_config.stim_2_name, self.output_suffix
        )
        for i, (A, C) in enumerate(zip(self.trigger_config.drs_pattern[0], self.trigger_config.drs_pattern[1])):
            match (A, C):
                case (1, 1):
                    self.logging(
                        f"Sequence {self.trigger_config.stim_1_name}+{self.trigger_config.stim_2_name} done!"
                    )
                    self.calc_sequence(i, s1s2_name_ending)
                case (1, 0):
                    self.logging(f"Sequence {self.trigger_config.stim_1_name} done!")
                    self.calc_sequence(i, s1_name_ending)
                case (0, 1):
                    self.logging(f"Sequence {self.trigger_config.stim_2_name} done!")
                    self.calc_sequence(i, s2_name_ending)
                case (0, 0):
                    pass
                case (None, None):
                    self.calc_sequence(i, "DERIVATIVES_auto_.tif")

        self.logging(
            f"Taken {self.trigger_config.n_epochs} epochs: {self.trigger_config.start_from_epoch-1 + 1} to {self.trigger_config.n_epochs + self.trigger_config.start_from_epoch-1}\n"
        )

        if DEBUG:
            self.debug_sync_during_derivatives()


         # Different stims - differrent colors
        merger_s1s2_s2 = TifDerivativeProcess(
            os.path.join(
                self.path, self.file + DERIVATIVES_SUBFOLDER_NAME + self.output_suffix
            ),
            s1s2_name_ending,
            s2_name_ending,
            s1s2_name_ending,
            "_{2}_{1}-green_{0}&{1}-magenta_auto_{3}.tif".format(
                self.trigger_config.stim_1_name,
                self.trigger_config.stim_2_name,
                self.output_suffix,
                self.file_nosuffix,
            ),
            self.trigger_config.stim_1_name,
            self.trigger_config.stim_2_name,
            self.output_suffix,
            ratio_heatmap=ratio_heatmap,
            stims_overlap_png=stims_overlap_png,
            stims_overlap_tif=stims_overlap_tif,
            stims_substracted=False,
            stims_substracted_diff=False,
        )

        merger_s1s2_s2.process_directory()
        del merger_s1s2_s2

         # Different stims - differrent colors
        # merger_s1_s2 = TifDerivativeProcess(
        #     os.path.join(
        #         self.path, self.file + DERIVATIVES_SUBFOLDER_NAME + self.output_suffix
        #     ),
        #     s2_name_ending,
        #     s1_name_ending,
        #     s1_name_ending,
        #     "_{2}_{1}-red_{0}-cyan_auto_{3}.tif".format(
        #         self.trigger_config.stim_1_name, self.trigger_config.trigger_config.stim_2_name, self.output_suffix, self.file_nosuffix
        #     ),
        #     self.trigger_config.stim_1_name,
        #     self.trigger_config.trigger_config.stim_2_name,
        #     self.output_suffix,
        #     ratio_heatmap=ratio_heatmap,
        #     stims_overlap_png=stims_overlap_png,
        #     stims_overlap_tif=stims_overlap_tif,
        #     stims_substracted=stims_substracted,
        #     stims_substracted_diff=stims_substracted_diff,
        # )

        # merger_s1_s2.process_directory()
        # del merger_s1_s2





class TifDerivativeProcess(Helpers):

    def __init__(
        self,
        dir,
        red_name_ending,
        green_name_ending,
        blue_name_ending,
        output_name_ending,
        stim_1_name,
        stim_2_name,
        output_suffix,
        ratio_heatmap=True,
        stims_overlap_png=True,
        stims_overlap_tif=True,
        stims_substracted=True,
        stims_substracted_diff=True,
    ):
        self.dir = dir
        self.red_name_ending = red_name_ending
        self.green_name_ending = green_name_ending
        self.blue_name_ending = blue_name_ending
        self.output_name_ending = output_name_ending
        self.stim_1_name = stim_1_name
        self.stim_2_name = stim_2_name
        self.output_suffix = output_suffix

        self.ratio_heatmap = ratio_heatmap
        self.stims_overlap_png = stims_overlap_png
        self.stims_overlap_tif = stims_overlap_tif
        self.stims_substracted = stims_substracted
        self.stims_substracted_diff = stims_substracted_diff

    def __make_png(self, channels, output_filename):
        for i in range(3 - len(channels)):
            channels.append(np.zeros_like(channels[0]))

        channels = np.array(channels)
        rgb_array_normalized = np.stack(
            [
                (channel - channels.min()) / (channels.max() - channels.min())
                for channel in channels
            ],
            axis=-1,
        )
        rgb_image = Image.fromarray((rgb_array_normalized * 255).astype("uint8"), "RGB")
        try:
            rgb_image.save(output_filename)
        except PermissionError as e:
            pass
            print("PermissionError:", e)

    def __process_derivative_images(
        self,
        red_channel_path,
        green_channel_path,
        blue_channel_path,
        output_path,
    ):
        channels = []

        if red_channel_path:
            red_channel = Image.open(red_channel_path)
            red_array = np.array(red_channel).astype(np.float32)
            channels.append(red_array)

        if green_channel_path:
            green_channel = Image.open(green_channel_path)
            green_array = np.array(green_channel).astype(np.float32)
            channels.append(green_array)

        if blue_channel_path:
            blue_channel = Image.open(blue_channel_path)
            blue_array = np.array(blue_channel)
            channels.append(blue_array)

        # Stack the arrays along the first axis to create a multi-channel image
        multi_channel_array = np.stack(channels, axis=0)

        # Save the multi-channel image in ImageJ format
        if self.stims_overlap_tif:
            try:
                self.save_tiff(
                    output_path + "stims_overlap" + self.output_name_ending,
                    multi_channel_array,
                    metadata={"axes": "CYX", "mode": "composite"},
                )
            except PermissionError as e:
                pass
                print("PermissionError:", e)

        # Save the image as a PNG file
        if self.stims_overlap_png:
            self.__make_png(
                multi_channel_array,
                output_path + "stims_overlap" + self.output_name_ending[:-4] + ".png",
            )

        # make heatmap and save as a PNG file
        if self.ratio_heatmap:
            self.__create_ratio_heatmap(channels, output_path, matplotlib_graph=True)

        # Crerating Stims Substracted images
        if self.stims_substracted:
            self.__create_substracted_image(
                channels[:2:1], output_path, self.stim_1_name, color="darkcyan"
            )
            self.__create_substracted_image(
                channels[1::-1], output_path, self.stim_2_name, color="darkred"
            )

        # Crerating stims Diff image
        if self.stims_substracted_diff:
            self.__create_diff_image(channels[:2], output_path)

    def __create_ratio_heatmap(self, channels, output_path, matplotlib_graph=False):
        # Crerating Heatmap
        # Calculate the ratio image if red and green channels are available
        if len(channels) >= 2:
            image = np.divide(
                channels[1],
                channels[0],
                out=np.zeros_like(channels[0]),
                where=channels[0] != 0,
            )
            image = np.clip(image, 0, np.max(image))

            # Save the ratio image as a single-frame TIFF file with inferno LUT metadata
            output_filename = output_path + "ratio_of_inhibition" + "_heatmap.tif"

            metadata = {
                "axes": "YX",
                "min": 1,
                "max": 4,  # np.max(image) * 0.73
            }

            try:
                # tifffile.imwrite(output_filename, image.astype(np.float32), imagej=True, metadata=metadata)
                self.save_tiff(output_filename, image, metadata=metadata)
                # self.logging(f"Created heatmap image: {output_filename}")

            except PermissionError as e:
                print("PermissionError:", e)

            if matplotlib_graph:
                # Save the ratio image as a heatmap in PNG format using matplotlib
                image = np.clip(image, 1, 4)
                output_filename = output_path + "ratio_of_inhibition" + "_heatmap.png"
                enlarged_shape = (int(image.shape[1] * 1.0), int(image.shape[0] * 1.0))
                fig, ax = plt.subplots(
                    figsize=(enlarged_shape[0] / 100, enlarged_shape[1] / 100), dpi=165
                )

                ax.imshow(
                    image, cmap="inferno", interpolation="bicubic", extent=[0, 1, 0, 1]
                )
                ax.set_position([0.02, 0.02, 0.98, 0.98])
                ax.axis("off")
                fig.patch.set_facecolor("white")
                cbar = plt.colorbar(
                    ax.imshow(
                        image,
                        cmap="inferno",
                        interpolation="bicubic",
                        extent=[0, 1, 0, 1],
                    ),
                    ax=ax,
                )
                cbar.set_label("C to A+C responses ratio", rotation=90, labelpad=5)
                plt.savefig(output_filename, bbox_inches="tight", pad_inches=0)
                plt.close()

    def __create_substracted_image(
        self, channels, output_path, stim, color="k", bins=1024, histogram_xlim=7
    ):
        # Crerating Substracted images
        # Calculate the substraction image if red and green channels are available
        # then set up min pixel values to 0
        if len(channels) >= 2:
            image = np.subtract(channels[1], channels[0])
            image = np.clip(image, 0, np.max(image))

            # Get the name of that directory of experiment day
            two_up_path = os.path.dirname(os.path.dirname(output_path))
            directory_name = os.path.basename(two_up_path)

            # Save the ratio image as a single-frame TIFF file with inferno LUT metadata
            output_filename = (
                output_path
                + "resp_to_stim_"
                + stim
                + "_"
                + self.output_suffix
                + "_"
                + directory_name
                + "_"
                + ".tif"
            )

            metadata = {
                "axes": "YX",
                "min": 0,
                "max": 50,  # np.max(image)
            }

            try:
                # tifffile.imwrite(output_filename, image.astype(np.float32), imagej=True, metadata=metadata)
                self.save_tiff(output_filename, image, metadata=metadata)
                # self.logging(f"Created heatmap image: {output_filename}")

                # Flatten for histogram
                values = image.flatten()
                values = values[values > 0]

                # Build histogram
                # plt.style.use('ggplot')
                fig, ax = plt.subplots(dpi=90, figsize=(3, 3))
                ax.hist(values, bins=bins, color=color)
                ax.set_xlim(0, histogram_xlim)
                ax.set_xlabel("Resp. Intensity \nto Stim " + stim + self.output_suffix)

                # Create histogram filename
                dirname = os.path.dirname(output_filename)
                basename = os.path.basename(output_filename)
                stem, _ = os.path.splitext(basename)
                hist_filename = os.path.join(dirname, stem + "_histogram.png")

                # Save PNG
                fig.savefig(hist_filename, bbox_inches="tight")
                plt.close(fig)

            except PermissionError as e:
                print("PermissionError:", e)

    def __create_diff_image(self, channels, output_path):
        # Crerating Diff image
        # Calculate the substraction image if red and green channels are available
        # then set up min pixel values to 0
        # and merge them into single diff img
        if len(channels) >= 2:
            image1 = np.subtract(channels[0], channels[1])
            image1 = np.clip(image1, 0, np.max(image1))

            image2 = np.subtract(channels[1], channels[0])
            image2 = np.clip(image2, 0, np.max(image2))

            channels = [image1, image2, image2]
            multi_channel_array = np.stack(channels, axis=0)

            # Save the ratio image as a single-frame TIFF file with inferno LUT metadata
            output_filename = (
                output_path
                + "diff_between_"
                + self.stim_2_name
                + "-red_"
                + self.stim_1_name
                + "-cyan_"
                + self.output_suffix
                + ".tif"
            )

            metadata = {
                "axes": "CYX",
                "mode": "composite",
                "min": 0,
                "max": 50,  # np.max(image)
            }

            try:
                # tifffile.imwrite(output_filename, image.astype(np.float32), imagej=True, metadata=metadata)
                self.save_tiff(output_filename, multi_channel_array, metadata=metadata)
                # self.logging(f"Created heatmap image: {output_filename}")

            except PermissionError as e:
                print("PermissionError:", e)

            self.__make_png(
                multi_channel_array,
                output_path
                + "diff_between_"
                + self.stim_2_name
                + "-red_"
                + self.stim_1_name
                + "-cyan_"
                + self.output_suffix
                + ".png",
            )

    def process_directory(
        self,
    ):
        for root, _, files in os.walk(self.dir):
            red_files = [f for f in files if f.endswith(self.red_name_ending)]
            green_files = [f for f in files if f.endswith(self.green_name_ending)]

            for red_file in red_files:
                base_name = red_file[: -len(self.red_name_ending)]
                matching_green_file = base_name + self.green_name_ending
                matching_blue_file = base_name + self.blue_name_ending

                if matching_green_file in green_files:
                    red_path = (
                        os.path.join(root, red_file) if self.red_name_ending else None
                    )
                    green_path = (
                        os.path.join(root, matching_green_file)
                        if self.green_name_ending
                        else None
                    )
                    blue_path = (
                        os.path.join(root, matching_blue_file)
                        if self.blue_name_ending
                        else None
                    )
                    output_path = os.path.join(root, base_name)

                    self.__process_derivative_images(
                        red_path, green_path, blue_path, output_path
                    )
                    # self.logging("\nCreated hyperstack image: {}".format(output_path))

# from suite2p.run_s2p import run_s2p
# import suite2p
# from tifffile import imread
# from tifffile import imwrite
# import matplotlib.pyplot as plt
# from natsort import natsorted
# import numpy as np
# import glob
# folder = '/Users/stemonitis/Desktop/data stabilization/3.27.2026/'
# filename = 'Field_1_exp3.tif'
# path_to_tif = folder+filename
# batch_size = 700
# nonrigid = True
# #%%
# # #%% split tiff into two movies
# # ##in case of two channel imaging

# # data = imread("/Users/stemonitis/Desktop/data stabilization/2025_10_10/Field_6_Dynorphin_application.tif")

# # ch1 = data[:, 0]  # first channel
# # ch2 = data[:, 1]  # second channel

# # imwrite("Field_6_Dynorphin_application_green.tif", ch2)

# #just to get the shape of the data
# data = imread(path_to_tif)
# print('Imaging data of shape:', data.shape)
# n_time,Ly, Lx = data.shape
# # Write binary file
# bin_file = 'movie.bin'

# # mask = np.ones_like(data, dtype=bool)

# # mask[:,400:, 200:] = False  # example: top-left 100x100 px is bad tissue

# # # Replace bad region with the mean brightness of the frame (so it's dark & neutral)
# # mean_val = np.mean(data)
# # data[~mask] = mean_val
# #%%
# with open(bin_file, 'wb') as f:
#     data.astype(np.int16).tofile(f)
# # Read in raw tif corresponding to our example tif
# f_raw = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename='movie.bin')
# # Create a binary file we will write our registered image to 
# f_reg = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename='movie_registered.bin', n_frames = f_raw.shape[0]) # Set registered binary file to have same n_frames
# ops = suite2p.default_ops()
# ops['batch_size'] = batch_size
# ops['reg_tif'] = True
# ops['save_path'] = folder
# ops['nonrigid'] = nonrigid
# ops['block_size'] = [64, 64]
# ops['maxregshiftNR'] = 10

# refImg, rmin, rmax, meanImg, rigid_offsets, \
# nonrigid_offsets, zest, meanImg_chan2, badframes, \
# yrange, xrange = suite2p.registration_wrapper(f_reg, f_raw=f_raw, f_reg_chan2=None, 
#                                                    f_raw_chan2=None, refImg=None, 
#                                                    align_by_chan2=False, ops=ops)

# #stitch tiffs together
# #%%
# # find all registered tiffs (Suite2P names them something like reg_tif/0_*.tif)
# tiff_files = natsorted(glob.glob(folder+'/reg_tif/*.tif'))
# print('Found tiff files:', tiff_files)
# #%%
# # read each tiff and stack into one array
# arrays = [imread(f) for f in tiff_files]
# #%%
# full_movie = np.concatenate(arrays, axis=0)   # stack along time axis
# #%%
# imwrite(folder+'/registered_full.tif', full_movie)
# # Split into groups
# #%%
# groups = [arrays[i:i + 11] for i in range(0, len(arrays), 11)]
# #%%
# for i, g in enumerate(groups):
#     chunk = np.concatenate(g, axis=0)
#     imwrite(folder+f"/chunk_{i+1}.tif", chunk)
# # write back as a single tiff

# #%%
# # assume rigid_offsets from Suite2p registration
# rigid_offsets = np.array(rigid_offsets)
# print('rigid_offsets shape', rigid_offsets.shape)
# #%%
# plt.plot(rigid_offsets[:,3], label='x'); plt.plot(rigid_offsets[:,1], label='y')
# plt.legend(); plt.title('Rigid offsets'); plt.show()
# #%%

# # #test if stable

# # #normcorr if not stable

# # #put chunks together

# # #option to draw a report

# # %%
# #stitch csv together
# import pandas as pd
# import glob
# folder = '/Users/stemonitis/Desktop/data stabilization/2025_10_16/reg_tif_rigid/'
# csv_files = [folder+x for x in ["1.csv", "2.csv", "3.csv"]]
# df = pd.concat(
#     map(pd.read_csv, csv_files), ignore_index=True)
# df.to_csv(folder+"registered_full_L5.csv", index=False)



# # %% plot registered and mean image 
# #%%

# plt.imshow(refImg, cmap='gray', )
# plt.title("Reference Image for Registration");
# #%%

# # Load your TIFF stack
# data = meanImg  # shape (frames, height, width)

# # Define a mask for the dying region
# mask = np.ones_like(data, dtype=bool)

# mask[300:, 300:] = False  # example: top-left 100x100 px is bad tissue

# # Replace bad region with the mean brightness of the frame (so it's dark & neutral)
# mean_val = np.mean(data)
# data[~mask] = mean_val

# plt.imshow(data, cmap='gray')
# plt.title("Mean registered image")
import os
import glob
import numpy as np
from natsort import natsorted
from tifffile import imread, imwrite
import suite2p
import torch


# suite2p's BinaryFile format is int16, so every movie has to be converted to
# it regardless of how it was acquired.
INT16_MIN, INT16_MAX = -32768, 32767
INT16_SPAN = INT16_MAX - INT16_MIN          # 65535
# frames converted at a time, to keep the temporary float copy small
BIN_CHUNK_FRAMES = 256


class Stabilization:
    def __init__(self, run_config, movie_config):
        print("stabilization")
        self.run_config = run_config
        self.movie_config = movie_config
        self.file_path = os.path.join(
            run_config.working_dir,
            movie_config.file_name)
        self.output_path = os.path.join(
            run_config.working_dir,
            movie_config.file_name[:-4] + '_registered.tif'
        )
        # affine mapping used to fit the movie into int16 for suite2p,
        # undone again when the registered movie is written back out
        self.input_dtype = None
        self.int16_offset = 0.0
        self.int16_scale = 1.0

    def run(self):
        if os.path.exists(self.output_path):
            print(f"Registered file already exists, skipping: {self.output_path}")
            return
        print(f"Stabilizing {self.file_path}")
        data, Ly, Lx, n_time = self._read_data_get_datashape()
        f_raw, f_reg = self._write_binary(data, Ly, Lx, n_time)
        del data # free memory
        self._register(f_raw, f_reg)
        self._stitch_output()
        #self._cleanup()
        print(f"Saved to {self.output_path}")

    
    def _read_data_get_datashape(self):
        data = imread(self.file_path)
        print(f"Imaging data of shape: {data.shape}, dtype: {data.dtype}")
        self.input_dtype = data.dtype
        n_time, Ly, Lx = data.shape
        return data, Ly, Lx, n_time

    def _plan_int16_mapping(self, data):
        """Work out how to fit `data` into int16 without wrapping around.

        A plain .astype(np.int16) silently wraps anything outside the int16
        range - a uint16 recording brighter than 32767 arrives at suite2p with
        negative pixels and registration then aligns garbage. Map it instead:

            stored = (value - offset) * scale

        `offset` just re-centres the range and is lossless, which is enough on
        its own for any 16-bit recording. `scale` only comes into play when the
        span is genuinely wider than int16 can hold (32-bit or float input).
        A movie that already fits keeps offset 0 / scale 1, so the common case
        is byte-for-byte what it was before.
        """
        if np.issubdtype(data.dtype, np.integer):
            info = np.iinfo(data.dtype)
            if info.min >= INT16_MIN and info.max <= INT16_MAX:
                return 0.0, 1.0          # uint8/int8/int16: cannot overflow

        lo, hi = float(data.min()), float(data.max())
        if lo >= INT16_MIN and hi <= INT16_MAX:
            return 0.0, 1.0

        span = hi - lo
        if span <= INT16_SPAN:
            # shift only: exact for e.g. uint16 0..65535 -> -32768..32767
            return lo - INT16_MIN, 1.0
        # too wide to shift alone; centre it and squeeze. Lossy, but the
        # alternative is wraparound.
        return (hi + lo) / 2.0, (INT16_SPAN - 1) / span

    def _work_dtype(self):
        """Float type wide enough to hold the input's values exactly.

        float32 only carries 24 bits of mantissa, so anything wider than a
        16-bit integer (or a float64 movie) has to go through float64. With
        float32 the clip bound for e.g. uint32 rounds up to 4294967296.0 and
        overflows back to 0 on the cast - the very wraparound this mapping
        exists to prevent.
        """
        dt = np.dtype(self.input_dtype)
        if np.issubdtype(dt, np.integer):
            return np.float32 if np.iinfo(dt).bits <= 16 else np.float64
        return np.float32 if dt == np.float32 else np.float64

    def _to_int16(self, block):
        """Apply the planned mapping to a block of frames."""
        if (self.int16_offset, self.int16_scale) == (0.0, 1.0):
            return block.astype(np.int16)
        work = self._work_dtype()
        shifted = block.astype(work) - work(self.int16_offset)
        if self.int16_scale != 1.0:
            shifted *= work(self.int16_scale)
        return np.clip(np.round(shifted), INT16_MIN, INT16_MAX).astype(np.int16)

    def _from_int16(self, block):
        """Undo the mapping, back to the input's units and dtype."""
        if (self.int16_offset, self.int16_scale) == (0.0, 1.0):
            return block
        work = self._work_dtype()
        restored = block.astype(work)
        if self.int16_scale != 1.0:
            restored /= work(self.int16_scale)
        restored += work(self.int16_offset)
        if np.issubdtype(self.input_dtype, np.integer):
            info = np.iinfo(self.input_dtype)
            # registration interpolates, so clip before casting back
            restored = np.clip(np.round(restored), info.min, info.max)
        return restored.astype(self.input_dtype)

    def _write_binary(self, data, Ly, Lx, n_time):
        bin_path = os.path.join(self.run_config.working_dir, 'movie.bin')
        reg_bin_path = os.path.join(self.run_config.working_dir, 'movie_registered.bin')

        # deliberately not routed through ssd_guard: run() already returns
        # early when the registered output exists, so these scratch binaries are
        # only written on a genuinely fresh registration - and buffering a
        # multi-GB movie in RAM just to hash it would cost far more than the
        # write it might save. The stitched .tif at the end goes through the
        # guard automatically via the patched tifffile.imwrite.

        self.int16_offset, self.int16_scale = self._plan_int16_mapping(data)
        if (self.int16_offset, self.int16_scale) != (0.0, 1.0):
            print(f"{data.dtype} movie does not fit int16: mapping with "
                  f"offset={self.int16_offset:g}, scale={self.int16_scale:g}"
                  f"{'' if self.int16_scale == 1.0 else ' (lossy)'} for "
                  f"registration; undone when the registered movie is saved")

        # converted per chunk so the temporary float copy stays small
        with open(bin_path, 'wb') as f:
            for i in range(0, n_time, BIN_CHUNK_FRAMES):
                self._to_int16(data[i:i + BIN_CHUNK_FRAMES]).tofile(f)
        # create registered binary file with correct size
        n_bytes = n_time * Ly * Lx * 2  # 2 bytes per int16 pixel
        with open(reg_bin_path, 'wb') as f:
            f.seek(n_bytes - 1)
            f.write(b'\0')  # write one byte at the end to set file size
        # Read in raw tif corresponding to our example tif
        f_raw = suite2p.io.BinaryFile(
            Ly=Ly, Lx=Lx, filename=bin_path)
        # Create a binary file we will write our registered image to 
        f_reg = suite2p.io.BinaryFile(
            Ly=Ly, Lx=Lx, filename=reg_bin_path,
            n_frames=n_time, write=True)  # Set registered binary file to have same n_frames
        return f_raw, f_reg
    
    # def _register(self, f_raw, f_reg):
    #     settings = suite2p.default_settings()
    #     settings['registration']['batch_size'] = self.run_config.stabilization_batch_size
    #     settings['registration']['nonrigid'] = "nonrigid" if self.run_config.stabilization_nonrigid else "rigid"
    #     settings['registration']['block_size'] = [64, 64]
    #     settings['registration']['maxregshiftNR'] = 10
    #     settings['registration']['do_bidiphase'] = False
    #     settings['registration']['bidiphase'] = 0
    #     settings['registration']['reg_tif'] = True
    #     settings['registration']['reg_tif_chan2'] = False

    #     reg_settings = settings['registration']
    #     reg_settings['reg_tif'] = True
    #     reg_settings['reg_tif_chan2'] = False
    #     print(f"Keys before call: {list(reg_settings.keys())}")
    #     print(f"reg_tif value: {reg_settings.get('reg_tif', 'NOT FOUND')}")

       
    #     refImg, rmin, rmax, meanImg, rigid_offsets, \
    #     nonrigid_offsets, zest, meanImg_chan2, badframes, \
    #     yrange, xrange = suite2p.registration_wrapper(f_reg, f_raw=f_raw, f_reg_chan2=None, f_raw_chan2=None, refImg=None, align_by_chan2=False, save_path=self.run_config.working_dir, settings=settings)
    def _register(self, f_raw, f_reg):
        reg_settings = {
            'align_by_chan2': False,
            'nimg_init': 300,
            'maxregshift': 0.1,
            'do_bidiphase': False,
            'bidiphase': 0,
            'batch_size': self.run_config.stabilization_batch_size,
            'nonrigid': self.run_config.stabilization_nonrigid,
            'maxregshiftNR': 10,
            'block_size': [64, 64],
            'smooth_sigma_time': 0,
            'smooth_sigma': 1.15,
            'spatial_taper': 50.0,
            'th_badframes': 1.0,
            'norm_frames': True,
            'snr_thresh': 1.25,
            'subpixel': 10,
            'two_step_registration': False,
            'reg_tif': True,
            'reg_tif_chan2': False,
            'upsample_meanImg': False,
        }
        

        result = suite2p.registration_wrapper(
            f_reg,
            f_raw=f_raw,
            f_reg_chan2=None,
            f_raw_chan2=None,
            refImg=None,
            align_by_chan2=False,
            save_path=self.run_config.working_dir,
            settings=reg_settings,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),  # or 'mps' for Mac gpu support

        )
    def _stitch_output(self):
        tiff_files = natsorted(glob.glob(
            os.path.join(self.run_config.working_dir, 'reg_tif', '*.tif')))
        print(f"Found {len(tiff_files)} registered tiff chunks")
        
        # undo the int16 mapping chunk by chunk, before concatenating, so no
        # second full-size copy of the movie is needed
        arrays = [self._from_int16(imread(f)) for f in tiff_files]
        full_movie = np.concatenate(arrays, axis=0)
        imwrite(self.output_path, full_movie, imagej=True, compression='zlib')
        # plain ASCII: the Windows console is cp1252 and raises
        # UnicodeEncodeError on anything outside it, aborting the run
        print(f"Stitched {len(arrays)} chunks -> {full_movie.shape}")
    
    def _cleanup(self):
        # remove temp binary files
        for f in ['movie.bin', 'movie_registered.bin']:
            path = os.path.join(self.run_config.working_dir, f)
            if os.path.exists(path):
                os.remove(path)
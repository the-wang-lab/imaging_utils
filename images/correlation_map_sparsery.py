# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: suite2p
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils import read_tif_files, post_processing_suite2p_gui

from suite2p.detection.sparsedetect import neuropil_subtraction, square_convolution_2d
from suite2p.detection.utils import temporal_high_pass_filter, standard_deviation_over_time, downsample

from scipy.interpolate import RectBivariateSpline

# %% [markdown]
# # Load data and settings

# %%
# load data
data_folder = Path('/home/nico/local/data/suite2p_data/dopa/')

# tif files to numpy array
tif_files = [
    data_folder / 'file00000_chan0.tif',
    data_folder / 'file00500_chan0.tif',
]
stack = read_tif_files(tif_files)

# suite2p settings file
ops = np.load(data_folder / 'ops.npy', allow_pickle=True).item()

# %% [markdown]
# # Preprocessing steps

# %%
###########
## TRIMMING

# trim movie according to ops file (probably due to motion correction)
x, y = ops['xrange'], ops['yrange']
trimmed = stack[:, y[0]:y[1], x[0]:x[1]]

###########
## BINNING

# dynamically choose bin size
bin_size_frames = ops["nframes"] // ops["nbinned"]
bin_size_tau = ops["tau"] * ops["fs"]

bin_size = np.max([1, bin_size_frames, bin_size_tau])
bin_size = int(np.round(bin_size))

# drop last frames so that number is divisible by bin_size
trimmed = trimmed[: trimmed.shape[0] // bin_size * bin_size]

# bin movie (non-overlapping sliding window)
binned = trimmed.reshape(
    trimmed.shape[0] // bin_size, bin_size, *trimmed.shape[1:]
).mean(axis=1)

############
## FILTERING

# temporal high-pass filter
mov = temporal_high_pass_filter(mov=binned, width=int(ops['high_pass']))

################
## NORMALIZATION

# normalize by standard deviation
mov_std = standard_deviation_over_time(mov, batch_size=ops['batch_size'])
mov_norm = mov / mov_std

###########
## NEUROPIL
# subtract neuropil
mov = neuropil_subtraction(mov=mov_norm, filter_size=ops["spatial_hp_detect"])

# %% [markdown]
# # Downsampling and upsampling

# %%
################
## DOWN-SAMPLING

# meshgrid for downsampled movie
_, y, x = mov.shape
mesh = np.meshgrid(range(x), range(y))
grid = np.array(mesh).astype("float32")

# variables to be downsampled
mov_down, grid_down = mov, grid

# collect downsampled movies and grids
l_mov, l_grid = [], []

# downsample multiple times
for _ in range(5):

    # smooth movie
    smoothed = square_convolution_2d(mov_down, filter_size=3)
    l_mov.append(smoothed)

    # downsample movie
    mov_down = 2 * downsample(mov_down, taper_edge=True)

    # downsample grid
    l_grid.append(grid_down)
    grid_down = downsample(grid_down, taper_edge=False)

# note: len(l_grid) == 5, but len(gxy) == 6 in suite2p, but 6th element is never used

#############
## UPSAMPLING
    
# collect upsampled movies
l_upsampled = []

for mov_down, grid_down in zip(l_mov, l_grid):
    upsample_model = RectBivariateSpline(
        x=grid_down[1, :, 0],
        y=grid_down[0, 0, :],
        z=mov_down.max(axis=0),
        kx=min(3, grid_down.shape[1] - 1),
        ky=min(3, grid_down.shape[2] - 1),
    )
    up = upsample_model(grid[1, :, 0], grid[0, 0, :])
    l_upsampled.append(up)


##################
## CORRELATION MAP
v_corr = np.array(l_upsampled).max(axis=0)


# %% [markdown]
# # Plot resulting images

# %%
# plot
fig, axmat = plt.subplots(ncols=3, nrows=2, figsize=(15, 10))
axmat = axmat.flatten()

for i, img in enumerate(l_upsampled):
    ax = axmat[i]

    img = post_processing_suite2p_gui(img)

    ax.imshow(img, cmap="viridis")
    ax.set_title(f"upsampled from x{i}")
    ax.set_axis_off()
    
ax = axmat[-1]
img = post_processing_suite2p_gui(v_corr)
ax.imshow(img, cmap="viridis")
ax.set_title(f"V_corr (max projection)")
ax.set_axis_off()

# %% [markdown]
# # correlation map
#
# The correlation map `v_corr` is calculated in the first steps of `sparsery`
# (see [here](https://github.com/MouseLand/suite2p/blob/193e7f1f656bfbd1c100eb51411737c80f54ac3c/suite2p/detection/sparsedetect.py#L328C16-L328C16))
#
#
# The steps are
# - bin movie
# - temporal high pass
#   - `high_pass = 100`
# - neuropil subtraction
#   - divide movie by standard deviation in time
#   - `neuropil_high_pass = 25`
# - generate spatially downsampled movies: `movu`
#   - `square_convolution_2d` with `filter_size = 3`
#   - `downsample` 
# - upsample downsampled movies ("spline over scales"): `I`
#   - max project each downsampled movie along time dimension
#   - use scipy's `RectBivariateSpline` to upsample again
# - correlation map
#   - max project `I` across spatial scales equals `v_corr`
#
#
# # location of functions
# ## sparsery
# `sparsery` is in: `detection/sparsedetect.py`
#
# ## Binning movie
# ```
# bin_size = int(
#             max(1, n_frames // ops["nbinned"], np.round(ops["tau"] * ops["fs"])))
# mov = bin_movie(f_reg, bin_size, yrange=yrange, xrange=xrange,
#                 badframes=ops.get("badframes", None))
# ```
#
# `bin_movie` is in:
# `detection/detect.py`
#
# ## temporal high pass 
#
# ```
# utils.temporal_high_pass_filter(mov=mov, width=int(high_pass))
# ```
#
# `temporal_high_pass_filter` is in:
# `etection/utils.py`
#
# ## square convolution
# `square_convolution_2d` is in:
# `detection/sparsedetect.py`
#
# ## downsample
# `downsample` is in:
# `detection/utils.py`

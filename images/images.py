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
#     display_name: imaging_analysis
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt

from utils import read_tif_files, post_processing_mean_image

# %%
tif_files = [
    r"/home/nico/local/data/suite2p_data/reg_ch1/file000_chan0.tif", 
    r"/home/nico/local/data/suite2p_data/reg_ch1/file500_chan0.tif",
]

stack = read_tif_files(tif_files)

# %%
# calculate mean and standard deviation image
img_mean = np.mean(stack, axis=0)
img_std = np.std(stack, axis=0)

# apply same post processing as suite2p GUI
img_mean_proc = post_processing_mean_image(img_mean)
img_std_proc = post_processing_mean_image(img_std)

# %%
# plot
fig, axarr = plt.subplots(ncols=2, figsize=(10, 5))

ax = axarr[0]
ax.imshow(img_mean_proc, cmap="viridis")
ax.set_title("Mean image")

ax = axarr[1]
ax.set_title("Standard deviation image")
ax.imshow(img_std_proc, cmap="viridis")

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

# %% [markdown]
#

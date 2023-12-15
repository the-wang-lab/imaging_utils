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
import numpy as np
import matplotlib.pyplot as plt

from utils import read_tif_files

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

# %%
# plot
fig, axarr = plt.subplots(ncols=2, figsize=(10, 5))
ax = axarr[0]
ax.imshow(img_mean, cmap="viridis")
ax.set_title("Mean image")
ax = axarr[1]
ax.set_title("Standard deviation image")
ax.imshow(img_std, cmap="viridis")

# %%

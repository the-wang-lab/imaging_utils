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

from utils import post_processing_suite2p_gui, get_vcorr_sparsery, load_data_file

# %% [markdown]
# # Load data and settings

# %%
# load data
data_folder = Path(r'Z:\Jingyu\2P_Recording\AC918\AC918-20231017\02\axons_v2.0\suite2p\plane0')

# suite2p settings file
ops = np.load(data_folder / 'ops.npy', allow_pickle=True).item()

# load data
data = load_data_file(ops)

# %% [markdown]
# # Calculate for different bin_size and high_pass

# %%
bin_sizes = [ 10, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
high_pass_orig = ops['high_pass']

vcorrs = []
for bin_size in bin_sizes:
    ops['high_pass'] = high_pass_orig * 30 / bin_size
    vcorr = get_vcorr_sparsery(data, ops, bin_size=bin_size)
    vcorrs.append(vcorr)


# %% [markdown]
# # Plot results

# %%
fig, axmat = plt.subplots(ncols=5, nrows=2, figsize=(30, 13))
vmin, vmax = np.array(vcorrs).min(), np.array(vcorrs).max()

for ax, vcorr, bin_size in zip(axmat.flatten(), vcorrs, bin_sizes):
    img = post_processing_suite2p_gui(vcorr)
    ax.imshow(img, cmap='viridis')
    ax.set_title(f'bin_size = {bin_size}, high_pass = {high_pass_orig * 30 / bin_size:1.1f}')
    ax.set_axis_off()

fig.tight_layout()
fig.savefig('vcorr_bin_size.png')


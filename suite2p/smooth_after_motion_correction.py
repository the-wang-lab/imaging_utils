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
import numpy as np
from utils import rolling_window_bin_file

# %% [markdown]
# # Apply a rolling window after motion correction
# This is a simple script designed to apply a rolling window average
# to the time domain before ROI detection. 
# It is designed to be used within two partial suite2p runs:
#
# 1. register data with suite2p (ROI detection not necessary)
# 2. run this script to apply rolling window average to `data.bin` (old file is saved to `orig.bin`)
# 3. run suite2p again to detect ROIs (registration will be skipped if `data.bin` is present)

# %%
p_ops = r"C:\temp\s2p\test_recording\gui\suite2p\plane0\ops.npy"
ops = np.load(p_ops, allow_pickle=True).item()
rolling_window_bin_file(ops, window_size=10)


import numpy as np
from skimage.draw import polygon, ellipse
from scipy.ndimage.filters import uniform_filter1d

from pathlib import Path
from read_roi import read_roi_zip


def read_imagej_rois(p_roi, res_yx):
    """Convert ROIs exported from ImageJ to boolean array

    ImageJ ROIs can be exported as a zip file. This function
    reads the zip file and converts the ROIs to a boolean array
    with the shape (n_roi, n_y, n_x).

    Only polygon and oval ROIs are supported.

    Parameters
    ----------
    p_roi : path-like
        Path to the ROI zip file
    res_yx : tuple of int
        Y and X resolution of the image in which to place the ROIs

    Returns
    -------
    r : numpy.ndarray
        3D array with shape (n_roi, n_y, n_x) with boolean values
    """

    d_roi = read_roi_zip(p_roi)
    n_y, n_x = res_yx
    # all-false array with shape: (n_roi, n_y,  n_x)
    r = np.zeros((len(d_roi), n_y, n_x)).astype(bool)

    # set rois mask to true
    for i, v in enumerate(d_roi.values()):
        if v["type"] == "polygon":
            x, y = polygon(v["x"], v["y"])

        elif v["type"] == "oval":
            r_x = v["width"] / 2
            r_y = v["height"] / 2
            c_x = v["left"] + r_x
            c_y = v["top"] + r_y
            x, y = ellipse(c_x, c_y, r_x, r_y)

        else:
            print(
                f'WARNING skipping ROI {i+1}, because it has type {v["type"]} not implemented'
            )
            continue

        # ignore out of bounds
        m = (y < n_y) * (x < n_x)
        x, y = x[m], y[m]

        r[i, y, x] = True

    return r


def rolling_window_bin_file(ops, window_size=1):
    '''Smooths binary file with rolling average

    1. Loads binary file ops['reg_file']
    2. Smoothes time axis with uniform_filter1d
    3. Writes output to ops['reg_file'] after renaming to `orig.bin`

    Parameters
    ----------
    ops : dict
        suite2p ops file
    window_size : int, optional
        size for rolling average window, by default 1
    '''
        
    # set up paths
    p_data = Path(ops['reg_file'])
    p_out = p_data.parent / "tmp.bin"

    # set up memory-mapped files
    shape = ops["nframes"], ops["Ly"], ops["Lx"]
    f_in = np.memmap(p_data, mode='r', dtype='int16', shape=shape)
    f_out = np.memmap(p_out, mode='w+', dtype='int16', shape=shape)

    # apply filter
    uniform_filter1d(f_in, window_size, axis=0, output=f_out)

    del f_in, f_out

    # rename data.bin to orig.bin
    p_data.rename(p_data.parent / "orig.bin")
    # rename tmp.bin to data.bin
    p_out.rename(p_data)
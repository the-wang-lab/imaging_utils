
import numpy as np
from skimage.draw import polygon, ellipse

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
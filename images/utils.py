from tifffile import TiffFile
import numpy as np


def read_tif_files(tif_files):
    """Reads tif files and returns a numpy array of shape (n_frames, X, Y)"""

    stack = []
    for tif_file in tif_files:
        with TiffFile(tif_file) as tif:
            # Iteration over pages necessary, 
            # because imread only reads first page for suite2p tif files
            for page in tif.pages:
                stack.append(page.asarray())

    stack = np.array(stack)

    return stack


def post_processing_suite2p_gui(img_orig):
    '''Applies similar post processing to what is done to images in suite2p gui

    Correlation map and max projection include additional post processing steps
    that depend on `ops['xrange']` and `ops['yrange']`. These are not included here.

    Parameters
    ----------
    img_orig : np.ndarray
        Original image

    Returns
    -------
    img_proc : np.ndarray
        Post-processed image
    '''

    # normalize to 1st and 99th percentile
    perc_low, perc_high = np.percentile(img_orig, [1, 99])
    img_proc = (img_orig - perc_low) / (perc_high - perc_low)
    img_proc = np.maximum(0, np.minimum(1, img_proc))

    # convert to uint8
    img_proc *= 255
    img_proc = img_proc.astype(np.uint8)

    return img_proc

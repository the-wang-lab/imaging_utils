from tifffile import TiffFile
import numpy as np

def read_tif_files(tif_files):
    """Reads tif files and returns a numpy array of shape (n_frames, X, Y)"""
    stack = []
    for tif_file in tif_files:
        with TiffFile(tif_file) as tif:
            # Iteration over pages necessary, because imread does not work with suite2p tif files
            for page in tif.pages: 
                stack.append(page.asarray())
    stack = np.array(stack)
    return stack
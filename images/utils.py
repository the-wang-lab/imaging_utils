from tifffile import TiffFile
import numpy as np

from suite2p.detection.sparsedetect import neuropil_subtraction, square_convolution_2d
from suite2p.detection.utils import temporal_high_pass_filter, standard_deviation_over_time, downsample
from scipy.interpolate import RectBivariateSpline


def read_tif_files(tif_files, sort_suite2p=True):
    """Reads tif files and returns a numpy array of shape (n_frames, X, Y)"""

    if sort_suite2p:
        print('INFO Sorting TIF files...')
        tif_ids = np.array([ ''.join(i for i in p.name.split('_')[0] if i.isdigit()) for p in tif_files ]).astype(int)
        isort = np.argsort(tif_ids)        
        tif_files = np.array(tif_files)[isort]

    stack = []
    print('INFO Loading TIF files...')
    for tif_file in tif_files:
        print(f'     {tif_file}')
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



def get_vcorr_sparsery(stack, ops, bin_size=None):
    '''Calculate correlation map `v_corr` as done in `sparsery`

    Uses parameters in `ops`. 
    `bin_size` overwrites suite2p's dynamical bin size calculation.

    Parameters
    ----------
    stack : numpy.ndarray
        Imaging stack of shape (n_frames, Y, X)
    ops : dict
        suite2p ops settings file
    bin_size : int, optional
        If not None, this bin size takes priority, by default None

    Returns
    -------
    v_corr : numpy.ndarray
        Correlation map as defined in `sparsery`
    '''

    x, y = ops['xrange'], ops['yrange']
    trimmed = stack[:, y[0]:y[1], x[0]:x[1]]

    ###########
    ## BINNING

    # dynamically choose bin size
    if not bin_size:
        bin_size_frames = ops["nframes"] // ops["nbinned"]
        bin_size_tau = ops["tau"] * ops["fs"]

        bin_size = np.max([1, bin_size_frames, bin_size_tau])
        bin_size = int(np.round(bin_size))
    print(f'INFO using bin size: {bin_size}')

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

    return v_corr
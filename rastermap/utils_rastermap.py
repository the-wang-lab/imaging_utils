import numpy as np
from scipy.stats import zscore
from rastermap import Rastermap
import matplotlib.pylab as plt
from scipy.ndimage import gaussian_filter1d
from pathlib import Path


def normalize_traces(f):
    '''Do the same normalization of the data that suite2p is using

    Parameters
    ----------
    f : np.array
        Fluorescence trace with shape ``n x t``, where `n` is neurons and `t` the frames

    Returns
    -------
    f_n : np.array
        Normalized traces, same shape as f
    '''
    
    # do preprocessing as suite2p
    f_n = zscore(f, axis=1)
    f_n = np.maximum(-4, np.minimum(8, f_n)) + 4
    f_n /= 12

    return f_n

def get_trace(folder, trace='F-Fneu', only_cell=True, chan2=False):
    '''Load fluorescence data from disk. Multiple selections available.
    Applies same preprocessing as done in suite2p

    Parameters
    ----------
    folder : str or pathlib.Path
        Location of the suite2p .npy files, typically in `suite2p/plane0`
    trace : {'F-Fneu', 'F', 'Fneu'}, optional
        'F-Fneu':
          By default trace is 'F-Fneu'. Trace is calculated as
          ``F - 0.7 * Fneu``.

        'F':
          Trace is cell signal ``F```

        'Fneu':
          Trace is neuropil signal ``Fneu``   
    only_cell : bool, optional
        If True, select only cells as defined in ``iscell.npy``, by default True
    chan2 : bool, optional
        If True, select channel 2 signal instead of channel 1, by default False

    Returns
    -------
    f : np.array
        Fluorescence trace with shape ``n x t``, where `n` is neurons and `t` the frames
    '''

    # ensure Path object
    folder = Path(folder)

    # choose if channel 1 or 2
    F_cell = 'F_chan2.npy' if chan2 else 'F.npy'
    F_neu = 'F_chan2.npy' if chan2 else 'Fneu.npy'


    # choose which trace to use
    if trace == 'F-Fneu':
        f1 = np.load(folder / F_cell)
        f2 = np.load(folder / F_neu)
        f = f1 - 0.7 * f2

    elif trace == 'F':
        f = np.load(folder / F_cell)

    elif trace == 'Fneu':
        f = np.load(folder / F_neu)

    # select only cells
    if only_cell:
        iscell = np.load(folder / 'iscell.npy')
        mask_cell = iscell[:, 0].astype(bool)
        f = f[mask_cell]

    return f

def compute_rastermap(f, rastermap_kw={}):
    '''Computer rastermap and sort neurons acordingly

    Parameters
    ----------
    f : np.array
        Fluorescence trace with shape ``n x t``, where `n` is neurons and `t` the frames
    rastermap_kw : dict, optional
        Dictionary of keywoard arguments passed to ``rastermap.Rastermap``, by default {}

    Returns
    -------
    f_s : np.array
        Rastermap-sorted array, same shape as `f`
    '''

    # call rastermap 
    r = Rastermap(**rastermap_kw)
    r.fit(f)

    # sort neurons accoring to rastermap
    isort = np.argsort(r.embedding[:, 0])
    f_s = f[isort]

    return f_s

def smooth_rastermap(f):
    '''Apply smoothing across neurons as done in suite2p

    Parameters
    ----------
    f : np.array
        Fluorescence trace with shape ``n x t``, where `n` is neurons and `t` the frames

    Returns
    -------
    f_smth : np.array
        Smoothed array
    '''

    # same processing as suite2p
    sigma = np.minimum(8, np.maximum(1, int(f.shape[0] * 0.005)))
    f = gaussian_filter1d(f, sigma, axis=0)

    # normalize
    f = normalize_traces(f)

    return f

def plot_rastermap(f, figsize=(15, 7), title='', path=''):
    '''Plot rastermap with `matplotlib` and optionally save file

    Parameters
    ----------
    f : np.array
        Fluorescence trace with shape ``n x t``, where `n` is neurons and `t` the frames
    figsize : tuple, optional
        X and Y dimentions of the figure, by default (15, 7)
    path : str or pathlib.Path, optional
        Path to save figure. If not '', figure is saved on final plot closed, by default ''
    '''

    # plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.pcolormesh(f, cmap='gray_r', vmin=0.3, vmax=0.7)
    ax.set_xlabel('time [frames]')
    ax.set_ylabel('neurons')
    ax.set_title(title)

    # save figure
    if path:
        fig.savefig(path)
        plt.close(fig)
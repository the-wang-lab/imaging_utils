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
    res : dict
        Dictionary containing interesting parts from rastermap

        f_s : np.array
            Rastermap-sorted array, same shape as `f`
        clu : np.array
            Cluster IDs for each neuron, only returned if return_cluster is True
        cc : np.array
            Sorted cluster correlation matrix
    '''

    # dict to collect results
    res = dict()

    # call rastermap 
    r = Rastermap(**rastermap_kw)
    r.fit(f)

    # sort neurons accoring to rastermap
    isort = np.argsort(r.embedding[:, 0])
    f_s = f[isort]
    res['f_sorted'] = f_s

    # cluster IDs    
    clu = r.embedding_clust[isort]
    res['clu_ids'] = clu

    # sorted cluster correlation matrix
    res['cc_sorted'] = r.cc

    # time traces of cluster centers
    res['clu_traces'] = r.X_nodes

    return res

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

def plot_rastermap(f, clu_ids=None, figsize=(15, 7), title='', path=''):
    '''Plot rastermap with `matplotlib` and optionally save file

    Parameters
    ----------
    f : np.array
        Fluorescence trace with shape ``n x t``, where `n` is neurons and `t` the frames
    clu_ids : np.array, optional
        Cluster ID for each neuron. If not None, additional plots shows cluster composition,
        by default None
    figsize : tuple, optional
        X and Y dimentions of the figure, by default (15, 7)
    title : str, optional
        Title to place above first axis, by default ''
    path : str or pathlib.Path, optional
        Path to save figure. If not '', figure is saved on final plot closed, by default ''
    '''

    # plot
    if clu_ids is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    else:
        fig, axarr = plt.subplots(figsize=figsize, ncols=2, width_ratios=[2, 1])
    
        ax = axarr[1]
        for c in np.unique(clu_ids):
            i = np.argwhere(clu_ids == c).flatten()
            ax.scatter([c for _ in i], i, s=1)
        ax.margins(y=0)
        ax.set_xlabel('cluster')
        ax.set_yticklabels([])


        ax = ax.twinx()
        i, n = np.unique(clu_ids, return_counts=True)
        ax.bar(i, n)
        ax.set_ylim(0, np.max(n) * 10)
        ax.set_yticks([])
        ax.set_yticklabels([])

        ax = axarr[0]


    ax.pcolormesh(f, cmap='gray_r', vmin=0.3, vmax=0.7)
    ax.set_xlabel('time [frames]')
    ax.set_ylabel('neurons')
    ax.set_title(title)

    fig.tight_layout()

    # save figure
    if path:
        fig.savefig(path)
        plt.close(fig)


def plot_clusters(clu, cc, figsize=(15, 6), title='', path=''):
    '''Plot cluster traces and cross-correaltion matrix with `matplotlib` and optionally save file

    Parameters
    ----------
    clu : np.array
        Cluster trace with shape ``n x t``, where `n` is clusters and `t` the frames
    cc : np.array
        sorted cluster cross-correlation matrix with shape ``n x n``, where `n` is clusters
    figsize : tuple, optional
        X and Y dimentions of the figure, by default (15, 6)
    title : str, optional
        Title to place above first axis, by default ''
    path : str or pathlib.Path, optional
        Path to save figure. If not '', figure is saved on final plot closed, by default ''
    '''

    # plot
    fig, axarr = plt.subplots(figsize=figsize, ncols=2)

    ax = axarr[0]
    ax.set_box_aspect(.75)
    ax.pcolormesh(clu, cmap='gray_r', vmin=0.3, vmax=0.7)
    ax.invert_yaxis()

    ax.set_title(title)
    ax.set_xlabel('time [frames]')
    ax.set_ylabel('clusters')

    ax = axarr[1]
    ax.set_box_aspect(1)
    arr = cc - np.eye(cc.shape[0])
    ax.pcolormesh(arr, cmap='viridis')
    ax.invert_yaxis()

    ax.set_xlabel('cluster')

    fig.tight_layout()

    # save figure
    if path:
        fig.savefig(path)
        plt.close(fig)
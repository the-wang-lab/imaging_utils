from pathlib import Path
from tifffile import TiffFile
import numpy as np

import matplotlib.pyplot as plt

def read_tif_stack(p_tif):

    with TiffFile(p_tif) as tif:
        arr = np.array([ p.asarray() for p in tif.pages ])

    return arr

def load_first_last(folder, nframes=500):

    # get sorted list of all tif files
    p_tifs = [ *Path(folder).glob('*.tif') ]
    p_tifs = sorted(p_tifs, key=lambda p: int(''.join(i for i in p.name.split('_')[0] if i.isdigit())))

    # load first tif files
    stack = read_tif_stack(p_tifs.pop(0)) # load first tif file
    while stack.shape[0] < nframes: # check if at least nframes are available
        s = read_tif_stack(p_tifs.pop(0)) # load next tif file
        stack = np.concatenate([stack, s], axis=0) # combine stacks
    
    img_0 = stack[:nframes].mean(axis=0) # mean across frames

    # load last tif files
    stack = read_tif_stack(p_tifs.pop(-1)) # same as above, but backwards
    while stack.shape[0] < nframes:
        s = read_tif_stack(p_tifs.pop(-1))
        stack = np.concatenate([s, stack], axis=0)

    img_f = stack[-nframes:].mean(axis=0)


    return img_0, img_f


def plot_diff(name2imgs, path=''):
    
    n = len(name2imgs)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axmat = plt.subplots(figsize=(ncols*5, nrows*4), ncols=ncols, nrows=nrows, squeeze=False)

    axarr = axmat.flatten()

    # calculate diff and determine vmax
    name2dimg = {}
    vmax = 0
    for name, imgs in name2imgs.items():
        a, b = imgs
        dimg = b - a
        dimg = dimg / np.max([a.max(), b.max()])

        name2dimg[name] = dimg

        vmax = np.max([vmax, np.max(np.abs(dimg))])

    for i, (name, dimg) in enumerate(name2dimg.items()):
        ax = axarr[i]

        pos = ax.imshow(dimg, cmap='seismic', vmin=-vmax, vmax=vmax)
        fig.colorbar(pos, ax=ax)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name)
    
    for i in range(len(name2dimg), len(axarr)):
        ax = axarr[i]
        ax.set_axis_off()

    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)
    

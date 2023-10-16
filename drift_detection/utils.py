from pathlib import Path
from tifffile import TiffFile
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def read_tif_stack(p_tif):

    with TiffFile(p_tif) as tif:
        arr = np.array([ p.asarray() for p in tif.pages ])

    return arr


def read_tif_pages(p_tif, pages):

    with TiffFile(p_tif) as tif:
        arr = np.array([ tif.pages[i].asarray() for i in pages])

    return arr



def load_sample_images(folder, nblocks=10, nframes=50):

    # get sorted list of all tif files
    p_tifs = [ *Path(folder).glob('*.tif') ]
    f_tifs = np.array([ ''.join(i for i in p.name.split('_')[0] if i.isdigit()) for p in p_tifs ]).astype(int)

    # sort files
    isort = np.argsort(f_tifs)
    p_tifs = [ p_tifs[i] for i in isort ]
    f_tifs = f_tifs[isort]

    # get total number of frames
    f_tot = len(TiffFile(p_tifs[-1]).pages) + f_tifs[-1]

    # start frames for each black
    f_b0 = np.linspace(0, f_tot - nframes, nblocks) # nblocks equal-sized blocks
    f_b0 = f_b0.astype(int) # convert to int

    # create dataframe with number of tif files x number of frames per file
    x = np.append(f_tifs, f_tot)    
    df_all = pd.DataFrame([ np.arange(x[i], x[i+1]) for i in range(len(x))[:-1] ])

    # cycle through blocks, collect mean images in `imgs`
    imgs = []
    for f in f_b0:
        # frames per block
        f_b = np.arange(f, f + nframes)

        # select only rows with block frames
        df_b = df_all.isin(f_b)
        m = df_b.any(axis=1)
        df = df_b.loc[m]

        # load relevant frames from relevant files
        stack = []
        for i, ds in df.iterrows():
            p_tif = p_tifs[i]
            pages = np.flatnonzero(ds)
            s = read_tif_pages(p_tif, pages)
            stack.append(s)
        
        stack = np.concatenate(stack, axis=0) # combine all
        img = stack.mean(axis=0)
        imgs.append(img)

    imgs = np.array(imgs)

    return imgs


def plot_diff(name2imgs, path=''):
    
    nrows = len(name2imgs)
    ncols = next(iter(name2imgs.values())).shape[0]

    fig, axmat = plt.subplots(figsize=(ncols*4+4, nrows*3.5), ncols=ncols, nrows=nrows, squeeze=False)

    # calculate diff and determine vmax
    name2dimgs = {}
    vmax = 0
    for name, imgs in name2imgs.items():
        avg = imgs.mean(axis=0)
        dimgs = imgs - avg
        dimgs = dimgs / imgs.max()

        name2dimgs[name] = dimgs

        vmax = np.max([vmax, np.percentile(np.abs(dimgs), 95)])

    for i, (name, dimgs) in enumerate(name2dimgs.items()):
        axarr = axmat[i]
        for ax, dimg in zip(axarr, dimgs):

            pos = ax.imshow(dimg, cmap='seismic', vmin=-vmax, vmax=vmax)
            
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
        
        # left-most panel
        ax = axarr[0]
        ax.set_ylabel(name)

        # right-most panel
        ax = axarr[-1]
        fig.colorbar(pos, ax=ax)

    
    # for i in range(len(name2dimg), len(axarr)):
    #     ax = axarr[i]
    #     ax.set_axis_off()

    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)
    

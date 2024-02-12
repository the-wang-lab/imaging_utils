from pathlib import Path
from tifffile import imread, imwrite, TiffFile
import numpy as np
import mesmerize_core as mc

def split_dual_channel_tif(p_tif, p_out):
    """Split scanimage dual-channel tif file into two separate files

    Writes new files ending with `<filename>_ch1.tif` and `<filename>_ch2.tif`

    May not work for scanimage recordings containing a single file.

    Parameters
    ----------
    p_tif : pathlike
        path to tif file
    p_out : pathlike
        folder to store the split tifs (e.g. tmp folder on local disk)
    """
    
    p_tif = Path(p_tif)
    p_out = Path(p_out)

    tif = TiffFile(p_tif)
    n_pages = len(tif.pages)

    # frame rate from scangimage metadata, rate includes both channels
    fps = tif.scanimage_metadata["FrameData"]["SI.hRoiManager.scanFrameRate"]

    ch1 = imread(p_tif, key=range(0, n_pages, 2))
    ch2 = imread(p_tif, key=range(1, n_pages, 2))

    p_tif_ch1 = p_out / f"{p_tif.stem}_ch1{p_tif.suffix}"
    p_tif_ch2 = p_out / f"{p_tif.stem}_ch2{p_tif.suffix}"

    imwrite(p_tif_ch1, ch1, metadata={"axes": "TYX", "fps": fps})
    imwrite(p_tif_ch2, ch2, metadata={"axes": "TYX", "fps": fps})


def load_bin(p_root):
    p_data = p_root / 'data.bin'
    ops = np.load(p_root / 'ops.npy', allow_pickle=True).item()
    shape = ops["nframes"], ops["Ly"], ops["Lx"]
    data = np.memmap(p_data, mode='r', dtype='int16', shape=shape)
    return data


def run_mesmerize(df):
    for i, row in df.iterrows():
        if row["outputs"] is not None: # item has already been run
            continue # skip
            
        process = row.caiman.run()
        
        # on Windows you MUST reload the batch dataframe after every iteration because it uses the `local` backend.
        # this is unnecessary on Linux & Mac
        # "DummyProcess" is used for local backend so this is automatic
        if process.__class__.__name__ == "DummyProcess":
            df = df.caiman.reload_from_disk()

def create_batch(p_root):
    # set up mesmerize
    mc.set_parent_raw_data_path(p_root)
    batch_path = mc.get_parent_raw_data_path() / "mesmerize-batch/batch.pickle"

    if batch_path.is_file():
        batch_path.unlink()

    # create a new batch
    df = mc.create_batch(batch_path)
    
    return df

def load_batch(p_root):
    # set up mesmerize
    mc.set_parent_raw_data_path(p_root)
    batch_path = mc.get_parent_raw_data_path() / "mesmerize-batch/batch.pickle"

    # to load existing batches use `load_batch()`
    df = mc.load_batch(batch_path)
    
    return df
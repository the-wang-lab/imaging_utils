from pathlib import Path
from tifffile import imread, imwrite, TiffFile
import numpy as np

import caiman as cm

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

def save_data_as_mmap(p_ops):

    ops = np.load(p_ops, allow_pickle=True).item()
    p_data = p_ops.with_name("data.bin")
    p_memmap_base = p_data.with_name('memmap_')

    # set up memory-mapped files
    shape = ops["nframes"], ops["Ly"], ops["Lx"]
    data = np.memmap(p_data, mode='r', dtype='int16', shape=shape, order='C')

    p_memmap = cm.save_memmap(
        filenames=[data],
        base_name=str(p_memmap_base), # TODO test this basename
        order='C', border_to_0=0)
    
    return p_memmap

def load_ref_img(p_ops):
    ops = np.load(p_ops, allow_pickle=True).item()
    return ops['refImg']

def reshape(arr, dims, num_frames):
    return np.reshape(arr.T, [num_frames] + list(dims), order='F')

def write_results_tifs(cnmf_estimates, orig, dims, p_out):

    num_frames = orig.shape[1]

    A, C, b, f = cnmf_estimates.A, cnmf_estimates.C, cnmf_estimates.b, cnmf_estimates.f,

    neural_activity = A.astype(np.float32) @ C.astype(np.float32)
    background = b.astype(np.float32) @ f.astype(np.float32)
    residual = orig.astype(np.float32) - neural_activity - background

    imwrite(p_out / 'neural_activity.tif', reshape(neural_activity, dims, num_frames))   
    imwrite(p_out / 'background.tif', reshape(background, dims, num_frames))
    imwrite(p_out / 'residual.tif', reshape(residual, dims, num_frames))

def save_rois_imagej(cnmf_estimates, dims, perc, p_roi):

    from roifile import ImagejRoi
    from skimage import measure

    p_roi.unlink(missing_ok=True)

    for i in range(cnmf_estimates.A.shape[1]):
        img = np.reshape(cnmf_estimates.A[:, i], dims, order='F').toarray()
        thresh = np.percentile(img[img > 0], perc)
        xy = measure.find_contours(img, thresh)[0]
        roi = ImagejRoi.frompoints(list(zip(xy[:, 1], xy[:, 0])))
        roi.name = str(i)
        roi.tofile(p_roi)

def run_cnmf(images, parameter_dict, p_out):
    
    # start cluster
    _, clu, n_proc = cm.cluster.setup_cluster(backend='multiprocessing', n_processes=None, single_thread=False)

    # convert parameter dict to CNMFParams object
    parameters = cm.source_extraction.cnmf.params.CNMFParams(params_dict=parameter_dict) 
    
    # fit model
    cnmf_model = cm.source_extraction.cnmf.cnmf.CNMF(n_proc, params=parameters, dview=clu)
    cnmf_fit = cnmf_model.fit(images)

    # refit
    cnmf_refit = cnmf_fit.refit(images, dview=clu)

    # save
    cnmf_refit.save(str(p_out / 'cnmf_fit.hdf5'))

    # stop cluster
    cm.stop_server(dview=clu)

def load_cnmf(p_out):
    cnmf_refit = cm.source_extraction.cnmf.cnmf.load_CNMF(str(p_out / 'cnmf_fit.hdf5'))
    return cnmf_refit
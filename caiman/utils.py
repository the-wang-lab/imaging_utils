from pathlib import Path
from tifffile import imread, imwrite, TiffFile

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
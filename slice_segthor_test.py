#!/usr/bin/env python3.7

import pickle
import random
import argparse
import warnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from typing import Callable

import numpy as np
import nibabel as nib
from skimage.io import imsave
from skimage.transform import resize

from tqdm import tqdm as tqdm_  # Assuming tqdm_ is an alias for tqdm

def norm_arr(img: np.ndarray) -> np.ndarray:
    casted = img.astype(np.float32)
    shifted = casted - casted.min()
    norm = shifted / shifted.max()
    res = 255 * norm

    assert 0 == res.min(), res.min()
    assert res.max() == 255, res.max()

    return res.astype(np.uint8)

resize_: Callable = partial(resize, mode="constant", preserve_range=True, anti_aliasing=False)

def slice_patient(id_: str, dest_path: Path, source_path: Path, shape: tuple[int, int]) -> tuple[float, float, float]:
    ct_path = source_path / "test" / f"{id_}.nii.gz"
    if not ct_path.exists():
        raise FileNotFoundError(f"No such file or no access: '{ct_path}'")
    nib_obj = nib.load(str(ct_path))
    ct = np.asarray(nib_obj.dataobj)
    x, y, z = ct.shape
    dx, dy, dz = nib_obj.header.get_zooms()

    # Normalize the CT scan to 0-255
    norm_ct = norm_arr(ct)

    for idz in range(z):
        img_slice = resize_(norm_ct[:, :, idz], shape).astype(np.uint8)
        filename = f"{id_}_{idz:04d}.png"
        save_path = dest_path / "img"
        save_path.mkdir(parents=True, exist_ok=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            imsave(str(save_path / filename), img_slice)

    return dx, dy, dz

def get_test_ids(src_path: Path) -> list[str]:
    test_files = sorted((src_path / 'test').glob('*.nii.gz'))
    # Corrected line:
    test_ids = [p.with_suffix('').stem for p in test_files]
    print(f"Found {len(test_ids)} test ids")
    print(test_ids[:10])
    return test_ids

def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)

    # Ensure destination directory exists
    dest_path.mkdir(parents=True, exist_ok=True)

    test_ids = get_test_ids(src_path)

    resolution_dict: dict[str, tuple[float, float, float]] = {}

    dest_mode: Path = dest_path / "test"
    print(f"Slicing {len(test_ids)} patients to {dest_mode}")

    pfun: Callable = partial(slice_patient,
                             dest_path=dest_mode,
                             source_path=src_path,
                             shape=tuple(args.shape))
    iterator = tqdm_(test_ids)
    if args.process == 1:
        resolutions = list(map(pfun, iterator))
    elif args.process == -1:
        resolutions = Pool().map(pfun, iterator)
    else:
        resolutions = Pool(args.process).map(pfun, iterator)

    for key, val in zip(test_ids, resolutions):
        resolution_dict[key] = val

    with open(dest_path / "spacing.pkl", 'wb') as f:
        pickle.dump(resolution_dict, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved spacing dictionary to {f}")

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Slicing parameters')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)

    parser.add_argument('--shape', type=int, nargs="+", default=[256, 256])
    parser.add_argument('--process', '-p', type=int, default=1,
                        help="The number of cores to use for processing")
    args = parser.parse_args()
    random.seed(0)

    print(args)

    return args

if __name__ == "__main__":
    main(get_args())

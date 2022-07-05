from pathlib import Path
from typing import Tuple

import numpy as np

import open3d as o3d


def keep_only_air(chunk_file: Path):
    """Loads a chunk and keeps only the air blocks."""
    
    chunk = np.load(chunk_file)
    chunk = chunk[chunk[:, 3] == 0]
    # uncomment to keep only below sealevel
    chunk = chunk[chunk[:, 2] < 63]
    
    return chunk[:, :3]


def bound_region(region: np.ndarray, x_bounds: Tuple[float], y_bounds: Tuple[float]):
    """Keeps only the subset of the region that is within the bounds.
    
    Parameters
    ----------
    region : np.ndarray
        An N x d array of the Minecraft blocks.
    x_bounds : tuple
        Constrains the region to lie within (lower_bound, upper_bound) in the x dimension.
    y_bounds : tuple
        Constrains the region to lie within (lower_bound, upper_bound) in the y dimension.

    Returns
    -------
    region : np.ndarray
        A subset of the original region contained within the specified bounds.
    """
    
    # make sure bounds make sense
    assert x_bounds[0] < x_bounds[1]
    assert y_bounds[0] < y_bounds[1]
    
    upper_mask = (region[:, 0] >= x_bounds[0]) & (region[:, 0] <= x_bounds[1])
    lower_mask = (region[:, 1] >= y_bounds[0]) & (region[:, 1] <= y_bounds[1])
    
    return region[lower_mask & upper_mask]
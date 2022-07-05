from glob import glob
from pathlib import Path
from typing import List

import numpy as np
from scipy.spatial import distance_matrix
from ripser import ripser

from tqdm import tqdm

from utils import mc_utils, bunny_utils

rng = np.random.default_rng()


def load_minecraft_region(region_dir: Path=None, 
                          bound_region=True, x_bounds=(-16, 16), y_bounds=(-16,16)):
    """Load a Minecraft region.
    
    Loads a Minecraft region as an np.ndarray.
    Bound the region by a lower_bound + upper_bound sized square to keep
    a subset of the region.
    """

    region_dir = Path('/home/seangrate/Projects/data/minecraft/raw/') if region_dir is None else region_dir

    for i, chunk_file in enumerate(tqdm(glob((region_dir / '*.npy').as_posix()), 
                                        desc='Loading chunks')):
        if i == 0:
            region = mc_utils.keep_only_air(chunk_file)
        else:
            chunk = mc_utils.keep_only_air(chunk_file)
            region = np.vstack([region, chunk])
    
    if bound_region:
        region = mc_utils.bound_region(region, x_bounds, y_bounds)

    return region


def load_mobius_band(num_points: int):
    """Sample points uniformly from the Mobius band."""
    
    u = rng.uniform(0, 2*np.pi, size=(num_points,1))
    v = rng.uniform(-1, 1, size=(num_points,1))

    x = (1 + (v/2) * np.cos(u/2)) * np.cos(u)
    y = (1 + (v/2) * np.cos(u/2)) * np.sin(u)
    z = (v/2) * np.sin(u/2)
    data = np.hstack([x, y, z])

    return data


def load_klein_bottle(num_points: int, immersion: str):
    """Sample points uniformly from the Klein bottle.
    
    https://en.wikipedia.org/wiki/Klein_bottle#Parametrization

    Parameters
    ----------
    num_points : int
        Number of points to sample from surface
    immersion : str
        How to realize the Klein bottle in R^3
    """

    if immersion == 'figure eight':
        v, theta = rng.uniform(0, 2*np.pi, num_points), rng.uniform(0, 2*np.pi, num_points)
        r = 1

        x = (r + np.cos(theta/2) * np.sin(v) - np.sin(theta/2) * np.sin(2*v)) * np.cos(theta)
        y = (r + np.cos(theta/2) * np.sin(v) - np.sin(theta/2) * np.sin(2*v)) * np.sin(theta)
        z = np.sin(theta/2) * np.sin(v) + np.cos(theta/2) * np.cos(2*v)
        data = np.vstack([x, y, z]).T

    elif immersion == 'bottle':
        u, v = rng.uniform(0, 2*np.pi, num_points), rng.uniform(0, 2*np.pi, num_points)

        x = -(2/15) * np.cos(u) * (3*np.cos(v) - 30*np.sin(u) + 90*(np.cos(u)**4)*np.sin(u) - 60*(np.cos(u)**6)*np.sin(u) + 5*np.cos(u)*np.cos(v)*np.sin(u))
        y = -(1/15) * np.sin(u) * (3*np.cos(v) - 3*(np.cos(u)**2)*np.cos(v) - 48*(np.cos(u)**4)*np.cos(v) + 48*(np.cos(u)**6)*np.cos(v) - 60*np.sin(u) + 5*np.cos(u)*np.cos(v)*np.sin(u) - 5*(np.cos(u)**3)*np.cos(v)*np.sin(u) - 80*(np.cos(u)**5)*np.cos(v)*np.sin(u) + 80*(np.cos(u)**7)*np.cos(v)*np.sin(u))
        z = (2/15) * (3 + 5*np.cos(u)*np.sin(u)) * np.sin(v)

        data = np.vstack([x, y, z]).T

    return data


def load_torus(num_points: int, R: float=2, r: float=1):
    """Sample points uniformly from the torus.
    
    https://en.wikipedia.org/wiki/Torus#Geometry

    Parameters
    ----------
    num_points : int
        Number of points to sample from surface
    """

    theta = rng.uniform(0, 2*np.pi, size=(num_points,1))
    phi = rng.uniform(0, 2*np.pi, size=(num_points,1))

    x = (R + r*np.cos(theta)) * np.cos(phi)
    y = (R + r*np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    data = np.hstack([x, y, z])

    return data


def load_bunny(data_dir: Path):
    alignments = bunny_utils.read_conf_file(data_dir / 'bun.conf')
    for i, f in enumerate(tqdm(glob((data_dir / '*.ply').as_posix()),
                            desc='Loading Stanford bunny')):
        if i == 0:
            data = bunny_utils.read_ply(Path(f), alignments)
        else:
            new_data = bunny_utils.read_ply(Path(f), alignments)
            data = np.vstack([data, new_data])

    # remove random point present in raw data
    data = data[data[:, 2] > -0.0693]

    return data


def load_modelnet_files(data_dir: Path=None):
    """Load all of the ModelNet file paths.
    
    Parameters
    ----------
    data_dir : Path
        The directory where all of the ModelNet files are stored.

    Returns
    -------
    modelnet_files : List[Path]
        A list of the ModelNet shape files.
    """
    
    data_dir = Path('/home/seangrate/Projects/data/modelnet40_normal_resampled/') if data_dir is None else data_dir
    modelnet_files = [Path(f) for f in glob((data_dir / '*' / '*.txt').as_posix())]

    return modelnet_files


def load_modelnet_model(data_path: Path):
    """Loads a model from ModelNet.
    
    ModelNet models are stored as .txt files.
    Each row is x, y, z,normal_x, normal_y, normal_z.

    Parameters
    ----------
    data_path : Path
        The path to the file containing the model data.

    Returns
    -------
    data : np.ndarray
        An N x 3 array of the data.
    """

    return np.loadtxt(data_path, delimiter=',')
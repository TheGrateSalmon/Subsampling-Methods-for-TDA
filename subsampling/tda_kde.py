import csv
from glob import glob
from pathlib import Path

# computation
import numpy as np
from numpy.random import default_rng
from sklearn.neighbors import KernelDensity
from ripser import ripser


# visualization
mpl_style = 'fivethirtyeight'
# ax_size = (8,4)
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use(mpl_style)
import open3d as o3d
from tqdm import tqdm


def read_conf_file(conf_file_path):
    """Read the configuration file for the Stanford bunny."""
    
    alignments = {}
    
    with open(conf_file_path, 'r') as f:
        conf_file = csv.DictReader(f, fieldnames=['object_type', 'file_name', 't_x', 't_y', 't_z', 'q_i', 'q_j', 'q_k', 'q_r'], delimiter=' ')
        
        for line in conf_file:
            if line['object_type'] == 'camera':
                continue
            else:
                translation = np.array([line['t_x'], line['t_y'], line['t_z']])
                quaternion = np.array([line['q_i'], line['q_j'], line['q_k'], line['q_r']])
                alignments[line['file_name']] = {'translation': translation.astype(float),
                                                 'quaternion': quaternion.astype(float)}
                
    return alignments


def quaternion_to_rotation(quaternion):
    """Converts a quaternion to its rotation matrix representation.
    
    See https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
    
    Parameters
    ----------
    q : np.array
        Should be of the form [q_i, q_j, q_k, q_r], using the notation from the above link.
    """
    
    # make sure quaternion is unit
    quaternion = quaternion / np.linalg.norm(quaternion, ord=2)
    i, j, k, r = quaternion
    
    # construct rotation matrix
    rot_matrix = np.zeros(shape=(3,3))
    # first row
    rot_matrix[0,0] = 1 - 2*(j**2 + k**2)
    rot_matrix[0,1] = 2*(i*j - k*r)
    rot_matrix[0,2] = 2*(i*k + j*r)
    
    # second row
    rot_matrix[1,0] = 2*(i*j + k*r)
    rot_matrix[1,1] = 1 - 2*(i**2 + k**2)
    rot_matrix[1,2] = 2*(j*k - i*r)
    
    # third row
    rot_matrix[2,0] = 2*(i*k - j*r)
    rot_matrix[2,1] = 2*(j*k + i*r)
    rot_matrix[2,2] = 1 - 2*(i**2 + j**2)
    
    return rot_matrix
   
    
def align_data(data, quaternion, translation):
    """Align the data according to its quaternion and translation vector."""
    
    rot_matrix = quaternion_to_rotation(quaternion)
    
    return (data @ rot_matrix + translation.reshape((1,3)))


def read_ply(file_path, alignments):
    """Read and format data from the Stanford bunny .ply files."""
    
    data = o3d.io.read_point_cloud(file_path.as_posix(), format='ply')
    data = np.asarray(data.points)
    translation, quaternion = alignments[file_path.name]['translation'], alignments[file_path.name]['quaternion']
    data = align_data(data, quaternion, translation)
    
    return data


def vis_bunny(data):
    """Visualize the Stanford bunny as a point cloud."""
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    
    # normalize colors to [0,1]
    # colors = (data[:, 2] - np.min(data[:, 2])) / (np.max(data[:, 2]) - np.min(data[:, 2]))
    # colors = np.vstack([np.zeros(colors.shape), np.zeros(colors.shape), colors]).T
    # pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd], window_name='Stanford Bunny', left=0, top=0)

    
def kde_fit(data, **kwargs):
    step_size = kwargs.get('step_size', 100j)
    
    # instantiate and fit the KDE model
    kde = KernelDensity(bandwidth=0.01, kernel='gaussian')
    kde.fit(data)

    x = data[:, 0]
    y = data[:, 1]
    # score_samples returns the log of the probability density
    xx, yy = np.mgrid[x.min():x.max():step_size, 
                      y.min():y.max():step_size]
    xy_sample = np.vstack([xx.ravel(), yy.ravel()]).T
    logprob = np.exp(kde.score_samples(xy_sample)).reshape(xx.shape)
    
    return xx, yy, logprob


def main():
    rng = default_rng()

    # read data and align
    data_dir = Path('/home/seangrate/Projects/data/bunny/data/')

    alignments = read_conf_file(data_dir / 'bun.conf')
    for i, f in enumerate(tqdm(glob((data_dir / '*.ply').as_posix()),
                            desc='Loading Stanford bunny')):
        if i == 0:
            data = read_ply(Path(f), alignments)
        else:
            new_data = read_ply(Path(f), alignments)
            data = np.vstack([data, new_data])

    # remove random point present in raw data
    data = data[data[:, 2] > -0.0693]

    # resize to fit unit cube
    data = (data - np.amin(data, axis=0).T) / (np.amax(data, axis=0) - np.amin(data, axis=0)).reshape((-1,3))
    
    ################################################################################
    
#     data = rng.normal(size=(10**2,3))
#     data = data / np.linalg.norm(data, ord=2, axis=1).reshape((-1,1))
    
    ################################################################################
    
#     sqrt_r = np.sqrt(rng.uniform(0, 1, size=(10**2,1))) # technically should be drawn from [0,1], not [0,1)
#     theta = rng.uniform(0, 2*np.pi, size=(10**2,1))
#     xs = sqrt_r * np.cos(theta)
#     ys = sqrt_r * np.sin(theta)    
#     data = np.hstack([xs, ys])
    
    ################################################################################

    # Stanford bunny bootstrapping
    boot_num_points = 100
    target_num_points = 1000
    max_dim = 1
    processes = 100

    # target PH
    random_subset = rng.integers(0, data.shape[0], size=target_num_points)
    data_subset = data[random_subset, :]

    target_rips = ripser(data_subset, maxdim=1)
    target_diagrams = target_rips['dgms']

    # do the same thing for smaller number of points, but many times
    for i in tqdm(range(processes)):
        # sample bunny
        random_subset = rng.integers(0, data.shape[0], size=boot_num_points)
        data_subset = data[random_subset, :]
        
        # compute persistent homology
        boot_rips = ripser(data_subset, maxdim=1)
        
        # combine persistence diagrams
        if i == 0:
            boot_diagrams = boot_rips['dgms']
        else:
            for dim, pairs in enumerate(boot_rips['dgms']):
                boot_diagrams[dim] = np.vstack([boot_diagrams[dim], pairs])
                
        # plot KDE
        # fig = plt.figure(figsize=(15,3))
        fig = plt.figure(figsize=(8,8))

        boot_xx, boot_yy, boot_zz = kde_fit(boot_diagrams[1], step_size=1000j)
        plt.pcolormesh(boot_xx, boot_yy, boot_zz)
        plt.scatter(boot_diagrams[1][:, 0], boot_diagrams[1][:, 1], s=2, facecolor='white', alpha=0.25)
        plt.title(f'Bootstrap KDE ($N={i+1}$)')
        plt.xlabel('Birth')
        plt.ylabel('Death')

        save_dir = Path().cwd() / 'plots'
        plt.savefig(save_dir / f'processes-{i+1}.png')
        plt.close(fig)


if __name__ == '__main__':
    main()
import csv

import numpy as np
import open3d as o3d


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
import numpy as np

import open3d as o3d


def vis_modelnet(data_xyz: np.ndarray, data_normals: np.ndarray=None, window_name: str=''):
    """Visualize a ModelNet model as a point cloud."""

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data_xyz)
    
    if data_normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(data_normals)

    o3d.visualization.draw_geometries([pcd], window_name=window_name, left=0, top=0)


def vis_mobius_band(data: np.ndarray):
    """Visualize the Mobius band as a point cloud."""
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)

    o3d.visualization.draw_geometries([pcd], window_name='Mobius band', left=0, top=0)


def vis_klein_bottle(data: np.ndarray):
    """Visualize the Klein bottle as a point cloud."""
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)

    o3d.visualization.draw_geometries([pcd], window_name='Mobius band', left=0, top=0)


def vis_torus(data: np.ndarray):
    """Visualize the torus as a point cloud."""
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)

    o3d.visualization.draw_geometries([pcd], window_name='Torus', left=0, top=0)


def vis_minecraft(data: np.ndarray, as_voxels: bool=False):
    """Visualize a Minecraft region as a point cloud."""

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    
    # normalize colors to [0,1]
    colors = (data[:, 2] - np.min(data[:, 2])) / (np.max(data[:, 2]) - np.min(data[:, 2]))
    colors = np.vstack([np.zeros(colors.shape), np.zeros(colors.shape), colors]).T
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    if as_voxels:
        pcd = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1)

    o3d.visualization.draw_geometries([pcd], window_name='Minecraft Region', left=0, top=0)


def vis_bunny(data: np.ndarray):
    """Visualize the Stanford bunny as a point cloud."""
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    
    # normalize colors to [0,1]
    # colors = (data[:, 2] - np.min(data[:, 2])) / (np.max(data[:, 2]) - np.min(data[:, 2]))
    # colors = np.vstack([np.zeros(colors.shape), np.zeros(colors.shape), colors]).T
    # pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd], window_name='Stanford Bunny', left=0, top=0)
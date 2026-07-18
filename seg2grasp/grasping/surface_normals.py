"""Per-point surface-normal estimation for a point cloud.

The suction planner fits normals analytically per planar patch; this helper
provides dense per-point normals (Open3D) for candidate filtering or debugging,
matching the surface-normal step of the paper's Algorithm 2.
"""
import numpy as np


def estimate_normals(points, radius=40.0, max_nn=50, orient_toward_camera=True):
    """Estimate unit surface normals for an (N, 3) point cloud (mm).

    Args:
        points (np.ndarray): (N, 3) points, or an organized [H, W, 3] cloud
            (flattened internally).
        radius (float): neighborhood search radius (mm).
        max_nn (int): max neighbors used per normal.
        orient_toward_camera (bool): flip normals to face the camera (z < 0).

    Returns:
        np.ndarray: (N, 3) unit normals aligned with the input point order.
    """
    import open3d as o3d

    pts = points.reshape(-1, 3).astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    normals = np.asarray(pcd.normals)
    if orient_toward_camera:
        flip = normals[:, 2] > 0
        normals[flip] = -normals[flip]
    return normals

import os
import numpy as np
import open3d as o3d
from open3d import JVisualizer

app_root = os.path.dirname(os.path.dirname(__file__))


def get_np_filename():
    filename = app_root + "/python/_new_processed/trajectory.ply"
    return filename


def ply_trajectory_np():
    filename = get_np_filename()

    pcd1 = o3d.io.read_point_cloud(filename)

    o3d.visualization.draw_geometries([pcd1],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

ply_trajectory_np()

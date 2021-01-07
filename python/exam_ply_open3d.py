import os
import numpy as np
import open3d as o3d
from open3d import JVisualizer

app_root = os.path.dirname(os.path.dirname(__file__))


def get_filenames(ds_name):
    filename1 = app_root + \
        "/ds_rdii/{}/processed/trajectory.ply".format(ds_name)
    filename2 = app_root + \
        "/ds_rdii/{}/processed/trajectory_rv.ply".format(ds_name)
    return filename1, filename2


def ply_trajectory_sep(ds_name):
    filename1, filename2 = get_filenames(ds_name)

    pcd1 = o3d.io.read_point_cloud(filename1)
    pcd2 = o3d.io.read_point_cloud(filename2)

    o3d.visualization.draw_geometries([pcd1],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

    o3d.visualization.draw_geometries([pcd2],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])


def ply_trajectory_sep_nb(ds_name):
    filename1, filename2 = get_filenames(ds_name)

    fragment = o3d.io.read_point_cloud(filename1)

    visualizer = JVisualizer()
    visualizer.add_geometry(fragment)
    visualizer.show()


#ply_plot_demo()
#ply_trajectory_sep("dan_body1")
ply_trajectory_sep("dan_body1")

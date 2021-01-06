import pyvista as pv
import numpy as np
from pyvista import examples
import os

app_root = os.path.dirname(os.path.dirname(__file__))

def ply_plot_demo():
    filename = examples.planefile
    print(filename)

    mesh = pv.read(filename)
    cpos = mesh.plot()
    print(cpos)

def get_filenames(ds_name):
    filename1 = app_root + "/ds_rdii/{}/processed/trajectory.ply".format(ds_name)
    filename2 = app_root + "/ds_rdii/{}/processed/trajectory_rv.ply".format(ds_name)
    return filename1, filename2

def ply_trajectory_sep(ds_name):
    filename1, filename2 = get_filenames(ds_name)
    mesh1 = pv.read(filename1)
    mesh1.plot()

    if not os.path.isfile(filename2):
        return
    mesh2 = pv.read(filename2)
    mesh2.plot()

def ply_trajectory_cmp(ds_name):
    filename1, filename2 = get_filenames(ds_name)
    mesh1 = pv.read(filename1)
    if not os.path.isfile(filename2):
        mesh1.plot()
        return

    mesh2 = pv.read(filename2)

    p = pv.Plotter(notebook=0, shape=(1, 2), border=False)
    p.subplot(0, 0)
    p.add_text("org", font_size=24)
    p.add_mesh(mesh1, show_edges=True, color=True)
    p.subplot(0, 1)
    p.add_text("rv", font_size=24)
    p.add_mesh(mesh2, color=True, show_edges=True)

    #p.link_views()  # link all the views
    # Set a camera position to all linked views
    #p.camera_position = [(15, 5, 0), (0, 0, 0), (0, 1, 0)]

    p.show(auto_close=False)
    #p.open_gif("linked.gif")

    # Update camera and write a frame for each updated position
    # nframe = 15
    # for i in range(nframe):
    #     p.camera_position = [
    #         (15 * np.cos(i * np.pi / 45.0), 5.0, 15 * np.sin(i * np.pi / 45.0)),
    #         (0, 0, 0),
    #         (0, 1, 0),
    #     ]
    #     p.write_frame()

    # # Close movie and delete object
    # p.close()


#ply_plot_demo()
#ply_trajectory_sep("dan_body1")
ply_trajectory_cmp("ruixuan_body1")

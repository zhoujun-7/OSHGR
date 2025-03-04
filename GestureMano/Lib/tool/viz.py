import open3d as o3d
import numpy as np 
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import matplotlib.pyplot as plt
import trimesh
import pyrender
import torch

def show_pcd_index(p):
    # pcd: (N, 3)
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer()
    vis.show_settings = True

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p)

    vis.add_geometry("MANO", pcd)
    
    for idx in range(0, len(pcd.points)):
        vis.add_3d_label(pcd.points[idx], "{}".format(idx))

    app.add_window(vis)
    app.run()


def show_pcd(p):
    if isinstance(p, list) or isinstance(p, tuple):
        num = len(p)
        assert num > 0, 'get an empty list.'
        color_label = np.arange(num)
        colors = plt.get_cmap("tab20")(color_label / num)

        pcds = []
        for i, pp in enumerate(p):
            type(pp)
            if isinstance(pp, np.ndarray):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pp)
                pcd.paint_uniform_color(colors[i][:3])
                pcds.append(pcd)
            elif isinstance(pp, o3d.geometry.AxisAlignedBoundingBox):
                pp.color = colors[i][:3]
                pcds.append(pp)

        o3d.visualization.draw_geometries(pcds)

    elif isinstance(p, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p)
        o3d.visualization.draw_geometries([pcd])



def display_hand_pyrender(verts, faces):
    vertex_colors = np.array([200, 200, 200, 150])
    tri_mesh = trimesh.Trimesh(verts, faces, vertex_colors=vertex_colors)
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)

    scene = pyrender.Scene()
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
    node_cam = pyrender.Node(camera=cam, matrix=np.eye(4))
    scene.add_node(node_cam)
    scene.set_pose(node_cam, pose=np.eye(4))
    scene.add(mesh)

    pyrender.Viewer(scene, viewport_size=(1280, 768), use_raymond_lighting=True)


def display_hand_o3d(verts, faces):
    if isinstance(verts, torch.Tensor):
        verts = verts.clone().detach().cpu().numpy()

    geometry = o3d.geometry.TriangleMesh()
    geometry.triangles = o3d.utility.Vector3iVector(faces)
    geometry.vertices = o3d.utility.Vector3dVector(verts)
    geometry.compute_vertex_normals()
    vis = o3d.visualization.VisualizerWithKeyCallback()

    vis.create_window(
        window_name="display",
        width=1024,
        height=768,
    )
    vis.add_geometry(geometry)

    def kill(vis):
        exit(0)

    vis.register_key_callback(ord("Q"), kill)

    while True:
        geometry.compute_vertex_normals()
        vis.update_geometry(geometry)
        vis.update_renderer()
        vis.poll_events()


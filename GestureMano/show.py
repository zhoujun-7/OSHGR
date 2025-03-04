# show the final mano pose

import numpy as np
import os
import open3d as o3d
import time
import torch
import trimesh
import pyrender
from Lib.ManoLoader import ManoLoader
from Lib.tool.viz import display_hand_o3d
from Lib import ManoLayer, AManoLayer, AxisLayer
from Lib import display_hand_pyrender
import math
from kornia.geometry.conversions import rotation_matrix_to_angle_axis, angle_axis_to_rotation_matrix

def get_mesh(mano_blob, faces, batch_idx=0):
    vertex_colors = np.array([200, 200, 200, 150])
    joint_colors = np.array([10, 73, 233, 255])

    verts, joints = mano_blob.verts[batch_idx], mano_blob.joints[batch_idx]
    transforms = mano_blob.transforms_abs[batch_idx]
    dt = np.array([0, 0, -270.0])
    dt = dt[np.newaxis, :]

    joints = joints.detach().cpu().numpy()
    verts = verts.detach().cpu().numpy()
    transforms = np.array(transforms.detach().cpu())

    joints = joints * 1000.0 + dt
    verts = verts * 1000.0 + dt
    transforms[:, :3, 3] = transforms[:, :3, 3] * 1000.0 + dt

    tri_mesh = trimesh.Trimesh(verts, faces, vertex_colors=vertex_colors)
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)

    return mesh

def bul_to_abs(bul_pose):
    rt_mat = torch.from_numpy(np.load('Lib/source_mano/bul_to_abs_transmat.npy'))
    bul_pose = bul_pose.reshape(-1, 3)
    bul_pose = angle_axis_to_rotation_matrix(bul_pose).reshape(-1, 16, 3, 3)
    bul_pose[:, 1:] = rt_mat @ bul_pose[:, 1:] @ np.linalg.inv(rt_mat)
    abs_pose = bul_pose
    abs_pose = rotation_matrix_to_angle_axis(bul_pose).reshape(-1, 48)

    return abs_pose

def display_hand_open3d(
    mano_blob,
    faces=None,
    batch_idx=0,
):
    geometry = o3d.geometry.TriangleMesh()
    geometry.triangles = o3d.utility.Vector3iVector(faces)
    verts, joints = mano_blob.verts[batch_idx], mano_blob.joints[batch_idx]

    o_s_T = torch.tensor([[0,0,1],
                          [-1,0,0],
                          [0,-1,0.]])
    verts = (o_s_T@verts.mT).mT

    theta = 10 / 180 * torch.pi
    rot = torch.tensor([[math.cos(theta), math.sin(theta), 0],
                        [-math.sin(theta), math.cos(theta), 0],
                        [0,0,1]])
    verts = (rot@verts.mT).mT

    verts[:, 1] += 0.02
    verts[:, 2] -= 0.02

    geometry.vertices = o3d.utility.Vector3dVector(verts.detach().cpu().numpy())
    geometry.compute_vertex_normals()
    vis = o3d.visualization.VisualizerWithKeyCallback()

    vis.create_window(
        window_name="display",
        width=1024,
        height=768,
    )
    vis.add_geometry(geometry)

    o3d.io.write_triangle_mesh("hand.ply", geometry)

    # axis = np.array([[1,0,0,0],
    #                  [0,1,0,0],
    #                  [0,0,1,0],
    #                  [0,0,0,1]])
    # axis = show_frame(axis, is_show=False)
    # for a in axis:
    #     vis.add_geometry(a)

    def kill(vis):
        exit(0)

    vis.register_key_callback(ord("Q"), kill)

    while True:
        geometry.compute_vertex_normals()
        vis.update_geometry(geometry)
        vis.update_renderer()
        vis.poll_events()


def show_frame(*frame_rt_mat_ls, is_show=True):
    n = len(frame_rt_mat_ls)

    _frame_rt_mat_ls = [m[None] for m in frame_rt_mat_ls]
    frame_rt_mat_np = np.concatenate(_frame_rt_mat_ls, axis=0)
    frame_size = np.abs(frame_rt_mat_np[..., :3, 3]).max() / 10
    frame_size = 0.15
    print("the frame size is ", frame_size)

    frame_ls = []
    for i, rt_mat in enumerate(frame_rt_mat_ls):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=frame_size,
        )
        frame.transform(rt_mat)
        color = np.asarray(frame.vertex_colors)
        color[color == 1] = 100 / 255 + (i + 1 / (n)) * 155 / 255
        frame.vertex_colors = o3d.utility.Vector3dVector(color)
        frame_ls.append(frame)

    # axis_pcd = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)]
    # axis_pcd.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.1 * frame_size))


    if is_show:
        o3d.visualization.draw_geometries([*frame_ls])
    else:
        return [*frame_ls]
    

if __name__ == '__main__':
    mano = ManoLoader('Lib/source_mano/MANO_RIGHT.pkl')
    mano_layer = AManoLayer(use_pca=False, flat_hand_mean=True, mano_assets_path=mano.path)
    axis_layer = AxisLayer()

    pose = torch.zeros([1, 48])
    
    n_ls = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
            'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',

            '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', 
            '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
            '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', 
            '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', 
            '41', '42', '43', '44', '45']


    p = f'./asserts/pose_{n_ls[67]}.npy'
    # p = f'./pose_58.npy'
    
    mano_pose = np.load(p)
    pose[0, 3:] = torch.from_numpy(mano_pose)
    pose = bul_to_abs(pose)
    shape = torch.randn([1, 10])
    mano_output = mano_layer(pose, shape)
    display_hand_open3d(mano_output, mano.faces)







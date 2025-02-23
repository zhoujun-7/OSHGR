from copy import deepcopy
import numpy as np
import torch
import random
import math
import scipy
import cv2


def rt_trans(RT, P):
    """
    Support batch data.
    p.shape = (n, 3) or (B, n, 3)
    """
    assert P.shape[-1] == 3
    if len(P.shape) == 3:
        op = "bni,bji->bnj"
    elif len(P.shape) == 2:
        op = "ni, ji->nj"
    elif len(P.shape) == 4:
        op = "abni,abji->abnj"

    if RT.shape[-1] == 4:
        is_sum = True
    elif RT.shape[-1] == 3:
        is_sum = False

    if isinstance(RT, np.ndarray):
        new_P = np.einsum(op, P, RT[..., :3, :3])
        if is_sum:
            new_P = new_P + RT[..., :3, [3]].swapaxes(-1, -2)
    elif isinstance(RT, torch.Tensor):
        new_P = torch.einsum(op, P, RT[..., :3, :3])
        if is_sum:
            new_P = new_P + RT[..., :3, [3]].swapaxes(-1, -2)
    return new_P


def bmm_np(m1, m2):
    if len(m1.shape) == 4:
        return np.einsum("bijk,bikl->bijl", m1, m2)
    if len(m1.shape) == 3:
        return np.einsum("bij,bjk->bik", m1, m2)
    else:
        return m1 @ m2


setattr(np, "bmm", bmm_np)


def camera_intr_trans(xyz, cam_intr):
    z = xyz[..., [2]]
    uvd = (xyz / z) @ cam_intr.swapaxes(-1, -2)
    uvd[:, [2]] = z
    return uvd


def camera_intr_trans_inv(uvd, cam_intr):
    # uvd: (..., 3)
    # cam_intr: (..., 3, 3)
    if isinstance(cam_intr, torch.Tensor):
        cam_intr_T_inv = torch.linalg.inv(cam_intr.mT)
    elif isinstance(cam_intr, np.ndarray):
        cam_intr_T_inv = np.linalg.inv(cam_intr.swapaxes(-1, -2))

    z = deepcopy(uvd[..., [2]])
    xyz = deepcopy(uvd)
    xyz[..., 2] = 1
    xyz = xyz @ cam_intr_T_inv * z
    return xyz


def depth_to_uvd(depth):
    h, w = depth.shape
    d_max = depth.max()

    X = np.arange(w)
    Y = np.arange(h)
    XX, YY = np.meshgrid(X, Y)
    fore_mask = depth < d_max - 1
    u = XX[fore_mask][:, None]
    v = YY[fore_mask][:, None]
    d = depth[fore_mask][:, None]
    uvd = np.concatenate([u, v, d], axis=1)  # n, 3
    return uvd


def get_rand_view_point(rot):
    R_theta = (random.random() - 0.5) * 2 * math.pi * rot / 180 + math.pi
    R_alpha = (random.random() - 0.5) * 2 * math.pi
    view_point_norm = np.array(
        [
            math.sin(R_theta) * math.sin(R_alpha),
            math.sin(R_theta) * math.cos(R_alpha),
            math.cos(R_theta),
        ]
    )
    return view_point_norm


def vec_dot_product(vec1, vec2):
    # vec: (..., 3)
    if isinstance(vec1, torch.Tensor):
        return torch.sum(vec1 * vec2, dim=-1, keepdim=True)
    elif isinstance(vec1, np.ndarray):
        return np.sum(vec1 * vec2, axis=-1, keepdims=True)


def get_vecs_angle(vec1, vec2):
    v1 = norm_vec(vec1)
    v2 = norm_vec(vec2)
    if isinstance(vec1, torch.Tensor):
        return torch.arccos(vec_dot_product(v1, v2)) / torch.pi * 180
    elif isinstance(vec1, np.ndarray):
        return np.arccos(vec_dot_product(v1, v2)) / np.pi * 180


def get_vec_length(vec):
    if isinstance(vec, torch.Tensor):
        return torch.sqrt(torch.sum(torch.pow(vec, 2), dim=-1, keepdim=True))
    elif isinstance(vec, np.ndarray):
        return np.sqrt(np.sum(vec**2, axis=-1, keepdims=True))


def norm_vec(vec):
    # vec: (..., 3)
    v = vec / get_vec_length(vec)
    return v


def vec_z_to_frame(vec_z):
    z_vec = vec_z
    x_vec = np.zeros_like(z_vec)
    x_vec[..., -1] = 1
    y_vec = np.cross(z_vec, x_vec, axis=-1)
    y_vec = norm_vec(y_vec)
    x_vec = np.cross(y_vec, z_vec, axis=-1)
    x_vec = norm_vec(x_vec)
    frame = np.concatenate([x_vec[..., None], y_vec[..., None], z_vec[..., None]], axis=-1)
    return frame


def rotate_z_axis_randomly(r_mat):
    if len(r_mat.shape) == 2:
        n = 1
    elif len(r_mat.shape) == 3:
        n = r_mat.shape[0]

    theta = np.random.rand(n) * 2 * np.pi
    rot_along_z_mat = np.zeros_like(r_mat)
    rot_along_z_mat[..., 0, 0] = np.cos(theta)
    rot_along_z_mat[..., 0, 1] = np.sin(theta)
    rot_along_z_mat[..., 1, 0] = -np.sin(theta)
    rot_along_z_mat[..., 1, 1] = np.cos(theta)
    rot_along_z_mat[..., 2, 2] = 1
    return r_mat @ rot_along_z_mat


def uvd_to_depth(uvd, image_size=(640, 480), bg=0):
    image_size = image_size[1], image_size[0]
    if isinstance(uvd, torch.Tensor):
        depth = torch.ones(image_size).to(uvd) * bg
        u = uvd[:, 0].to(torch.int64)
        v = uvd[:, 1].to(torch.int64)
        d = uvd[:, 2]
    elif isinstance(uvd, np.ndarray):
        depth = np.ones(image_size) * bg
        u = uvd[:, 0].astype(np.int64)
        v = uvd[:, 1].astype(np.int64)
        d = uvd[:, 2]

    m1 = u > 0
    m2 = u < image_size[1] - 1
    m3 = v > 0
    m4 = v < image_size[0] - 1

    m = m1 * m2 * m3 * m4

    u = u[m]
    v = v[m]
    d = d[m]

    depth[v, u] = d

    return depth


def trans_multiview(image, ct, cam_intr, res, pose_uvd=None, rot=30, distance=(400, 1000)):
    # image: (h, w)
    # ct: (1, 3)
    # cam_intr: (3, 3)
    # pose_uvd: (21, 3)
    h, w = image.shape
    image = image.copy() + ct[0, 2]
    uvd = depth_to_uvd(image)

    cam_intr = cam_intr.copy()
    cam_intr[0, 2] = w / 2
    cam_intr[1, 2] = h / 2

    chP = camera_intr_trans_inv(uvd, cam_intr)
    chT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, ct[0, 2]], [0, 0, 0, 1]])

    a = random.random()
    D = a * distance[0] + (1 - a) * distance[1]
    sample_vec_z = get_rand_view_point(rot)
    cam_r_mat = vec_z_to_frame(sample_vec_z)
    hnT_ = rotate_z_axis_randomly(cam_r_mat)
    hnT = np.eye(4)
    hnT[:3, :3] = hnT_
    hnT[:, [1, 2]] *= -1
    hnT[:3, 3] = sample_vec_z * D

    ncT = np.linalg.inv(hnT) @ np.linalg.inv(chT)
    nhP = rt_trans(chP, ncT)
    new_ct = rt_trans(np.array([[0, 0, ct[0, 2]]]), ncT)

    ul = new_ct.copy()
    ul[:, :2] -= 150
    br = new_ct.copy()
    br[:, :2] += 150
    ul_uv = camera_intr_trans(ul, cam_intr)
    br_uv = camera_intr_trans(br, cam_intr)
    new_w = br_uv[0, 0] - ul_uv[0, 0]
    new_h = br_uv[0, 1] - ul_uv[0, 1]
    new_cam_intr = cam_intr.copy()
    new_cam_intr[0, 2] = new_w / 2
    new_cam_intr[1, 2] = new_h / 2

    new_uvd = camera_intr_trans(nhP, new_cam_intr)
    new_uvd[:, 2] -= new_ct[0, 2]
    new_img = uvd_to_depth(new_uvd, (int(new_w), int(new_h)), bg=150)
    back_mask = np.logical_or(new_img > 150, new_img < -150)
    new_img[back_mask] = 150

    if pose_uvd is not None:
        pose_uvd = pose_uvd.copy()
        pose_uvd[:, 2] += ct[0, 2]
        pose_xyz = camera_intr_trans_inv(pose_uvd, cam_intr)
        new_pose_xyz = rt_trans(pose_xyz, ncT)
        new_pose_uvd = camera_intr_trans(new_pose_xyz, new_cam_intr)
        new_pose_uvd[:, 2] -= new_ct[0, 2]
        return new_img, new_pose_uvd
    return new_img


def keep_ratio_resize(image: np.ndarray, max_resolution, kpt=None, interpolation=cv2.INTER_NEAREST):
    h, w = image.shape
    le = max(h, w)
    ratio = max_resolution / le
    image_re = cv2.resize(image, (int(w * ratio), int(h * ratio)), interpolation=interpolation)

    if kpt is None:
        return image_re
    else:
        kpt_re = kpt.copy()
        kpt_re[:, :2] *= np.array([int(w * ratio) / w, int(h * ratio) / h])
        return image_re, kpt_re

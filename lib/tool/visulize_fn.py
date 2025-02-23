import math
import cv2
import torch
import numpy as np
import open3d as o3d
import math
import open3d.visualization.gui as gui
import matplotlib.pyplot as plt


# * >>>>>>>>>>>>> 2D visual functions <<<<<<<<<<<<<<
def show_img(*d):
    row = math.ceil(math.sqrt(len(d)))
    col = math.ceil(len(d) / row)
    for c in range(col):
        for r in range(row):
            if c * row + r == len(d):
                break
            plt.subplot(col, row, c * row + r + 1)
            plt.imshow(d[c * row + r])
    plt.show()


def get_color_bar(n, bar_type="winter"):
    color_bar = plt.get_cmap(bar_type)(np.arange(n) / n)[:, :3] * 256
    color_bar = color_bar.astype(np.uint8)
    return color_bar


def depth_to_rgb(image, d_min=None, d_max=None, fake_color=False):
    # bina: (H, W)
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    else:
        image = image.copy()

    H, W = image.shape[0], image.shape[1]
    d_min = image.min() if d_min is None else d_min
    d_max = image.max() if d_max is None else d_max

    rgb = np.zeros([H, W, 3], dtype=np.uint8)
    image = (image - d_min) / (d_max - d_min + 1e-5) * 255
    image = image.astype(np.uint8)

    if fake_color:
        color_bar = get_color_bar(256)
        rgb = color_bar[image]
    else:
        image = image[..., None]
        rgb = rgb + image

    return rgb


def draw_pts_on_image(image, pts, color=None):
    # image: (H, W) or (H, W, 3)
    # pts: (N, 2) or (N, 3)

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    else:
        image = image.copy()

    if len(image.shape) == 2:
        image = depth_to_rgb(image)

    color_bar = get_color_bar(pts.shape[0])
    for i, pt in enumerate(pts):
        pt = int(pt[0]), int(pt[1])
        if color is None:
            color_ = color_bar[i].tolist()
        else:
            color_ = color
        cv2.circle(image, pt, radius=2, color=color_, thickness=2)

    return image


HANDS19_ORDER = [
    [0, 1, 6, 7, 8],
    [0, 2, 9, 10, 11],
    [0, 3, 12, 13, 14],
    [0, 4, 15, 16, 17],
    [0, 5, 18, 19, 20],
]
MANOPTH_ORDER = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
]
MANO_ORDER = [
    [0, 13, 14, 15, 16],
    [0, 1, 2, 3, 17],
    [0, 4, 5, 6, 18],
    [0, 10, 11, 12, 19],
    [0, 7, 8, 9, 20],
]
ICVL_ORD = [
    [0, 1, 2, 3],  # T
    [0, 4, 5, 6],  # I
    [0, 7, 8, 9],  # M
    [0, 10, 11, 12],  # R
    [0, 13, 14, 15],  # P
]
NATURAL_ORDER = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
]
kpt_color = [
    [0, 127, 255],
    [255, 0, 0],
    [255, 165, 0],
    [0, 255, 0],
    [0, 0, 255],
    [139, 0, 255],
]


def draw_skeleton(image: torch.Tensor or np.ndarray, kpts: np.ndarray, k_type="manopth"):
    assert k_type in [
        "hands19",
        "mano",
        "manopth",
        "icvl",
        "natural",
    ], "unsupport keypoint type."

    if k_type == "hands19":
        kpt_ord = HANDS19_ORDER
    elif k_type == "mano":
        kpt_ord = MANO_ORDER
    elif k_type == "manopth":
        kpt_ord = MANOPTH_ORDER
    elif k_type == "icvl":
        kpt_ord = ICVL_ORD
    elif k_type == "natural":
        kpt_ord = NATURAL_ORDER

    if len(image.shape) == 2:
        img = depth_to_rgb(image)
    else:
        img = image.copy()

    for i in range(6):
        pt = kpts[kpt_ord[i]]
        if len(pt.shape) != 2:
            pw = (int(pt[0]), int(pt[1]))
        else:
            for j in range(len(pt)):
                p = (int(pt[j][0]), int(pt[j][1]))
                if j == 0:
                    dst = pw
                else:
                    dst = (int(pt[j - 1][0]), int(pt[j - 1][1]))
                cv2.line(img, p, dst, color=kpt_color[i], thickness=1)
                cv2.circle(img, p, radius=2, color=kpt_color[i], thickness=2)
    cv2.circle(img, pw, radius=2, color=kpt_color[i], thickness=2)
    return img


def draw_bbox_on_image(image, bbox):
    # image: (H, W) or (H, W, 3)
    # bbox: (N, 4)

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    else:
        image = image.copy()

    if len(image.shape) == 2:
        image = depth_to_rgb(image)

    color_bar = get_color_bar(bbox.shape[0])
    for i, b in enumerate(bbox):
        b1 = int(b[0]), int(b[1])
        b2 = int(b[2]), int(b[3])
        cv2.rectangle(image, b1, b2, color=color_bar[i].tolist(), thickness=2)

    return image


def batch_imgs_to_big_img(batch_img, row):
    # batch_img: (B, 3, H, W)
    b, c, h, w = batch_img.shape
    col = math.ceil(b / row)
    W = row * w
    H = col * h

    out_img = np.zeros([H, W, 3], dtype=np.uint8)
    for bb in range(b):
        r = bb % row
        c = bb // row
        bb_img = batch_img[bb]
        bb_img = bb_img.transpose(1, 2, 0)
        bb_img = (bb_img - bb_img.min()) / (bb_img.max() - bb_img.min() + 1e-5) * 255
        bb_img = bb_img.astype(np.uint8)
        out_img[c * h : (c + 1) * h, r * w : (r + 1) * w] += bb_img

    out_img = out_img.astype(np.uint8)
    return out_img


def detach_and_draw_batch_img(batch_img, kpt=None, k_type="hands19"):
    batch_img = batch_img.detach().cpu().numpy()
    if kpt is not None:
        kpt = kpt.detach().cpu().numpy()

    img_ls = []
    for i, img in enumerate(batch_img):
        rgb = depth_to_rgb(img[0])
        if kpt is not None:
            rgb = draw_skeleton(rgb, kpt[i], k_type=k_type)
        img_ls.append(rgb)

    return img_ls


def save_dealed_batch(img_ls_ls, width=8, save_path=None):

    L = len(img_ls_ls)
    B = len(img_ls_ls[0])
    h, w, _ = img_ls_ls[0][0].shape
    W = w * width
    H = math.ceil(L * B / width) * h

    img_out = np.zeros([H, W, 3], dtype=np.uint8)

    img_ls = []
    for b in range(B):
        for l in range(L):
            img_ls.append(img_ls_ls[l][b])

    n = 0
    for i in range(0, H, h):
        for j in range(0, W, w):
            img_out[i : i + h, j : j + w] = img_ls[n]
            n += 1
            if n == len(img_ls):
                break

    if save_path is not None:
        cv2.imwrite(save_path, img_out)
    else:
        return img_out


# * >>>>>>>>>>>>> 3D visual functions <<<<<<<<<<<<<<


def dep_to_pcd(img, d_min=200, d_max=1500):
    h, w = img.shape
    Y = np.arange(h)
    X = np.arange(w)
    XX, YY = np.meshgrid(X, Y)
    UVD = np.concatenate([XX[..., None], YY[..., None], img[..., None]], axis=-1)
    m1 = img > d_min
    m2 = img < d_max
    m = np.logical_and(m1, m2)
    pcd = UVD[m]
    return pcd


def show_vertices(verts):
    app = gui.Application.instance
    app.initialize()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    # o3d.visualization.draw_geometries([pcd])

    vis = o3d.visualization.O3DVisualizer()
    vis.show_settings = True
    vis.add_geometry("Points", pcd)
    for idx in range(0, len(pcd.points)):
        vis.add_3d_label(pcd.points[idx], "{}".format(idx))
    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()


def show_pcd(*pcds, frame_size=None, frame_loc=[0, 0, 0], is_show=True):
    color_bar = get_color_bar(len(pcds))
    color_bar = color_bar.astype(np.float64) / 256

    o3d_obj_ls = []
    for i, p in enumerate(pcds):
        if isinstance(p, torch.Tensor):
            pcd_np = p.detach().cpu().numpy()
        else:
            pcd_np = p

        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np)
        pcd_o3d.paint_uniform_color(color_bar[i])

        o3d_obj_ls.append(pcd_o3d)

    if is_show:
        if frame_size is None:
            frame_size = np.abs(pcds[0]).mean()
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=frame_loc)
        o3d_obj_ls.append(axis_pcd)
        o3d.visualization.draw_geometries(o3d_obj_ls)
    else:
        return o3d_obj_ls


def show_vec(*vec_rt_mat_ls):
    color_bar = get_color_bar(len(vec_rt_mat_ls))
    color_bar = color_bar.astype(np.float64) / 256

    _vec_rt_mat_ls = [m[None] for m in vec_rt_mat_ls]
    vec_rt_mat_np = np.concatenate(_vec_rt_mat_ls, axis=0)
    vec_size = np.abs(vec_rt_mat_np[..., :3, 3]).max() - np.abs(vec_rt_mat_np[..., :3, 3]).min()

    for i, rt_mat in enumerate(vec_rt_mat_ls):
        vec = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.3 * 0.02 * vec_size,
            cylinder_height=0.3 * 0.5 * vec_size,
            cone_radius=0.3 * 0.04 * vec_size,
            cone_height=0.3 * 0.2 * vec_size,
        )
        vec.transform(rt_mat)
        vec.paint_uniform_color(color_bar[i])

    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([vec, axis_pcd])


def show_frame(*frame_rt_mat_ls, is_show=True):
    n = len(frame_rt_mat_ls)

    _frame_rt_mat_ls = [m[None] for m in frame_rt_mat_ls]
    frame_rt_mat_np = np.concatenate(_frame_rt_mat_ls, axis=0)
    frame_size = np.abs(frame_rt_mat_np[..., :3, 3]).max() / 10
    frame_size = 0.1 if frame_size < 0 else frame_size
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

    axis_pcd = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)]
    axis_pcd.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.1 * frame_size))

    if is_show:
        o3d.visualization.draw_geometries([*frame_ls, *axis_pcd])
    else:
        return [*frame_ls, *axis_pcd]


def show_o3d(*o3d_obj):
    o3d.visualization.draw_geometries(o3d_obj)


def draw_skeleton(image: torch.Tensor or np.ndarray, kpts: np.ndarray, k_type="manopth"):
    assert k_type in [
        "hands19",
        "mano",
        "manopth",
        "icvl",
        "natural",
    ], "unsupport keypoint type."

    if k_type == "hands19":
        kpt_ord = HANDS19_ORDER
    elif k_type == "mano":
        kpt_ord = MANO_ORDER
    elif k_type == "manopth":
        kpt_ord = MANOPTH_ORDER
    elif k_type == "icvl":
        kpt_ord = ICVL_ORD
    elif k_type == "natural":
        kpt_ord = NATURAL_ORDER

    if len(image.shape) == 2:
        img = depth_to_rgb(image)
    else:
        img = image.copy()

    kpts = kpts.copy()
    kpts = np.nan_to_num(kpts)

    for i in range(5):
        pt = kpts[kpt_ord[i]]
        for j in range(len(pt)):
            src = (int(pt[j][0]), int(pt[j][1]))
            if j < len(pt) - 1:
                dst = (int(pt[j + 1][0]), int(pt[j + 1][1]))
                cv2.line(img, src, dst, color=kpt_color[i], thickness=1)
            cv2.circle(img, src, radius=2, color=kpt_color[i], thickness=2)
    # cv2.circle(img, pw, radius=2, color=kpt_color[i], thickness=2)
    return img


def show_skeleton_o3d(kpt, k_type="manopth", is_show=True):
    assert k_type in [
        "hands19",
        "mano",
        "manopth",
        "icvl",
        "natural",
    ], "unsupport keypoint type."

    if k_type == "hands19":
        kpt_ord = HANDS19_ORDER
    elif k_type == "mano":
        kpt_ord = MANO_ORDER
    elif k_type == "manopth":
        kpt_ord = MANOPTH_ORDER
    elif k_type == "icvl":
        kpt_ord = ICVL_ORD
    elif k_type == "natural":
        kpt_ord = NATURAL_ORDER

    global kpt_color
    kpt_color_ = np.array(kpt_color) / 255
    kpt_color_ = kpt_color_.tolist()

    point_set = []
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(kpt)

    line = []
    color = []
    for j, f in enumerate(kpt_ord):
        for i, k in enumerate(f):
            if i != len(f) - 1:
                line.append([f[i], f[i + 1]])
                color.append(kpt_color_[j])
            pt = o3d.geometry.TriangleMesh.create_sphere()
            pt.vertices = o3d.utility.Vector3dVector(np.asarray(pt.vertices) + kpt[f[i]])
            pt.paint_uniform_color(kpt_color_[j])
            point_set.append(pt)

    line_set.lines = o3d.utility.Vector2iVector(line)
    line_set.colors = o3d.utility.Vector3dVector(color)

    if is_show:
        o3d.visualization.draw_geometries([line_set, *point_set])
    else:
        return [line_set, *point_set]


def draw_image_pose(image_pose, image_size=(224, 224), k_type="manopth", background="white"):
    dummy_image = torch.ones([image_pose.shape[0], 3, image_size[0], image_size[0]], dtype=torch.float32)
    if background == "black":
        dummy_image[..., 0] = 1
    elif background == "white":
        dummy_image[..., 0] = 0
    elif isinstance(background, float):
        dummy_image[:] = background
        dummy_image[..., 0, 0] = 0.0
        dummy_image[..., -1, -1] = 1.0

    return detach_and_draw_batch_img(dummy_image, image_pose, k_type=k_type)

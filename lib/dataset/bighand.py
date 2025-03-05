import os
import cv2
import numpy as np
import torch
from .basic import Basic
from ..tool.vision_fn import dep_to_3channel_inv
from ..tool.hand_fn import kpt_to_angle, HANDS19_TO_MANO


class BigHand(Basic):
    def __init__(
        self,
        DATA_DIR="data/BigHand/bighand_crop",
        RESOLUTION=(224, 224),
        DEPTH_NORMALIZE=150,
        SHIFT_U=(-30, 30),
        SHIFT_V=(-30, 30),
        SHIFT_D=(-30, 30),
        ROTATION=(-180, 180),
        SCALE=(0.8, 1.3),
        GAUSS_NOISE_PROBABILITY=0.5,
        GAUSS_NOISE_MU=(-3, 3),
        GAUSS_NOISE_SIGMA=(3, 30),
        ERASER_PROBABILITY=0.5,
        ERASE_RATIO=(0.02, 0.4),
        ERASE_PATH_RATIO=0.3,
        ERASE_MU=(-3, 3),
        ERASE_SIGMA=(30, 80),
        SMOOTH_PROBABILITY=0.5,
        SMOOTH_KERNEL=(2, 5),
        IS_AUGMENT=True,
        **kwargs,
    ) -> None:
        self.DATA_DIR = DATA_DIR
        self.cam_intr = np.array(
            [
                [475.065948, 0, 112],  # 315.944855
                [0, 475.065857, 112],  # 315.944855
                [0, 0, 1],
            ]
        )

        super().__init__(
            RESOLUTION,
            DEPTH_NORMALIZE,
            SHIFT_U,
            SHIFT_V,
            SHIFT_D,
            ROTATION,
            SCALE,
            GAUSS_NOISE_PROBABILITY,
            GAUSS_NOISE_MU,
            GAUSS_NOISE_SIGMA,
            ERASER_PROBABILITY,
            ERASE_RATIO,
            ERASE_PATH_RATIO,
            ERASE_MU,
            ERASE_SIGMA,
            SMOOTH_PROBABILITY,
            SMOOTH_KERNEL,
            IS_AUGMENT,
        )

    def load_data_ls(self):
        ann = np.load(os.path.join(self.DATA_DIR, "bighand_pose_v2.npy"), allow_pickle=True).item()
        image_ls = ann["image"]
        image_ls_ = []
        for i, _ in enumerate(image_ls):
            path = os.path.join(self.DATA_DIR, "image", image_ls[i])
            if os.path.exists(path):
                image_ls_.append(image_ls[i])
        pose_ls = [f[:-3] + "npy" for f in image_ls_]
        label = [7777] * len(image_ls_)
        return image_ls_, label, pose_ls, None

    def load_data(self, index):
        image = cv2.imread(os.path.join(self.DATA_DIR, "image", self.image_path_ls[index]))
        image = dep_to_3channel_inv(image) - 150
        pose = np.load(os.path.join(self.DATA_DIR, "label", self.pose_ls[index]))
        pose = pose.astype(np.float32)
        pose = pose[HANDS19_TO_MANO]
        angle = kpt_to_angle(torch.from_numpy(pose), k_type="manopth").numpy()
        angle = angle.astype(np.float32)
        label = 7777
        return image, label, pose, angle

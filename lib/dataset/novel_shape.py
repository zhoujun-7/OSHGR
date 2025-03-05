import os
import cv2
import numpy as np
from natsort import natsorted
import glob
from .basic import Basic


class ShapeDataset(Basic):
    def __init__(
        self,
        INCR_SHAPE="Half-finger",
        DATA_DIR="data/HGR68/OSHGR",
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
        IS_AUGMENT=False,
        **kwargs,
    ):

        self.DATA_DIR = DATA_DIR
        self.INCR_SHAPE = INCR_SHAPE

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
        img_ls = natsorted(glob.glob(os.path.join(self.DATA_DIR, self.INCR_SHAPE, "*/*.png")))

        image_ls = []
        label_ls = []
        for img_path in img_ls:
            d = img_path.split("/")
            label = int(d[-2])
            image_ls.append(img_path)
            label_ls.append(label)

        label_np = np.array(label_ls)

        return image_ls, label_np, None, None

    def load_data(self, index):
        image = cv2.imread(self.image_path_ls[index], -1).astype(np.float32)
        label = self.label_ls[index]

        image = image - 150
        pose = np.ones([21, 3]) * 7777
        angle = np.ones([16, 4]) * 7777

        return image, label, pose, angle


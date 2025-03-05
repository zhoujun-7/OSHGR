import os
import cv2
import numpy as np
from collections import defaultdict
from natsort import natsorted
import glob

from .basic import Basic
from ..tool.vision_fn import dep_to_3channel_inv
from ..tool.misc import search_files
from ..tool.vision_fn import copy_dep_3channel


class ClassDataset(Basic):
    def __init__(
        self,
        DATA_DIR="data/OHG_cropped/TrainingBase",
        LABELS = list(range(23)),
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
    ):
        self.DATA_DIR = DATA_DIR
        self.LABELS = LABELS

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
        img_ls = natsorted(glob.glob(os.path.join(self.DATA_DIR, "*/*.png")))

        image_ls = []
        label_ls = []
        for image_path in img_ls:
            d = image_path.split("/")
            label = int(d[-2])
            if label in self.LABELS:
                image_ls.append(image_path)
                label_ls.append(label)

        label_np = np.array(label_ls)

        return image_ls, label_np, None, None

    def load_data(self, index):
        image = cv2.imread(self.image_path_ls[index], -1).astype(np.float32)
        image = image - 150
        label = self.label_ls[index]

        pose = np.ones([21, 3]) * 7777
        angle = np.ones([16, 4]) * 7777
        return image, label, pose, angle
    



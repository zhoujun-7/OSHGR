import os
import cv2
import math
import numpy as np
import random
import torch
from torch.utils.data import Dataset

from ..tool.vision_fn import copy_dep_3channel
from ..tool.hand_fn import kpt_to_angle


def do_random_erasing(img, probability=0.5, erase_ratio=(0.02, 0.4), patch_ratio=0.3, mu=(-20, 20), sigma=(3, 30)):
    if random.uniform(0, 1) > probability:
        return img
    img = img.copy()
    area = img.shape[0] * img.shape[1]
    for _ in range(100):
        target_area = random.uniform(erase_ratio[0], erase_ratio[1]) * area
        aspect_ratio = random.uniform(patch_ratio, 1 / patch_ratio)
        _mu = random.uniform(mu[0], mu[1])
        _sigma = random.uniform(sigma[0], sigma[1])
        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))
        if w < img.shape[1] and h < img.shape[0]:
            rand_patch = _sigma * np.random.randn(h, w) + _mu
            x1 = random.randint(0, img.shape[1] - h)
            y1 = random.randint(0, img.shape[0] - w)
            img[x1 : x1 + h, y1 : y1 + w] += rand_patch  # self.mean[0]
            return img
    return img


def do_gauss_noise(img, probability=0.5, mu=(-20, 20), sigma=(3, 30)):
    if random.uniform(0, 1) > probability:
        return img
    img = img.copy()
    h, w = img.shape
    noise_map = np.random.randn(h, w)
    _mu = random.uniform(mu[0], mu[1])
    _sigma = random.uniform(sigma[0], sigma[1])
    noise_map = noise_map * _sigma
    noise_map = noise_map + _mu
    img = img + noise_map
    return img


def do_smooth(img, probability=0.5, kernel=3):
    if random.uniform(0, 1) > probability:
        return img
    k = random.randint(kernel[0], kernel[1])
    kernel = np.ones((k, k), np.float32) / k**2
    smooth_img = cv2.filter2D(img, -1, kernel)
    return smooth_img


def do_augment(image, shift_u, shift_v, shift_d, rotation, scale, pose=None, d_range=(-150, 150)):
    h, w = image.shape
    rs_mat = cv2.getRotationMatrix2D((w / 2, h / 2), rotation, scale)

    src_start_u = max(-shift_u, 0)
    dst_start_u = max(shift_u, 0)
    src_start_v = max(-shift_v, 0)
    dst_start_v = max(shift_v, 0)
    src_end_u = min(w - 1, w - 1 - shift_u)
    dst_end_u = min(w - 1, w - 1 + shift_u)
    src_end_v = min(h - 1, h - 1 - shift_v)
    dst_end_v = min(h - 1, h - 1 + shift_v)

    image_ = np.ones_like(image) * d_range[1]
    image_[dst_start_v:dst_end_v, dst_start_u:dst_end_u] = image[src_start_v:src_end_v, src_start_u:src_end_u]
    image_[image_ > d_range[1]] = d_range[1]
    image_ = cv2.warpAffine(image_, rs_mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=d_range[1])
    image_ += shift_d
    mask = np.logical_or(image_ > d_range[1], image_ < d_range[0])
    image_[mask] = d_range[1]

    if pose is not None:
        if pose[0, 2] < 1000:
            pose_ = pose.copy()
            pose_[:, 0] = pose_[:, 0] + shift_u
            pose_[:, 1] = pose_[:, 1] + shift_v
            pose__ = np.ones_like(pose_)
            pose__[:, :2] = pose_[:, :2]
            pose__ = np.matmul(rs_mat, pose__.T).T
            pose_[:, :2] = pose__[:, :2]
            pose_[:, 2] += shift_d
            angle = kpt_to_angle(torch.from_numpy(pose_)).numpy()
        else:
            pose_ = pose
            angle = np.ones([16, 4]) * 7777
        return image_, pose_, angle
    else:
        return image_


class Basic(Dataset):
    def __init__(
        self,
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
    ):
        self.image_path_ls, self.label_ls, self.pose_ls, self.angle_ls = self.load_data_ls()

        self.RESOLUTION = RESOLUTION
        self.DEPTH_NORMALIZE = DEPTH_NORMALIZE
        self.SHIFT_U = SHIFT_U
        self.SHIFT_V = SHIFT_V
        self.SHIFT_D = SHIFT_D
        self.ROTATION = ROTATION
        self.SCALE = SCALE
        self.GAUSS_NOISE_PROBABILITY = GAUSS_NOISE_PROBABILITY
        self.GAUSS_NOISE_MU = GAUSS_NOISE_MU
        self.GAUSS_NOISE_SIGMA = GAUSS_NOISE_SIGMA
        self.ERASER_PROBABILITY = ERASER_PROBABILITY
        self.ERASE_RATIO = ERASE_RATIO
        self.ERASE_PATH_RATIO = ERASE_PATH_RATIO
        self.ERASE_MU = ERASE_MU
        self.ERASE_SIGMA = ERASE_SIGMA
        self.SMOOTH_PROBABILITY = SMOOTH_PROBABILITY
        self.SMOOTH_KERNEL = SMOOTH_KERNEL
        self.IS_AUGMENT = IS_AUGMENT

    def __len__(self):
        return len(self.image_path_ls)

    def load_data_ls(self):
        pass

    def load_data(self, index):
        pass

    def resize_to_fix(self, image, pose):
        h, w = image.shape
        image = cv2.resize(image, self.RESOLUTION, interpolation=cv2.INTER_NEAREST)
        pose[:, 0] = pose[:, 0] / w * self.RESOLUTION[0]
        pose[:, 1] = pose[:, 1] / h * self.RESOLUTION[1]
        return image, pose

    def do_augment(self, image, pose):
        shift_u = random.randint(self.SHIFT_U[0], self.SHIFT_U[1])
        shift_v = random.randint(self.SHIFT_V[0], self.SHIFT_V[1])
        shift_d = random.randint(self.SHIFT_D[0], self.SHIFT_D[1])
        rotation = random.randint(self.ROTATION[0], self.ROTATION[1])
        _a = random.random()
        scale = _a * self.SCALE[0] + (1 - _a) * self.SCALE[1]
        image, pose, angle = do_augment(image, shift_u, shift_v, shift_d, rotation, scale, pose=pose)

        noise_image = do_random_erasing(
            image,
            probability=self.ERASER_PROBABILITY,
            erase_ratio=self.ERASE_RATIO,
            patch_ratio=self.ERASE_PATH_RATIO,
            mu=self.ERASE_MU,
            sigma=self.ERASE_SIGMA,
        )

        noise_image = do_gauss_noise(
            noise_image,
            probability=self.GAUSS_NOISE_PROBABILITY,
            mu=self.GAUSS_NOISE_MU,
            sigma=self.GAUSS_NOISE_SIGMA,
        )

        noise_image = do_smooth(
            noise_image,
            probability=self.SMOOTH_PROBABILITY,
            kernel=self.SMOOTH_KERNEL,
        )

        noise_image[noise_image > 150] = 150
        noise_image[noise_image < -150] = -150

        pose = pose.astype(np.float32)
        angle = angle.astype(np.float32)
        return image, noise_image, pose, angle

    def __getitem__(self, index):
        image, label, pose, angle = self.load_data(index)
        image, pose = self.resize_to_fix(image, pose)

        if self.IS_AUGMENT:
            image, noise_image, pose, angle = self.do_augment(image, pose)
        else:
            noise_image = image.copy()

        image = image.astype(np.float32)
        image = image / self.DEPTH_NORMALIZE
        image = copy_dep_3channel(image)

        noise_image = noise_image.astype(np.float32)
        noise_image = noise_image / self.DEPTH_NORMALIZE
        noise_image = copy_dep_3channel(noise_image)


        timestamp = self.image_path_ls[index].split(os.sep)[-1][:-4]
        return image, noise_image, label, pose, angle, timestamp

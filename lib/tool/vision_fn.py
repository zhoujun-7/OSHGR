import numpy as np


def dep_to_3channel_inv(image):
    # image: (H, W, 3)
    out_image = np.zeros([image.shape[0], image.shape[1]], dtype=np.float32)
    out_image = out_image + image[..., 0] * 255 + image[..., 1]
    return out_image


def dep_to_3channel(image):
    # image: (H, W)
    image = image.copy().astype(np.uint16)
    out_image = np.zeros([image.shape[0], image.shape[1], 3], dtype=np.uint8)
    out_image[..., 0] = image // 256
    out_image[..., 1] = image % 256
    return out_image


def copy_dep_3channel(image: np.ndarray):
    h, w = image.shape
    image_ = np.zeros([3, h, w], dtype=image.dtype)
    image_ = image_ + image[None]
    return image_

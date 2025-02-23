import numpy as np
import cv2
import torch
import scipy
from typing import Union
from .geometry_fn import rt_trans, norm_vec
from scipy.spatial.transform import Rotation as R
from typing import Union
from copy import deepcopy
from kornia.geometry.conversions import (
    rotation_matrix_to_angle_axis,
    rotation_matrix_to_quaternion,
    angle_axis_to_rotation_matrix,
    quaternion_to_rotation_matrix,
    QuaternionCoeffOrder,
)


HANDS19_KID = {
    "W_": 0,
    "TM": 1,
    "TP": 6,
    "TD": 7,
    "TT": 8,
    "IM": 2,
    "IP": 9,
    "ID": 10,
    "IT": 11,
    "MM": 3,
    "MP": 12,
    "MD": 13,
    "MT": 14,
    "RM": 4,
    "RP": 15,
    "RD": 16,
    "RT": 17,
    "PM": 5,
    "PP": 18,
    "PD": 19,
    "PT": 20,
}

MANO_KID = {  #  mano pytorch, not mano original
    "W_": 0,
    "TM": 1,
    "TP": 2,
    "TD": 3,
    "TT": 4,
    "IM": 5,
    "IP": 6,
    "ID": 7,
    "IT": 8,
    "MM": 9,
    "MP": 10,
    "MD": 11,
    "MT": 12,
    "RM": 13,
    "RP": 14,
    "RD": 15,
    "RT": 16,
    "PM": 17,
    "PP": 18,
    "PD": 19,
    "PT": 20,
}

MANO_ANGLE_ID = {
    "IM": 0,
    "IP": 1,
    "ID": 2,
    "MM": 3,
    "MP": 4,
    "MD": 5,
    "PM": 6,
    "PP": 7,
    "PD": 8,
    "RM": 9,
    "RP": 10,
    "RD": 11,
    "TM": 12,
    "TP": 13,
    "TD": 14,
}

MAN0_ANGLE_LINK = [
    [MANO_ANGLE_ID["IM"], MANO_ANGLE_ID["IP"]],
    [MANO_ANGLE_ID["IM"], MANO_ANGLE_ID["TM"]],
    [MANO_ANGLE_ID["IM"], MANO_ANGLE_ID["MM"]],
    [MANO_ANGLE_ID["IP"], MANO_ANGLE_ID["IM"]],
    [MANO_ANGLE_ID["IP"], MANO_ANGLE_ID["ID"]],
    [MANO_ANGLE_ID["ID"], MANO_ANGLE_ID["IP"]],
    #
    [MANO_ANGLE_ID["MM"], MANO_ANGLE_ID["MP"]],
    [MANO_ANGLE_ID["MM"], MANO_ANGLE_ID["IM"]],
    [MANO_ANGLE_ID["MM"], MANO_ANGLE_ID["RM"]],
    [MANO_ANGLE_ID["MP"], MANO_ANGLE_ID["MM"]],
    [MANO_ANGLE_ID["MP"], MANO_ANGLE_ID["MD"]],
    [MANO_ANGLE_ID["MD"], MANO_ANGLE_ID["MP"]],
    #
    [MANO_ANGLE_ID["RM"], MANO_ANGLE_ID["RP"]],
    [MANO_ANGLE_ID["RM"], MANO_ANGLE_ID["MM"]],
    [MANO_ANGLE_ID["RM"], MANO_ANGLE_ID["PM"]],
    [MANO_ANGLE_ID["RP"], MANO_ANGLE_ID["RM"]],
    [MANO_ANGLE_ID["RP"], MANO_ANGLE_ID["RD"]],
    [MANO_ANGLE_ID["RD"], MANO_ANGLE_ID["RP"]],
    #
    [MANO_ANGLE_ID["PM"], MANO_ANGLE_ID["PP"]],
    [MANO_ANGLE_ID["PM"], MANO_ANGLE_ID["RM"]],
    [MANO_ANGLE_ID["PP"], MANO_ANGLE_ID["PM"]],
    [MANO_ANGLE_ID["PP"], MANO_ANGLE_ID["PD"]],
    [MANO_ANGLE_ID["PD"], MANO_ANGLE_ID["PP"]],
    #
    [MANO_ANGLE_ID["TM"], MANO_ANGLE_ID["TP"]],
    [MANO_ANGLE_ID["TM"], MANO_ANGLE_ID["IM"]],
    [MANO_ANGLE_ID["TP"], MANO_ANGLE_ID["TM"]],
    [MANO_ANGLE_ID["TP"], MANO_ANGLE_ID["TD"]],
    [MANO_ANGLE_ID["TD"], MANO_ANGLE_ID["TP"]],
]

MAN0_KPT_LINK = [
    [MANO_KID["W_"], MANO_KID["IM"]],
    [MANO_KID["W_"], MANO_KID["MM"]],
    [MANO_KID["W_"], MANO_KID["RM"]],
    [MANO_KID["W_"], MANO_KID["PM"]],
    [MANO_KID["W_"], MANO_KID["TM"]],
    #
    [MANO_KID["IM"], MANO_KID["W_"]],
    [MANO_KID["IM"], MANO_KID["IP"]],
    [MANO_KID["IM"], MANO_KID["TM"]],
    [MANO_KID["IM"], MANO_KID["MM"]],
    [MANO_KID["IP"], MANO_KID["IM"]],
    [MANO_KID["IP"], MANO_KID["ID"]],
    [MANO_KID["ID"], MANO_KID["IP"]],
    [MANO_KID["ID"], MANO_KID["IT"]],
    [MANO_KID["IT"], MANO_KID["ID"]],
    #
    [MANO_KID["MM"], MANO_KID["W_"]],
    [MANO_KID["MM"], MANO_KID["MP"]],
    [MANO_KID["MM"], MANO_KID["IM"]],
    [MANO_KID["MM"], MANO_KID["RM"]],
    [MANO_KID["MP"], MANO_KID["MM"]],
    [MANO_KID["MP"], MANO_KID["MD"]],
    [MANO_KID["MD"], MANO_KID["MP"]],
    [MANO_KID["MD"], MANO_KID["MT"]],
    [MANO_KID["MT"], MANO_KID["MD"]],
    #
    [MANO_KID["RM"], MANO_KID["W_"]],
    [MANO_KID["RM"], MANO_KID["RP"]],
    [MANO_KID["RM"], MANO_KID["MM"]],
    [MANO_KID["RM"], MANO_KID["PM"]],
    [MANO_KID["RP"], MANO_KID["RM"]],
    [MANO_KID["RP"], MANO_KID["RD"]],
    [MANO_KID["RD"], MANO_KID["RP"]],
    [MANO_KID["RD"], MANO_KID["RT"]],
    [MANO_KID["RT"], MANO_KID["RD"]],
    #
    [MANO_KID["PM"], MANO_KID["W_"]],
    [MANO_KID["PM"], MANO_KID["PP"]],
    [MANO_KID["PM"], MANO_KID["RM"]],
    [MANO_KID["PP"], MANO_KID["PM"]],
    [MANO_KID["PP"], MANO_KID["PD"]],
    [MANO_KID["PD"], MANO_KID["PP"]],
    [MANO_KID["PD"], MANO_KID["PT"]],
    [MANO_KID["PT"], MANO_KID["PD"]],
    #
    [MANO_KID["TM"], MANO_KID["W_"]],
    [MANO_KID["TM"], MANO_KID["TP"]],
    [MANO_KID["TM"], MANO_KID["IM"]],
    [MANO_KID["TP"], MANO_KID["TM"]],
    [MANO_KID["TP"], MANO_KID["TD"]],
    [MANO_KID["TD"], MANO_KID["TP"]],
    [MANO_KID["TD"], MANO_KID["TT"]],
    [MANO_KID["TT"], MANO_KID["TD"]],
]

HANDS19_TO_MANO = [0] * 21
MANO_TO_HANDS19 = [0] * 21
for k, v in MANO_KID.items():
    HANDS19_TO_MANO[MANO_KID[k]] = HANDS19_KID[k]
    MANO_TO_HANDS19[HANDS19_KID[k]] = MANO_KID[k]

MANO_KPT_TO_ANGLE_ID = [
    "IM",
    "IP",
    "ID",
    "MM",
    "MP",
    "MD",
    "PM",
    "PP",
    "PD",
    "RM",
    "RP",
    "RD",
    "TM",
    "TP",
    "TD",
]
for i in range(len(MANO_KPT_TO_ANGLE_ID)):
    MANO_KPT_TO_ANGLE_ID[i] = MANO_KID[MANO_KPT_TO_ANGLE_ID[i]]

MANO_KPT_TO_ANGLE_ID2 = [
    "IP",
    "ID",
    "IT",
    "MP",
    "MD",
    "MT",
    "PP",
    "PD",
    "PT",
    "RP",
    "RD",
    "RT",
    "TP",
    "TD",
    "TT",
]
for i in range(len(MANO_KPT_TO_ANGLE_ID2)):
    MANO_KPT_TO_ANGLE_ID2[i] = MANO_KID[MANO_KPT_TO_ANGLE_ID2[i]]

MANO_TO_NATURAL = [
    0,
    13,
    14,
    15,
    16,
    1,
    2,
    3,
    17,
    4,
    5,
    6,
    18,
    10,
    11,
    12,
    19,
    7,
    8,
    9,
    20,
]

HANDS19_TO_NATURAL = [
    0,
    1,
    6,
    7,
    8,
    2,
    9,
    10,
    11,
    3,
    12,
    13,
    14,
    4,
    15,
    16,
    17,
    5,
    18,
    19,
    20,
]


SKELETON_LENGTH = {
    "IM-W_": 59.55299004,
    "IP-IM": 33.23330005,
    "ID-IP": 18.62183555,
    "IT-ID": 15.53232585,
    "MM-W_": 53.03110326,
    "MP-MM": 37.8904979,
    "MD-MP": 24.4058474,
    "MT-MD": 17.96687391,
    "RM-W_": 50.19074925,
    "RP-RM": 36.76043908,
    "RD-RP": 20.75775447,
    "RT-RD": 18.60351234,
    "PM-W_": 47.86896277,
    "PP-PM": 30.96773914,
    "PD-PP": 18.41514609,
    "PT-PD": 16.88691568,
    "TM-W_": 12.55629209,
    "TP-TM": 42.20418299,
    "TD-TP": 26.06938172,
    "TT-TD": 20.80335779,
}


def kpt_to_angle(
    kpt: Union[np.ndarray, torch.Tensor],
    k_type: str = "manopth",
    angle_type: str = "quaternion",
    wrist_rotation: bool = True,
):
    """
    The angle is different from mano angle definition.
    kps shape should be (21, 3) or (N, 21, 3).
    """
    assert kpt.shape[-2:] == (21, 3) or len(kpt.shape) > 3
    assert angle_type in ["quaternion", "euler", "rotation_matrix"]

    if len(kpt.shape) == 2:
        frame_size = [4, 4]
    else:
        frame_size = list(kpt.shape[:-2]) + [4, 4]

    # kpt = deepcopy(kpt)

    if k_type == "hands19":
        KID = HANDS19_KID
    elif k_type == "manopth":
        KID = MANO_KID

    if isinstance(kpt, np.ndarray):
        kpt = kpt.copy()
        # build frame on wrist
        base_z_axis = np.concatenate(
            [
                kpt[..., [KID["IM"]], :] - kpt[..., [KID["MM"]], :],
                kpt[..., [KID["MM"]], :] - kpt[..., [KID["RM"]], :],
                kpt[..., [KID["RM"]], :] - kpt[..., [KID["PM"]], :],
                kpt[..., [KID["IM"]], :] - kpt[..., [KID["RM"]], :],
                kpt[..., [KID["MM"]], :] - kpt[..., [KID["PM"]], :],
                kpt[..., [KID["IM"]], :] - kpt[..., [KID["PM"]], :],
            ],
            axis=-2,
        )
        base_z_axis = norm_vec(base_z_axis)
        base_z_axis = base_z_axis.mean(-2)
        base_z_axis = norm_vec(base_z_axis)

        base_x_axis = np.concatenate(
            [
                kpt[..., [KID["W_"]], :] - kpt[..., [KID["IM"]], :],
                kpt[..., [KID["W_"]], :] - kpt[..., [KID["MM"]], :],
                kpt[..., [KID["W_"]], :] - kpt[..., [KID["RM"]], :],
            ],
            axis=-2,
        )
        base_x_axis = norm_vec(base_x_axis)
        base_x_axis = base_x_axis.mean(-2)
        base_x_axis = norm_vec(base_x_axis)
        base_y_axis = np.cross(base_z_axis, base_x_axis)
        base_y_axis = norm_vec(base_y_axis)
        base_z_axis = np.cross(base_x_axis, base_y_axis)

        base_frame = np.zeros(frame_size)
        base_frame[..., 3, 3] = 1
        base_frame[..., :3, 0] = base_x_axis
        base_frame[..., :3, 1] = base_y_axis
        base_frame[..., :3, 2] = base_z_axis
        base_frame[..., :3, 3] = kpt[..., KID["W_"], :]

        o_f_P = kpt
        w_o_T = np.linalg.inv(base_frame)
        w_f_P = rt_trans(w_o_T, o_f_P)

        w_f_V_x = np.concatenate(
            [
                #
                w_f_P[..., [KID["IM"]], :] - w_f_P[..., [KID["IP"]], :],
                w_f_P[..., [KID["IP"]], :] - w_f_P[..., [KID["ID"]], :],
                w_f_P[..., [KID["ID"]], :] - w_f_P[..., [KID["IT"]], :],
                #
                w_f_P[..., [KID["MM"]], :] - w_f_P[..., [KID["MP"]], :],
                w_f_P[..., [KID["MP"]], :] - w_f_P[..., [KID["MD"]], :],
                w_f_P[..., [KID["MD"]], :] - w_f_P[..., [KID["MT"]], :],
                #
                w_f_P[..., [KID["PM"]], :] - w_f_P[..., [KID["PP"]], :],
                w_f_P[..., [KID["PP"]], :] - w_f_P[..., [KID["PD"]], :],
                w_f_P[..., [KID["PD"]], :] - w_f_P[..., [KID["PT"]], :],
                #
                w_f_P[..., [KID["RM"]], :] - w_f_P[..., [KID["RP"]], :],
                w_f_P[..., [KID["RP"]], :] - w_f_P[..., [KID["RD"]], :],
                w_f_P[..., [KID["RD"]], :] - w_f_P[..., [KID["RT"]], :],
                #
                w_f_P[..., [KID["TM"]], :] - w_f_P[..., [KID["TP"]], :],
                w_f_P[..., [KID["TP"]], :] - w_f_P[..., [KID["TD"]], :],
                w_f_P[..., [KID["TD"]], :] - w_f_P[..., [KID["TT"]], :],
            ],
            axis=-2,
        )

        w_f_V_x = norm_vec(w_f_V_x)
        w_f_V_z = np.zeros_like(w_f_V_x)
        w_f_V_z[..., 2] = 1
        w_f_V_y = np.cross(w_f_V_z, w_f_V_x)
        w_f_V_y = norm_vec(w_f_V_y)
        w_f_V_z = np.cross(w_f_V_x, w_f_V_y)
        w_f_V_z = norm_vec(w_f_V_z)

        w_f_R = np.concatenate(
            [
                w_f_V_x[..., None],
                w_f_V_y[..., None],
                w_f_V_z[..., None],
            ],
            axis=-1,
        )
        frame_size.insert(-2, w_f_R.shape[-3])
        w_f_T = np.zeros(frame_size)
        w_f_T[..., -1] = 1
        w_f_T[..., :3, :3] = w_f_R
        w_f_T[..., :3, 3] = w_f_P[
            ...,
            [
                KID["IP"],
                KID["ID"],
                KID["IT"],
                KID["MP"],
                KID["MD"],
                KID["MT"],
                KID["PP"],
                KID["PD"],
                KID["PT"],
                KID["RP"],
                KID["RD"],
                KID["RT"],
                KID["TP"],
                KID["TD"],
                KID["TT"],
            ],
            :,
        ]

        f1_f2_R = deepcopy(w_f_R)
        f1_w__R = np.linalg.inv(w_f_R)
        w__f2_R = deepcopy(w_f_R)

        f1_f2_R[
            ...,
            [
                MANO_ANGLE_ID["IP"],
                MANO_ANGLE_ID["ID"],
                MANO_ANGLE_ID["MP"],
                MANO_ANGLE_ID["MD"],
                MANO_ANGLE_ID["PP"],
                MANO_ANGLE_ID["PD"],
                MANO_ANGLE_ID["RP"],
                MANO_ANGLE_ID["RD"],
                MANO_ANGLE_ID["TP"],
                MANO_ANGLE_ID["TD"],
            ],
            :,
            :,
        ] = np.bmm(
            f1_w__R[
                ...,
                [
                    MANO_ANGLE_ID["IM"],
                    MANO_ANGLE_ID["IP"],
                    MANO_ANGLE_ID["MM"],
                    MANO_ANGLE_ID["MP"],
                    MANO_ANGLE_ID["PM"],
                    MANO_ANGLE_ID["PP"],
                    MANO_ANGLE_ID["RM"],
                    MANO_ANGLE_ID["RP"],
                    MANO_ANGLE_ID["TM"],
                    MANO_ANGLE_ID["TP"],
                ],
                :,
                :,
            ],
            w__f2_R[
                ...,
                [
                    MANO_ANGLE_ID["IP"],
                    MANO_ANGLE_ID["ID"],
                    MANO_ANGLE_ID["MP"],
                    MANO_ANGLE_ID["MD"],
                    MANO_ANGLE_ID["PP"],
                    MANO_ANGLE_ID["PD"],
                    MANO_ANGLE_ID["RP"],
                    MANO_ANGLE_ID["RD"],
                    MANO_ANGLE_ID["TP"],
                    MANO_ANGLE_ID["TD"],
                ],
                :,
                :,
            ],
        )

        if wrist_rotation:
            w_R = base_frame[..., None, :3, :3]
            f1_f2_R = np.concatenate([f1_f2_R, w_R], axis=-3)

        if angle_type == "quaternion":
            quat = R.from_matrix(f1_f2_R.reshape(-1, 3, 3)).as_quat()
            if len(kpt.shape) == 2:
                quat = quat.reshape(f1_f2_R.shape[-3], 4)
            else:
                quat = quat.reshape(-1, f1_f2_R.shape[-3], 4)
            return quat

        elif angle_type == "euler":
            angle = R.from_matrix(f1_f2_R.reshape(-1, 3, 3)).as_euler("xyz")
            if len(kpt.shape) == 2:
                angle = angle.reshape(f1_f2_R.shape[-3], 3)
            else:
                angle = angle.reshape(-1, f1_f2_R.shape[-3], 3)
            return angle

        elif angle_type == "rotation_matrix":
            return f1_f2_R

    elif isinstance(kpt, torch.Tensor):
        kpt = kpt.clone()
        # build frame on wrist
        base_z_axis = torch.cat(
            [
                kpt[..., [KID["IM"]], :] - kpt[..., [KID["MM"]], :],
                kpt[..., [KID["MM"]], :] - kpt[..., [KID["RM"]], :],
                kpt[..., [KID["RM"]], :] - kpt[..., [KID["PM"]], :],
                kpt[..., [KID["IM"]], :] - kpt[..., [KID["RM"]], :],
                kpt[..., [KID["MM"]], :] - kpt[..., [KID["PM"]], :],
                kpt[..., [KID["IM"]], :] - kpt[..., [KID["PM"]], :],
            ],
            axis=-2,
        )
        base_z_axis = norm_vec(base_z_axis)
        base_z_axis = base_z_axis.mean(-2)
        base_z_axis = norm_vec(base_z_axis)

        base_x_axis = torch.cat(
            [
                kpt[..., [KID["W_"]], :] - kpt[..., [KID["IM"]], :],
                kpt[..., [KID["W_"]], :] - kpt[..., [KID["MM"]], :],
                kpt[..., [KID["W_"]], :] - kpt[..., [KID["RM"]], :],
            ],
            axis=-2,
        )
        base_x_axis = norm_vec(base_x_axis)
        base_x_axis = base_x_axis.mean(-2)
        base_x_axis = norm_vec(base_x_axis)
        base_y_axis = torch.cross(base_z_axis, base_x_axis)
        base_y_axis = norm_vec(base_y_axis)
        base_z_axis = torch.cross(base_x_axis, base_y_axis)

        base_frame = torch.zeros(frame_size, device=kpt.device, dtype=kpt.dtype)
        base_frame[..., 3, 3] = 1
        base_frame[..., :3, 0] = base_x_axis
        base_frame[..., :3, 1] = base_y_axis
        base_frame[..., :3, 2] = base_z_axis
        base_frame[..., :3, 3] = kpt[..., KID["W_"], :]

        o_f_P = kpt
        w_o_T = torch.linalg.inv(base_frame).to(o_f_P)
        w_f_P = rt_trans(w_o_T, o_f_P)

        w_f_V_x = torch.cat(
            [
                #
                w_f_P[..., [KID["IM"]], :] - w_f_P[..., [KID["IP"]], :],
                w_f_P[..., [KID["IP"]], :] - w_f_P[..., [KID["ID"]], :],
                w_f_P[..., [KID["ID"]], :] - w_f_P[..., [KID["IT"]], :],
                #
                w_f_P[..., [KID["MM"]], :] - w_f_P[..., [KID["MP"]], :],
                w_f_P[..., [KID["MP"]], :] - w_f_P[..., [KID["MD"]], :],
                w_f_P[..., [KID["MD"]], :] - w_f_P[..., [KID["MT"]], :],
                #
                w_f_P[..., [KID["PM"]], :] - w_f_P[..., [KID["PP"]], :],
                w_f_P[..., [KID["PP"]], :] - w_f_P[..., [KID["PD"]], :],
                w_f_P[..., [KID["PD"]], :] - w_f_P[..., [KID["PT"]], :],
                #
                w_f_P[..., [KID["RM"]], :] - w_f_P[..., [KID["RP"]], :],
                w_f_P[..., [KID["RP"]], :] - w_f_P[..., [KID["RD"]], :],
                w_f_P[..., [KID["RD"]], :] - w_f_P[..., [KID["RT"]], :],
                #
                w_f_P[..., [KID["TM"]], :] - w_f_P[..., [KID["TP"]], :],
                w_f_P[..., [KID["TP"]], :] - w_f_P[..., [KID["TD"]], :],
                w_f_P[..., [KID["TD"]], :] - w_f_P[..., [KID["TT"]], :],
            ],
            axis=-2,
        )

        w_f_V_x = norm_vec(w_f_V_x)
        w_f_V_z = torch.zeros_like(w_f_V_x)
        w_f_V_z[..., 2] = 1
        w_f_V_y = torch.cross(w_f_V_z, w_f_V_x)
        w_f_V_y = norm_vec(w_f_V_y)
        w_f_V_z = torch.cross(w_f_V_x, w_f_V_y)
        w_f_V_z = norm_vec(w_f_V_z)

        w_f_R = torch.cat(
            [
                w_f_V_x[..., None],
                w_f_V_y[..., None],
                w_f_V_z[..., None],
            ],
            axis=-1,
        )
        frame_size.insert(-2, w_f_R.shape[-3])
        w_f_T = torch.zeros(frame_size, device=kpt.device, dtype=kpt.dtype)
        w_f_T[..., -1] = 1
        w_f_T[..., :3, :3] = w_f_R
        w_f_T[..., :3, 3] = w_f_P[
            ...,
            [
                KID["IP"],
                KID["ID"],
                KID["IT"],
                KID["MP"],
                KID["MD"],
                KID["MT"],
                KID["PP"],
                KID["PD"],
                KID["PT"],
                KID["RP"],
                KID["RD"],
                KID["RT"],
                KID["TP"],
                KID["TD"],
                KID["TT"],
            ],
            :,
        ]

        f1_f2_R = w_f_R.clone()
        f1_w__R = w_f_R.mT  #  torch.linalg.inv(w_f_R).to(w_f_R)
        w__f2_R = w_f_R.clone()

        f1_f2_R[
            ...,
            [
                MANO_ANGLE_ID["IP"],
                MANO_ANGLE_ID["ID"],
                MANO_ANGLE_ID["MP"],
                MANO_ANGLE_ID["MD"],
                MANO_ANGLE_ID["PP"],
                MANO_ANGLE_ID["PD"],
                MANO_ANGLE_ID["RP"],
                MANO_ANGLE_ID["RD"],
                MANO_ANGLE_ID["TP"],
                MANO_ANGLE_ID["TD"],
            ],
            :,
            :,
        ] = torch.matmul(
            f1_w__R[
                ...,
                [
                    MANO_ANGLE_ID["IM"],
                    MANO_ANGLE_ID["IP"],
                    MANO_ANGLE_ID["MM"],
                    MANO_ANGLE_ID["MP"],
                    MANO_ANGLE_ID["PM"],
                    MANO_ANGLE_ID["PP"],
                    MANO_ANGLE_ID["RM"],
                    MANO_ANGLE_ID["RP"],
                    MANO_ANGLE_ID["TM"],
                    MANO_ANGLE_ID["TP"],
                ],
                :,
                :,
            ],
            w__f2_R[
                ...,
                [
                    MANO_ANGLE_ID["IP"],
                    MANO_ANGLE_ID["ID"],
                    MANO_ANGLE_ID["MP"],
                    MANO_ANGLE_ID["MD"],
                    MANO_ANGLE_ID["PP"],
                    MANO_ANGLE_ID["PD"],
                    MANO_ANGLE_ID["RP"],
                    MANO_ANGLE_ID["RD"],
                    MANO_ANGLE_ID["TP"],
                    MANO_ANGLE_ID["TD"],
                ],
                :,
                :,
            ],
        ).to(
            f1_f2_R
        )

        if wrist_rotation:
            w_R = base_frame[..., None, :3, :3]
            f1_f2_R = torch.cat([f1_f2_R, w_R], axis=-3)

        if angle_type == "quaternion":
            quat = rotation_matrix_to_quaternion(f1_f2_R, order=QuaternionCoeffOrder.WXYZ)
            return quat

        elif angle_type == "euler":
            euler = rotation_matrix_to_angle_axis(f1_f2_R)
            return euler

        elif angle_type == "rotation_matrix":
            return f1_f2_R


def angle_to_kpt(angle: Union[np.ndarray, torch.Tensor], k_type: str = "manopth", angle_type: str = "quaternion"):
    """
    If angle has wrist rotation, then the index of the wrist rotation is the last.
    angle shape should be (B, N, 3) or (B, N, 4) or (B, N, 3, 3)
    """

    if isinstance(angle, np.ndarray):
        if angle_type == "euler":
            rot_mat = R.from_euler("xyz", angle.reshape(-1, 3)).as_matrix()
            if len(angle.shape) == 2:
                rot_mat = rot_mat.reshape(angle.shape[-2], 3, 3)
                data_size = [21, 3]
            elif len(angle.shape) == 3:
                rot_mat = rot_mat.reshape(-1, angle.shape[-2], 3, 3)
                data_size = [angle.shape[0], 21, 3]
        elif angle_type == "quaternion":
            rot_mat = R.from_quat(angle.reshape(-1, 4)).as_matrix()
            rot_mat = rot_mat.reshape(-1, angle.shape[-2], 3, 3)
            if len(angle.shape) == 2:
                rot_mat = rot_mat.reshape(angle.shape[-2], 3, 3)
                data_size = [21, 3]
            elif len(angle.shape) == 3:
                rot_mat = rot_mat.reshape(-1, angle.shape[-2], 3, 3)
                data_size = [angle.shape[0], 21, 3]
        elif angle_type == "rotation_matrix":
            rot_mat = angle
            if len(angle.shape) == 3:
                rot_mat = rot_mat.reshape(angle.shape[-3], 3, 3)
                data_size = [21, 3]
            elif len(angle.shape) == 4:
                rot_mat = rot_mat.reshape(-1, angle.shape[-3], 3, 3)
                data_size = [angle.shape[0], 21, 3]
        else:
            raise ValueError(f"Unkonw angle_type = {angle_type}")
    elif isinstance(angle, torch.Tensor):
        if angle_type == "euler":
            rot_mat = angle_axis_to_rotation_matrix(angle.reshape(-1, 3))
            if len(angle.shape) == 2:
                rot_mat = rot_mat.reshape(angle.shape[-2], 3, 3)
                data_size = [21, 3]
            elif len(angle.shape) == 3:
                rot_mat = rot_mat.reshape(-1, angle.shape[-2], 3, 3)
                data_size = [angle.shape[0], 21, 3]
        elif angle_type == "quaternion":
            rot_mat = quaternion_to_rotation_matrix(angle.reshape(-1, 4), order=QuaternionCoeffOrder.WXYZ)
            if len(angle.shape) == 2:
                rot_mat = rot_mat.reshape(angle.shape[-2], 3, 3)
                data_size = [21, 3]
            elif len(angle.shape) == 3:
                rot_mat = rot_mat.reshape(-1, angle.shape[-2], 3, 3)
                data_size = [angle.shape[0], 21, 3]
        elif angle_type == "rotation_matrix":
            rot_mat = angle
            if len(angle.shape) == 3:
                rot_mat = rot_mat.reshape(angle.shape[-3], 3, 3)
                data_size = [21, 3]
            elif len(angle.shape) == 4:
                rot_mat = rot_mat.reshape(-1, angle.shape[-3], 3, 3)
                data_size = [angle.shape[0], 21, 3]
        else:
            raise ValueError(f"Unkonw angle_type = {angle_type}")

    if k_type == "hands19":
        KID = HANDS19_KID
    elif k_type == "manopth":
        KID = MANO_KID

    if isinstance(angle, np.ndarray):
        ### M
        finger_M = np.array(
            [
                [-55.64645397, 0.44047862, 21.20935553],
                [-53.02614766, -0.2821201, -0.66782192],
                [-38.26445541, 0.29198112, -32.47857],
                [-44.95528725, -0.09939869, -16.44536],
                [-0.83741711, -0.96920821, 12.49079017],
            ]
        )

        _data_size = deepcopy(data_size)
        _data_size[-2] = 5
        vec_x = np.zeros(_data_size)[..., None, :]
        vec_x[..., 0, 0] = 1

        ### P
        rot_mat_M = rot_mat[
            ...,
            [
                MANO_ANGLE_ID["IM"],
                MANO_ANGLE_ID["MM"],
                MANO_ANGLE_ID["PM"],
                MANO_ANGLE_ID["RM"],
                MANO_ANGLE_ID["TM"],
            ],
            :,
            :,
        ]

        vec_M = rt_trans(rot_mat_M, vec_x)
        vec_M = vec_M[..., 0, :]

        length_M = np.array(
            [
                SKELETON_LENGTH["IP-IM"],
                SKELETON_LENGTH["MP-MM"],
                SKELETON_LENGTH["PP-PM"],
                SKELETON_LENGTH["RP-RM"],
                SKELETON_LENGTH["TP-TM"],
            ]
        )[:, None]

        finger_P = -1 * vec_M * length_M

        ### D
        rot_mat_P = rot_mat[
            ...,
            [
                MANO_ANGLE_ID["IP"],
                MANO_ANGLE_ID["MP"],
                MANO_ANGLE_ID["PP"],
                MANO_ANGLE_ID["RP"],
                MANO_ANGLE_ID["TP"],
            ],
            :,
            :,
        ]

        # rot_mat_P = rt_trans(rot_mat_M, rot_mat_P)
        vec_P = rt_trans(rot_mat_P, vec_x)
        vec_P = rt_trans(rot_mat_M, vec_P)
        vec_P = vec_P[..., 0, :]

        length_P = np.array(
            [
                SKELETON_LENGTH["ID-IP"],
                SKELETON_LENGTH["MD-MP"],
                SKELETON_LENGTH["PD-PP"],
                SKELETON_LENGTH["RD-RP"],
                SKELETON_LENGTH["TD-TP"],
            ]
        )[:, None]

        finger_D = -1 * vec_P * length_P

        # T
        rot_mat_D = rot_mat[
            ...,
            [
                MANO_ANGLE_ID["ID"],
                MANO_ANGLE_ID["MD"],
                MANO_ANGLE_ID["PD"],
                MANO_ANGLE_ID["RD"],
                MANO_ANGLE_ID["TD"],
            ],
            :,
            :,
        ]
        # rot_mat_D = rt_trans(rot_mat_P, rot_mat_D)

        vec_D = rt_trans(rot_mat_D, vec_x)
        vec_D = rt_trans(rot_mat_P, vec_D)
        vec_D = rt_trans(rot_mat_M, vec_D)
        vec_D = vec_D[..., 0, :]

        length_D = np.array(
            [
                SKELETON_LENGTH["IT-ID"],
                SKELETON_LENGTH["MT-MD"],
                SKELETON_LENGTH["PT-PD"],
                SKELETON_LENGTH["RT-RD"],
                SKELETON_LENGTH["TT-TD"],
            ]
        )[:, None]

        finger_T = -1 * vec_D * length_D

        kpt = np.zeros(data_size)
        kpt[..., [KID["IM"], KID["MM"], KID["PM"], KID["RM"], KID["TM"]], :] = finger_M
        kpt[..., [KID["IP"], KID["MP"], KID["PP"], KID["RP"], KID["TP"]], :] = finger_P + finger_M
        kpt[..., [KID["ID"], KID["MD"], KID["PD"], KID["RD"], KID["TD"]], :] = finger_D + finger_P + finger_M
        kpt[..., [KID["IT"], KID["MT"], KID["PT"], KID["RT"], KID["TT"]], :] = finger_T + finger_D + finger_P + finger_M

        if rot_mat.shape[-3] == 16:
            rot_mat_W = rot_mat[..., -1, :, :]
            kpt = rt_trans(rot_mat_W, kpt)
        return kpt

    elif isinstance(angle, torch.Tensor):
        device = angle.device
        dtype = angle.dtype

        _data_size = deepcopy(data_size)
        _data_size[-2] = angle.shape[-2]
        _data_size.append(3)

        ### M
        finger_M = torch.tensor(
            [
                [-55.64645397, 0.44047862, 21.20935553],
                [-53.02614766, -0.2821201, -0.66782192],
                [-38.26445541, 0.29198112, -32.47857],
                [-44.95528725, -0.09939869, -16.44536],
                [-0.83741711, -0.96920821, 12.49079017],
            ],
            device=device,
            dtype=dtype,
        )

        _data_size = deepcopy(data_size)
        _data_size[-2] = 5
        vec_x = torch.zeros(_data_size, device=device, dtype=dtype)[..., None, :]
        vec_x[..., 0, 0] = 1

        ### P
        rot_mat_M = rot_mat[
            ...,
            [
                MANO_ANGLE_ID["IM"],
                MANO_ANGLE_ID["MM"],
                MANO_ANGLE_ID["PM"],
                MANO_ANGLE_ID["RM"],
                MANO_ANGLE_ID["TM"],
            ],
            :,
            :,
        ]

        vec_M = rt_trans(rot_mat_M, vec_x)
        vec_M = vec_M[..., 0, :]

        length_M = torch.tensor(
            [
                SKELETON_LENGTH["IP-IM"],
                SKELETON_LENGTH["MP-MM"],
                SKELETON_LENGTH["PP-PM"],
                SKELETON_LENGTH["RP-RM"],
                SKELETON_LENGTH["TP-TM"],
            ],
            device=device,
            dtype=dtype,
        )[:, None]

        finger_P = -1 * vec_M * length_M

        ### D
        rot_mat_P = rot_mat[
            ...,
            [
                MANO_ANGLE_ID["IP"],
                MANO_ANGLE_ID["MP"],
                MANO_ANGLE_ID["PP"],
                MANO_ANGLE_ID["RP"],
                MANO_ANGLE_ID["TP"],
            ],
            :,
            :,
        ]

        # rot_mat_P = rt_trans(rot_mat_M, rot_mat_P)
        vec_P = rt_trans(rot_mat_P, vec_x)
        vec_P = rt_trans(rot_mat_M, vec_P)
        vec_P = vec_P[..., 0, :]

        length_P = torch.tensor(
            [
                SKELETON_LENGTH["ID-IP"],
                SKELETON_LENGTH["MD-MP"],
                SKELETON_LENGTH["PD-PP"],
                SKELETON_LENGTH["RD-RP"],
                SKELETON_LENGTH["TD-TP"],
            ],
            device=device,
            dtype=dtype,
        )[:, None]

        finger_D = -1 * vec_P * length_P

        # T
        rot_mat_D = rot_mat[
            ...,
            [
                MANO_ANGLE_ID["ID"],
                MANO_ANGLE_ID["MD"],
                MANO_ANGLE_ID["PD"],
                MANO_ANGLE_ID["RD"],
                MANO_ANGLE_ID["TD"],
            ],
            :,
            :,
        ]
        # rot_mat_D = rt_trans(rot_mat_P, rot_mat_D)

        vec_D = rt_trans(rot_mat_D, vec_x)
        vec_D = rt_trans(rot_mat_P, vec_D)
        vec_D = rt_trans(rot_mat_M, vec_D)
        vec_D = vec_D[..., 0, :]

        length_D = torch.tensor(
            [
                SKELETON_LENGTH["IT-ID"],
                SKELETON_LENGTH["MT-MD"],
                SKELETON_LENGTH["PT-PD"],
                SKELETON_LENGTH["RT-RD"],
                SKELETON_LENGTH["TT-TD"],
            ],
            device=device,
            dtype=dtype,
        )[:, None]

        finger_T = -1 * vec_D * length_D

        kpt = torch.zeros(data_size, device=device, dtype=dtype)
        kpt[..., [KID["IM"], KID["MM"], KID["PM"], KID["RM"], KID["TM"]], :] = finger_M
        kpt[..., [KID["IP"], KID["MP"], KID["PP"], KID["RP"], KID["TP"]], :] = finger_P + finger_M
        kpt[..., [KID["ID"], KID["MD"], KID["PD"], KID["RD"], KID["TD"]], :] = finger_D + finger_P + finger_M
        kpt[..., [KID["IT"], KID["MT"], KID["PT"], KID["RT"], KID["TT"]], :] = finger_T + finger_D + finger_P + finger_M

        if rot_mat.shape[-3] == 16:
            rot_mat_W = rot_mat[..., -1, :, :]
            kpt = rt_trans(rot_mat_W, kpt)
            print(rot_mat_W)

        return kpt


def local_pose_2_image_pose(local_pose, image_size=(224, 224), k_type="manopth"):
    # local_pose : (B, 21, 3) or (21, 3)
    if k_type == "manopth":
        KID = MANO_KID
    elif k_type == "hands19":
        KID = HANDS19_KID

    local_pose = local_pose.clone().float()

    x_axis = torch.cat(
        [
            local_pose[..., [KID["IM"]], :] - local_pose[..., [KID["MM"]], :],
            local_pose[..., [KID["MM"]], :] - local_pose[..., [KID["RM"]], :],
            local_pose[..., [KID["RM"]], :] - local_pose[..., [KID["PM"]], :],
            local_pose[..., [KID["IM"]], :] - local_pose[..., [KID["RM"]], :],
            local_pose[..., [KID["MM"]], :] - local_pose[..., [KID["PM"]], :],
            local_pose[..., [KID["IM"]], :] - local_pose[..., [KID["PM"]], :],
        ],
        axis=-2,
    )
    x_axis = norm_vec(x_axis)
    x_axis = x_axis.mean(-2)
    x_axis = norm_vec(x_axis)

    y_axis = torch.cat(
        [
            local_pose[..., [KID["W_"]], :] - local_pose[..., [KID["IM"]], :],
            local_pose[..., [KID["W_"]], :] - local_pose[..., [KID["MM"]], :],
            local_pose[..., [KID["W_"]], :] - local_pose[..., [KID["RM"]], :],
        ],
        axis=-2,
    )
    y_axis = norm_vec(y_axis)
    y_axis = y_axis.mean(-2)
    y_axis = norm_vec(y_axis)

    o_mm_P = local_pose[..., MANO_KID["MM"], :]

    z_axis = torch.cross(x_axis, y_axis)
    x_axis = torch.cross(y_axis, z_axis)

    o_mm_T = torch.zeros([local_pose.shape[0], 4, 4], device=local_pose.device, dtype=local_pose.dtype)
    o_mm_T[..., 3, 3] = 1
    o_mm_T[..., :3, 0] = x_axis
    o_mm_T[..., :3, 1] = y_axis
    o_mm_T[..., :3, 2] = z_axis
    o_mm_T[..., :3, 3] = o_mm_P

    mm_o_T = torch.linalg.inv(o_mm_T)
    mm_o_T = torch.nan_to_num(mm_o_T, nan=0.0)

    image_pose = rt_trans(mm_o_T, local_pose)
    image_pose[..., 0] = image_pose[..., 0] + image_size[0] // 2
    image_pose[..., 1] = image_pose[..., 1] + image_size[1] // 2

    return image_pose


def get_plane_from_3d_points(points):
    # points : (B, N, 3) or (N, 3)
    # return normal : (B, 3) or (3)
    if isinstance(points, np.ndarray):
        if len(points.shape) == 2:
            points = points.reshape(1, -1, 3)
        points = points - points.mean(axis=-2, keepdims=True)
        cov = np.matmul(points.transpose(0, 2, 1), points)
        _, _, v = np.linalg.svd(cov)
        normal = v[..., -1, :]
        return normal
    elif isinstance(points, torch.Tensor):
        if len(points.shape) == 2:
            points = points.reshape(1, -1, 3)
        points = points - points.mean(axis=-2, keepdims=True)
        cov = torch.matmul(points.transpose(1, 2), points)
        _, _, v = torch.linalg.svd(cov)
        normal = v[..., -1, :]
        return normal
    

def normal_to_rotation_angle(normal):
    # normal : (B, 3) or (3)
    # return pitch and yaw angle in degree
    if isinstance(normal, np.ndarray):
        if len(normal.shape) == 1:
            normal = normal.reshape(1, -1)
        normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
        pitch = np.arcsin(normal[..., 2])
        yaw = np.arctan2(normal[..., 1], normal[..., 0])
        pitch = pitch * 180 / np.pi
        yaw = yaw * 180 / np.pi
        return pitch, yaw
    elif isinstance(normal, torch.Tensor):
        if len(normal.shape) == 1:
            normal = normal.reshape(1, -1)
        normal = normal / torch.linalg.norm(normal, axis=-1, keepdims=True)
        pitch = torch.asin(normal[..., 2])
        yaw = torch.atan2(normal[..., 1], normal[..., 0])
        pitch = pitch * 180 / np.pi
        yaw = yaw * 180 / np.pi
        return pitch, yaw

def vector_to_roll_angle(vector):
    # vector : (B, 3) or (3)
    # return roll angle in degree
    if isinstance(vector, np.ndarray):
        if len(vector.shape) == 1:
            vector = vector.reshape(1, -1)
        vector = vector / np.linalg.norm(vector, axis=-1, keepdims=True)
        roll = np.arctan2(vector[..., 1], vector[..., 2]) * -1
        roll = roll * 180 / np.pi
        return roll
    elif isinstance(vector, torch.Tensor):
        if len(vector.shape) == 1:
            vector = vector.reshape(1, -1)
        vector = vector / torch.linalg.norm(vector, axis=-1, keepdims=True)
        roll = torch.atan2(vector[..., 1], vector[..., 2]) * -1
        roll = roll * 180 / np.pi
        return roll

def get_hand_angle_from_3d_pose(pose, ktype="manopth", is_uvd=True):
    # pose : (B, 21, 3) or (21, 3)
    if ktype == "manopth":
        KID = MANO_KID
    if is_uvd:
        pose = pose.copy()
        pose = pose[..., [2, 0, 1]]
        pose[..., [0, 1]] *= -1
    palm_points = pose[..., [KID["IM"], KID["MM"], KID["RM"], KID["PM"], KID["TM"]], :]
    palm_plane_normal = get_plane_from_3d_points(palm_points)
    palm_pitch, palm_yaw = normal_to_rotation_angle(palm_plane_normal)
    MM_to_wrist = pose[..., KID["W_"], :] - pose[..., KID["MM"], :]
    palm_roll = vector_to_roll_angle(MM_to_wrist)
    return palm_roll, palm_pitch, palm_yaw





    

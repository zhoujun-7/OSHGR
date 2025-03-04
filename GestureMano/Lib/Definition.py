import numpy as np
import torch

# MANO_ANGLE = np.zeros([14, 3, 778])

# MANO_ANGLE[0][0][250, 706] = 1
# MANO_ANGLE[0][1][253] = 1
# MANO_ANGLE[0][2][88] = 1

# MANO_ANGLE[1][0][724, 723] = 1
# MANO_ANGLE[1][1][708] = 1
# MANO_ANGLE[1][2][250, 706] = 1

# MANO_ANGLE[2][0][105] = 1
# MANO_ANGLE[2][1][231] = 1
# MANO_ANGLE[2][2][227] = 1

# MANO_ANGLE[3][0][212, 261] = 1
# MANO_ANGLE[3][1][274] = 1
# MANO_ANGLE[3][2][144, 145] = 1

# MANO_ANGLE[4][0][283, 86] = 1
# MANO_ANGLE[4][1][87] = 1
# MANO_ANGLE[4][2][212, 261] = 1

# MANO_ANGLE[5][0][176] = 1
# MANO_ANGLE[5][1][187] = 1
# MANO_ANGLE[5][2][369] = 1

# MANO_ANGLE[6][0][388, 399] = 1
# MANO_ANGLE[6][1][270] = 1
# MANO_ANGLE[6][2][220, 18] = 1

# MANO_ANGLE[7][0][405, 364] = 1
# MANO_ANGLE[7][1][365] = 1
# MANO_ANGLE[7][2][388, 399] = 1

# MANO_ANGLE[8][0][383] = 1
# MANO_ANGLE[8][1][276] = 1
# MANO_ANGLE[8][2][481] = 1

# MANO_ANGLE[9][0][498, 509] = 1
# MANO_ANGLE[9][1][291] = 1
# MANO_ANGLE[9][2][290, 183] = 1

# MANO_ANGLE[10][0][516, 476] = 1
# MANO_ANGLE[10][1][477] = 1
# MANO_ANGLE[10][2][498, 509] = 1

# MANO_ANGLE[11][0][163] = 1
# MANO_ANGLE[11][1][182] = 1
# MANO_ANGLE[11][2][593] = 1

# MANO_ANGLE[12][0][590, 627] = 1
# MANO_ANGLE[12][1][289, 202] = 1
# MANO_ANGLE[12][2][181] = 1

# MANO_ANGLE[13][0][628, 588] = 1
# MANO_ANGLE[13][1][587, 589] = 1
# MANO_ANGLE[13][2][590, 627] = 1

MANO_J_ID = {
    'W': 0,
    'IM': 1,
    'IP': 2,
    'ID': 3,
    'MM': 4,
    'MP': 5,
    'MD': 6,
    'PM': 7,
    'PP': 8,
    'PD': 9,
    'RM': 10,
    'RP': 11,
    'RD': 12,
    'TM': 13,
    'TP': 14,
    'TD': 15,

    'IT': 16,
    'MT': 17,
    'PT': 18,
    'RT': 19,
    'TT': 20,
}

SensorToManoPoseIndex = np.zeros((14, ), dtype=np.int)
SensorToManoPoseIndex[0] = 41
SensorToManoPoseIndex[1] = 44
SensorToManoPoseIndex[2] = 37
SensorToManoPoseIndex[3] = 2
SensorToManoPoseIndex[4] = 5
SensorToManoPoseIndex[5] = 1
SensorToManoPoseIndex[6] = 11
SensorToManoPoseIndex[7] = 14
SensorToManoPoseIndex[8] = 28
SensorToManoPoseIndex[9] = 29
SensorToManoPoseIndex[10] = 32
SensorToManoPoseIndex[11] = 19
SensorToManoPoseIndex[12] = 20
SensorToManoPoseIndex[13] = 23


# ANGLE_RANGE = np.zeros([14, 2])
# ANGLE_RANGE[0][1] = 90
# ANGLE_RANGE[1][1] = 90
# ANGLE_RANGE[2][1] = 50

# ANGLE_RANGE[3][1] = 90
# ANGLE_RANGE[6][1] = 90
# ANGLE_RANGE[9][1] = 90
# ANGLE_RANGE[12][1] = 90

# ANGLE_RANGE[4][1] = 110
# ANGLE_RANGE[7][1] = 110
# ANGLE_RANGE[10][1] = 110
# ANGLE_RANGE[13][1] = 100

# ANGLE_RANGE[5][1] = 20
# ANGLE_RANGE[8][1] = 20
# ANGLE_RANGE[11][1] = 20
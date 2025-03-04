import pickle
import numpy as np
# from tool.viz import show_pcd_index, show_pcd

class ManoLoader:
    def __init__(self, path):
        with open(path, 'rb') as f:
            mano = pickle.load(f, encoding='latin1')

        self.path = path
        self.mano = mano
        self.verts = mano['v_template']
        self.faces = mano['f']
        self._joint_mat = mano['J_regressor'].T.toarray() # (16, 3)
        self.pca_mat = mano['hands_components']

    def vertex2joint(self, vertex):
        # vertex: (778, 3)
        joint = self.joint_mat.T @ vertex
        joint = joint
        return joint

    def ang2pca(self, ang):
        pca = ang @ self.pca_mat.inv()
        return pca

    def pca2ang(self, pca):
        ang = pca @ self.pca_mat
        return ang


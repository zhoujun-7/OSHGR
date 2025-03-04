import torch
import torch.nn as nn
from Lib import ManoLayer


class CompleteAngle(nn.module):
    def __init__(self, mano):
        super(CompleteAngle, ).__init__()
        self.mano = mano
        self.mano_layer = ManoLayer(mano.path)
    
    def forward(self, ang):
        # ang : (B, 45)
        pca = self.mano.pca2ang(ang)
        loss = torch.pow(ang, 2).sum().sum()
        return loss

    def ang14_to_ang45(self, ):
        pass

    def trainer(self, ang14):
        pass
        
        optimizer = torch.optim.Adam()
        for i in range(10000):



    
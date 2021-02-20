"""
Implementation of the paper:
    Transformation Consistency Regularization- A Semi Supervised Paradigm for Image to Image Translation
    ECCV 2020

This file generated the Transformation Matrix that is being used to train TCR.
"""

import torch
import numpy as np
import kornia
import torch.nn as nn

class TCR(nn.Module):
    def __init__(self, angle, trans):
        super(TCR, self).__init__()

        self.ang = angle
        self.ang_neg = -1*self.ang
        self.max_tx, self.max_ty = trans, trans
        self.min_tx, self.min_ty = -trans, -trans

        # TODO
        self.max_z, self.min_z = 1.00, 1.00         # Change as per the task
        
    def forward(self, img, random, scale=1):
        b, c, h, w = img.shape

        # Rotation
        center = torch.Tensor([w/2, h/2]).unsqueeze(0)
        center = center.repeat(b, 1)
        angle = ((self.ang - self.ang_neg)*random[:,0]  + self.ang_neg)
        zoom = torch.ones(b, 2)
        T = kornia.get_rotation_matrix2d(center, angle, zoom)

        # Translation
        tx = ((self.max_tx - self.min_tx)*random[:,1] + self.min_tx) / scale
        ty = ((self.max_ty - self.min_ty)*random[:,2] + self.min_ty) / scale
        T[:,0,2] += tx
        T[:,1,2] += ty

        transformed = kornia.warp_affine(img, T.to('cuda'), dsize=(h, w))

        return transformed    
    
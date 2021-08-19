# Copyright (c) Facebook, Inc. and its affiliates.

import torch.nn as nn
import torch
import pdb
import numpy as np
from pytorch3d.renderer import (
    look_at_view_transform, TexturesVertex
)
import math
from pytorch3d.structures import Meshes
import cv2
import matplotlib.pyplot as plt
from utils import rotation_matrix
from scipy.ndimage.filters import gaussian_filter1d
from pytorch3d.io import save_obj
from pytorch3d.transforms import (
    RotateAxisAngle, matrix_to_euler_angles
)
from pytorch3d.transforms.rotation_conversions import (
    rotation_6d_to_matrix, matrix_to_rotation_6d
)
from utils import (
    flow_to_image, flow_confidence
)
import time
from matplotlib.image import imsave
import os
from torch.autograd import Variable
from pytorch3d.transforms import (
    euler_angles_to_matrix
)
from utils import (
    rotation_matrix
)



class JOHMRLite(nn.Module):

    def __init__(self, obj_verts, obj_faces, vis_render, img_h, img_w):

        super().__init__()
        self.device = obj_verts.device
        self.vis_render = vis_render
        self.obj_verts = obj_verts.detach()
        self.obj_faces = obj_faces.detach()
        self.img_h = img_h
        self.img_w = img_w

        # camera is almost at the center (distance can't be zero for diff render)
        self.R, self.T = look_at_view_transform(0.1, 0.0, 0.0,device=self.device)
        self.T[0,2] = 0.0  # manually set to zero


        return


    def forward(self, obj_size, x_offset, y_offset, z_offset, yaw, pitch, roll):

        obj_verts = self.obj_verts.clone()
        # step 1: rescale object
        x_diff = torch.max(obj_verts[:,0]) - torch.min(obj_verts[:,0])
        x_ratio = float(obj_size[0]) / x_diff
        y_diff = torch.max(obj_verts[:,1]) - torch.min(obj_verts[:,1])
        y_ratio = float(obj_size[1]) / y_diff
        z_diff = torch.max(obj_verts[:,2]) - torch.min(obj_verts[:,2])
        z_ratio = float(obj_size[2]) / z_diff
        obj_verts[:, 0] *=  x_ratio
        obj_verts[:, 1] *=  y_ratio
        obj_verts[:, 2] *=  z_ratio

        # step 2: part motion
        #part_state = torch.tensor(90 * (math.pi/180)).cuda()
        #axis = torch.tensor([0, -0.9999999999999999, -0]).cuda().float()
        #rot_o = torch.tensor([0.37487859368179954*x_ratio, -0.859491*y_ratio, -0.24141621508844158*z_ratio]).cuda()
        #assert(part_state>=0)  # non negative value
        #start = 380
        #end = 380+198
        #partrot_mat = rotation_matrix(axis, part_state).cuda() # part rotation matrix
        #obj_verts_part = obj_verts[start:end, :] - rot_o
        #obj_verts_part2 =  torch.mm(partrot_mat, obj_verts_part.permute(1,0)).permute(1,0)
        #obj_verts[start:end, :] = obj_verts_part2 + rot_o


        # step 3: object orientation
        euler_angle = torch.tensor([pitch, yaw, roll]).reshape(1,3)
        objrot_mat = euler_angles_to_matrix(euler_angle, ["X","Y","Z"]).to(self.device)
        rot_alpha, rot_beta, rot_gamma = matrix_to_euler_angles(objrot_mat, ["X","Y","Z"])[0]
        rot_alpha = float(rot_alpha)
        rot_beta = float(rot_beta)
        rot_gamma = float(rot_gamma)
        objrot_mat = objrot_mat[0]
        obj_verts = torch.mm(objrot_mat, obj_verts.permute(1,0)).permute(1,0)

        # step 4: object offset
        obj_verts[:, 0] += 100.0*x_offset
        obj_verts[:, 1] += 100.0*y_offset
        obj_verts[:, 2] += 100.0*z_offset

        self.obj_verts_output = obj_verts.clone()

        obj_verts[:,1:] *= -1
        # create object mesh for diff render and visualization
        tex = torch.ones_like(obj_verts).unsqueeze(0)
        textures = TexturesVertex(verts_features=tex).to(self.device)
        self.obj_mesh = Meshes(verts=[obj_verts],faces=[self.obj_faces],textures=textures)
        vis_image = self.vis_render(meshes_world=self.obj_mesh, R=self.R, T=self.T)

        return vis_image[...,:3], rot_alpha, rot_beta, rot_gamma




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

    def __init__(self, obj_verts, obj_faces, vis_render, rot_o, axis, vertexSegs, faceSegs, obj_size, gt_x, gt_y, gt_z, R):

        super().__init__()
        self.device = obj_verts.device
        self.vis_render = vis_render
        self.obj_verts = obj_verts.detach()
        self.obj_faces = obj_faces.detach()
        self.rot_origs = rot_o
        self.rot_axises = axis
        self.vertexSegs = vertexSegs
        self.faceSegs = faceSegs
        self.obj_size = obj_size
        self.x_offset = gt_x
        self.y_offset = gt_y
        self.z_offset = gt_z
        self.objR = R[0]

        x_diff = torch.max(obj_verts[:,0]) - torch.min(obj_verts[:,0])
        self.x_ratio = float(obj_size[0]) / x_diff
        y_diff = torch.max(obj_verts[:,1]) - torch.min(obj_verts[:,1])
        self.y_ratio = float(obj_size[1]) / y_diff
        z_diff = torch.max(obj_verts[:,2]) - torch.min(obj_verts[:,2])
        self.z_ratio = float(obj_size[2]) / z_diff

        # camera is almost at the center (distance can't be zero for diff render)
        self.R, self.T = look_at_view_transform(0.1, 0.0, 0.0,device=self.device)
        self.T[0,2] = 0.0  # manually set to zero


        return


    def forward(self, partrot, part_idx):

        # predefined object CAD part and axis
        self.vertexStart = self.vertexSegs[part_idx]
        self.vertexEnd = self.vertexSegs[part_idx+1]
        faceStart = self.faceSegs[part_idx]
        faceEnd = self.faceSegs[part_idx+1]
        self.rot_o = self.rot_origs[part_idx].clone()
        self.axis = self.rot_axises[part_idx].clone()

        # pytorch3d -> world coordinate
        self.rot_o[1:] *= -1
        self.axis[1:] *= -1

        # step 1: rescale object
        obj_verts = self.obj_verts.clone()
        obj_verts[:, 0] *=  self.x_ratio
        obj_verts[:, 1] *=  self.y_ratio
        obj_verts[:, 2] *=  self.z_ratio
        self.rot_o[0] *= self.x_ratio
        self.rot_o[1] *= self.y_ratio
        self.rot_o[2] *= self.z_ratio

        # step 2: part motion
        partrot_radian = torch.tensor(partrot*0.0174533)
        part_rot_mat = rotation_matrix(self.axis, partrot_radian)
        obj_verts_t1 = obj_verts[self.vertexStart:self.vertexEnd, :] - self.rot_o
        obj_verts_t2 = torch.mm(part_rot_mat.to(self.device), obj_verts_t1.permute(1,0)).permute(1,0)
        obj_verts[self.vertexStart:self.vertexEnd, :] = obj_verts_t2 + self.rot_o

        # step 3: object orientation
        obj_verts = torch.mm(self.objR.to(self.device), obj_verts.permute(1,0)).permute(1,0)

        # step 4: object offset
        obj_verts[:, 0] += 100.0*self.x_offset
        obj_verts[:, 1] += 100.0*self.y_offset
        obj_verts[:, 2] += 100.0*self.z_offset

        obj_verts[:,1:] *= -1
        # create object mesh for diff render and visualization
        tex = torch.ones_like(obj_verts).unsqueeze(0)
        textures = TexturesVertex(verts_features=tex).to(self.device)
        self.obj_mesh = Meshes(verts=[obj_verts],faces=[self.obj_faces],textures=textures)
        vis_image, _ = self.vis_render(meshes_world=self.obj_mesh, R=self.R, T=self.T)
        silhouette = vis_image[0,:,:,3]
        silhouette[silhouette>0] = 1

        return silhouette.detach().cpu().numpy()

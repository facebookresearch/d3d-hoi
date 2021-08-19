# Copyright (c) Facebook, Inc. and its affiliates.
import torch.nn as nn
import torch
import numpy as np
from pytorch3d.renderer import (
    look_at_view_transform, TexturesVertex
)
from pytorch3d.structures import Meshes
from utils import rotation_matrix
from pytorch3d.io import save_obj
from pytorch3d.transforms import (
    RotateAxisAngle, matrix_to_euler_angles
)
from pytorch3d.transforms.rotation_conversions import (
    rotation_6d_to_matrix, matrix_to_rotation_6d
)
import os
from pytorch3d.transforms import (
    euler_angles_to_matrix
)
from utils import (
    rotation_matrix
)



class JOHMRLite(nn.Module):

    def __init__(self, x_offset, y_offset, z_offset, yaw, pitch, roll, part_motion, obj_size, \
                 obj_verts, obj_faces, vis_render, part_idx, rot_o, axis, vertexSegs, rot_type):

        super().__init__()
        self.device = obj_verts.device
        self.vis_render = vis_render
        self.obj_verts = obj_verts.detach()
        self.obj_faces = obj_faces.detach()
        self.rot_type = rot_type
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z_offset = z_offset 
        self.part_motion = part_motion 

        # camera is almost at the center (distance can't be zero for diff render)
        self.R, self.T = look_at_view_transform(0.1, 0.0, 0.0,device=self.device)
        self.T[0,2] = 0.0  # manually set to zero

        x_diff = torch.max(obj_verts[:,0]) - torch.min(obj_verts[:,0])
        self.x_ratio = float(obj_size[0]) / x_diff
        y_diff = torch.max(obj_verts[:,1]) - torch.min(obj_verts[:,1])
        self.y_ratio = float(obj_size[1]) / y_diff
        z_diff = torch.max(obj_verts[:,2]) - torch.min(obj_verts[:,2])
        self.z_ratio = float(obj_size[2]) / z_diff

        # predefined object CAD part and axis
        self.vertexStart = vertexSegs[part_idx]
        self.vertexEnd = vertexSegs[part_idx+1]
        self.rot_o = rot_o[part_idx]
        self.axis = axis[part_idx]

        # pytorch3d -> world coordinate
        self.rot_o[1:] *= -1
        self.axis[1:] *= -1

        # rescale object
        self.obj_verts[:, 0] *=  self.x_ratio
        self.obj_verts[:, 1] *=  self.y_ratio
        self.obj_verts[:, 2] *=  self.z_ratio
        self.rot_o[0] *= self.x_ratio
        self.rot_o[1] *= self.y_ratio
        self.rot_o[2] *= self.z_ratio

        euler_angle = torch.tensor([pitch, yaw, roll]).reshape(1,3)
        self.objR = euler_angles_to_matrix(euler_angle, ["X","Y","Z"]).to(self.device)[0]

        return



    def forward(self, index):

        partmotion = self.part_motion[index]
        obj_verts = self.obj_verts.clone()
        
        # part motion
        if self.rot_type[0] == 'prismatic':
            part_state = torch.tensor(partmotion).to(self.device)
            obj_verts_t1 = obj_verts[self.vertexStart:self.vertexEnd, :] - self.rot_o
            obj_verts_t2 = obj_verts_t1 + self.axis * part_state  #/float(annotation['obj_dim'][2]) * z_ratio
            obj_verts[self.vertexStart:self.vertexEnd, :] = obj_verts_t2 + self.rot_o

        else:
            part_state = torch.tensor(partmotion*0.0174533)
            part_rot_mat = rotation_matrix(self.axis, part_state)
            obj_verts_t1 = obj_verts[self.vertexStart:self.vertexEnd, :] - self.rot_o
            obj_verts_t2 = torch.mm(part_rot_mat.to(self.device), obj_verts_t1.permute(1,0)).permute(1,0)
            obj_verts[self.vertexStart:self.vertexEnd, :] = obj_verts_t2 + self.rot_o
              
        # step 3: object orientation
        obj_verts = torch.mm(self.objR, obj_verts.permute(1,0)).permute(1,0)

        # step 4: object offset
        obj_verts[:, 0] += 100.0*self.x_offset
        obj_verts[:, 1] += 100.0*self.y_offset
        obj_verts[:, 2] += 100.0*self.z_offset

        obj_verts[:,1:] *= -1
        # create object mesh for diff render and visualization
        tex = torch.ones_like(obj_verts).unsqueeze(0)
        tex[:, :, 0] = 0
        tex[:, :, 1] = 1
        tex[:, :, 2] = 0
        textures = TexturesVertex(verts_features=tex).to(self.device)
        self.obj_mesh = Meshes(verts=[obj_verts],faces=[self.obj_faces],textures=textures)
        vis_image = self.vis_render(meshes_world=self.obj_mesh, R=self.R, T=self.T)
        silhouette = vis_image[0,:,:,:3]  

        return silhouette.detach().cpu().numpy()
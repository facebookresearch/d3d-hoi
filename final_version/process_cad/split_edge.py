import math
import os
import torch
import numpy as np
from tqdm import tqdm_notebook
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import pdb
# io utils
from pytorch3d.io import load_obj, save_obj

# datastructures
from pytorch3d.structures import Meshes, Textures

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights
)





threshold = 0.02
verts, faces_idx, _ = load_obj('8.obj')
faces = faces_idx.verts_idx
verts_size = verts.shape[0]
faces_size = faces.shape[0]


while True:
    all_right = 1
    # iterative edge split 
    for count, f in enumerate(faces):

        # three points of the face triangle
        f = faces[count]
        f_1 = verts[f[0]]
        f_2 = verts[f[1]]
        f_3 = verts[f[2]]

        # compute area using cross product 
        f_area = 0.5*torch.norm(torch.cross(f_2 - f_1, f_3 - f_1))

        # subdive face 
        if f_area > threshold:

            # compute sides, pick largest one
            l1 = torch.norm(f_2 - f_1)
            l2 = torch.norm(f_3 - f_1)
            l3 = torch.norm(f_3 - f_2)
            max_side = torch.max(torch.FloatTensor([l1,l2,l3]))
            if l1 == max_side:
                f_mid = (f_1 + f_2)/2.0
                new_f_1 = torch.LongTensor([f[1], f[2], verts_size])
                new_f_2 = torch.LongTensor([f[0], f[2], verts_size])
            elif l2 == max_side:
                f_mid = (f_1 + f_3)/2.0
                new_f_1 = torch.LongTensor([f[0], f[1], verts_size])
                new_f_2 = torch.LongTensor([f[1], f[2], verts_size])
            else:
                assert l3 == max_side
                f_mid = (f_2 + f_3)/2.0
                new_f_1 = torch.LongTensor([f[0], f[1], verts_size])
                new_f_2 = torch.LongTensor([f[0], f[2], verts_size])

            verts = torch.cat((verts, f_mid.reshape(1,3)),0)
            faces[count] = new_f_1
            faces = torch.cat((faces, new_f_2.reshape(1,3)),0)

            
            # compute mid-point
            #f_mid = (f_1+f_2+f_3)/3.0

            # add new mid-point to verts
            #verts = torch.cat((verts, f_mid.reshape(1,3)),0)

            # add new face to faces
            #new_f_1 = torch.LongTensor([f[0], f[1], verts_size])
            #new_f_2 = torch.LongTensor([f[0], f[2], verts_size])
            #new_f_3 = torch.LongTensor([f[1], f[2], verts_size])
            #faces[count] = new_f_1
            #faces = torch.cat((faces, new_f_2.reshape(1,3)),0)
            #faces = torch.cat((faces, new_f_3.reshape(1,3)),0)
            
            verts_size += 1

            all_right = 0

    if all_right:
        break

print(verts_size)
save_obj('./test1.obj', verts, faces)


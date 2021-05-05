from pytorch3d.utils import ico_sphere

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
from pytorch3d.io import load_obj

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


# Set the cuda device 
device = torch.device("cuda:0")
torch.cuda.set_device(device)

# Initialize an OpenGL perspective camera.
cameras = OpenGLPerspectiveCameras(device=device)

# To blend the 100 faces we set a few parameters which control the opacity and the sharpness of 
# edges. Refer to blending.py for more details. 
blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 256x256. To form the blended image we use 100 faces for each pixel. We also set bin_size and max_faces_per_bin to None which ensure that 
# the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
# explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
# the difference between naive and coarse-to-fine rasterization. 
raster_settings = RasterizationSettings(
    image_size=256, 
    blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
    faces_per_pixel=100, 
)

# Create a silhouette mesh renderer by composing a rasterizer and a shader. 
silhouette_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftSilhouetteShader(blend_params=blend_params)
)


# We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
raster_settings = RasterizationSettings(
    image_size=256, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)
# We can add a point light in front of the object. 
lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
phong_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
)



def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])



# Initialize two ico spheres of different sizes
verts1, faces_idx1, _ = load_obj("./11124/textured_objs/new-1.obj")
faces1 = faces_idx1.verts_idx
verts2, faces_idx2, _ = load_obj("./11124/textured_objs/new-3.obj")
faces2 = faces_idx2.verts_idx

# world coordinate to local coordinate (rotation origin)
rot_o = [0.5487992993753168, 0.4783358535896229, -0.5365869414619843]
verts1[:, 0] -= rot_o[0]
verts1[:, 1] -= rot_o[1]
verts1[:, 2] -= rot_o[2]

# rotate around local axis [-1 0 0]
axis = [-1, 0, 0]
theta =0.6
rot_mat = torch.from_numpy(rotation_matrix(axis, theta)).float()  # 3x3
verts1 = torch.t(torch.mm(rot_mat, torch.t(verts1)))

# local coordinate to world coordinate
verts1[:, 0] += rot_o[0]
verts1[:, 1] += rot_o[1]
verts1[:, 2] += rot_o[2]

#mesh_origin = ico_sphere(3)
#verts_origin, faces_origin = mesh_origin.get_mesh_verts_faces(0)
#verts_origin *= 0.1
#verts_origin[:,0] += 0.5487992993753168
#verts_origin[:,1] += 0.4783358535896229
#verts_origin[:,2] += -0.5365869414619843



# Initalize the textures as an RGB color per vertex
tex1 = torch.ones_like(verts1)
tex2 = torch.ones_like(verts2)
#tex3 = torch.ones_like(verts_origin)
tex1[:, 1:] *= 0.0  # red
tex2[:, :2] *= 0.0  # blue
#tex3[:, 1:] *= 0.1
#tex3[:, :2] = 0.7  


# Create one mesh which contains two spheres of different sizes.
# To do this we can concatenate verts1 and verts2
# but we need to offset the face indices of faces2 so they index
# into the correct positions in the combined verts tensor. 

verts = torch.cat([verts1, verts2]).to(device)  #(204, 3)

#  Offset by the number of vertices in mesh1
faces2 = faces2 + verts1.shape[0]
#faces_origin = faces_origin + faces2.shape[0]
faces = torch.cat([faces1, faces2]).to(device)  # (400, 3)

tex = torch.cat([tex1, tex2])[None]  # (1, 204, 3)
textures = Textures(verts_rgb=tex).to(device)

mesh = Meshes(verts=[verts], faces=[faces], textures=textures)



distance = 5   # distance from camera to the object
elevation = 25.0   # angle of elevation in degrees
azimuth = 60.0  # No rotation so the camera is positioned on the +Z axis. 

# Get the position of the camera based on the spherical angles
R, T = look_at_view_transform(distance, elevation, azimuth, device=device)

# Render the teapot providing the values of R and T. 
silhouete = silhouette_renderer(meshes_world=mesh, R=R, T=T)
image_ref = phong_renderer(meshes_world=mesh, R=R, T=T)

silhouete = silhouete.cpu().numpy()
image_ref = image_ref.cpu().numpy()

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(silhouete.squeeze()[..., 3])  # only plot the alpha channel of the RGBA image
plt.grid(False)
plt.subplot(1, 2, 2)
plt.imshow(image_ref.squeeze())
plt.grid(False)
plt.show()
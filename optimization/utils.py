# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
import natsort
import glob
import open3d as o3d
# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras,RasterizationSettings,
    MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, DirectionalLights,
    PerspectiveCameras
)
from pytorch3d.io import save_obj, load_obj
import math
import cv2
import matplotlib.pyplot as plt
import os
import imageio
from decimal import Decimal
import json
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d
from scipy.ndimage.filters import gaussian_filter1d
from numpy.linalg import svd
from multiprocessing import Pool, Manager, cpu_count
from pytorch3d.transforms import Rotate, Translate
from matplotlib.image import imsave
from pathlib import Path


def planeFit(points):
    """
    p, n = planeFit(points)
    Given an array, points, of shape (d,...)
    representing points in d-dimensional space,
    fit an d-dimensional plane to the points.
    Return a point, p, on the plane (the point-cloud centroid),
    and the normal, n.
    """
    points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
    ctr = points.mean(axis=1)
    x = points - ctr[:,np.newaxis]
    M = np.dot(x, x.T) # Could also use np.cov(x) here.
    return ctr, svd(M)[0][:,-1]



def initialize_render(device, focal_x, focal_y, img_square_size, img_small_size):
    """ initialize camera, rasterizer, and shader. """
    # Initialize an OpenGL perspective camera.
    #cameras = FoVPerspectiveCameras(znear=1.0, zfar=9000.0, fov=20, device=device)
    #cameras = FoVPerspectiveCameras(device=device)
    #cam_proj_mat = cameras.get_projection_transform()
    img_square_center = int(img_square_size/2)
    shrink_ratio = int(img_square_size/img_small_size)
    focal_x_small = int(focal_x/shrink_ratio)
    focal_y_small = int(focal_y/shrink_ratio)
    img_small_center = int(img_small_size/2)

    camera_sfm = PerspectiveCameras(
                focal_length=((focal_x, focal_y),),
                principal_point=((img_square_center, img_square_center),),
                image_size = ((img_square_size, img_square_size),),
                device=device)

    camera_sfm_small = PerspectiveCameras(
                focal_length=((focal_x_small, focal_y_small),),
                principal_point=((img_small_center, img_small_center),),
                image_size = ((img_small_size, img_small_size),),
                device=device)

    # To blend the 100 faces we set a few parameters which control the opacity and the sharpness of
    # edges. Refer to blending.py for more details.
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 256x256. To form the blended image we use 100 faces for each pixel. We also set bin_size and max_faces_per_bin to None which ensure that
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
    # the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=img_small_size,
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
        faces_per_pixel=50,
    )

    # Create a silhouette mesh renderer by composing a rasterizer and a shader.
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera_sfm_small,
            raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )


    # We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=img_square_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # We can add a point light in front of the object.
    lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
    #lights = DirectionalLights(device=device, direction=((0, 0, 1),))
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera_sfm,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(device=device, cameras=camera_sfm, lights=lights)
    )

    return silhouette_renderer, phong_renderer


def merge_meshes(obj_path):
    """ helper function for loading and merging meshes. """
    verts_list = torch.empty(0,3)
    faces_list = torch.empty(0,3).long()
    num_vtx = [0]
    num_faces = [0]

    # merge meshes, load in ascending order
    meshes = natsort.natsorted(glob.glob(obj_path+'/final/*_rescaled_sapien.obj'))
    
    for part_mesh in meshes:
        verts, faces, aux = load_obj(part_mesh)
        faces = faces.verts_idx
        faces = faces + verts_list.shape[0]
        verts_list = torch.cat([verts_list, verts])
        faces_list = torch.cat([faces_list, faces])
        num_vtx.append(verts_list.shape[0])
        num_faces.append(faces_list.shape[0])

    return verts_list, faces_list, num_vtx, num_faces



def load_motion(motions, device):
    """ load rotation axis, origin, and limit. """
    rot_origin = []
    rot_axis = []
    rot_type = []
    limit_a = []
    limit_b = []
    contact_list = []

    # load all meta data
    for idx, key in enumerate(motions.keys()):
        jointData = motions[key]

        # if contains movable parts
        if jointData is not None:
            origin = torch.FloatTensor(jointData['axis']['origin']).to(device)
            axis = torch.FloatTensor(jointData['axis']['direction']).to(device)
            mobility_type = jointData['type']
            contact_list.append(jointData['contact'])

            # convert to radians if necessary
            if mobility_type == 'revolute':
                mobility_a = math.pi*jointData['limit']['a'] / 180.0
                mobility_b = math.pi*jointData['limit']['b'] / 180.0
            else:
                assert mobility_type == 'prismatic'
                mobility_a = jointData['limit']['a']
                mobility_b = jointData['limit']['b']

            rot_origin.append(origin)
            rot_axis.append(axis)
            rot_type.append(mobility_type)
            limit_a.append(mobility_a)
            limit_b.append(mobility_b)

    return rot_origin, rot_axis, rot_type, limit_a, limit_b, contact_list



def save_object(id):
    global obj_verts_dict
    global obj_faces_dict
    global save_path_mesh
    verts = obj_verts_dict[str(id+1)]
    faces = obj_faces_dict[str(id+1)]
    path = os.path.join(save_path_mesh, str(id+1)+'_object.obj')
    save_obj(path, torch.from_numpy(verts), torch.from_numpy(faces))
    

def save_human(id):
    global human_verts_dict
    global human_faces_dict
    global save_path_mesh
    verts = human_verts_dict[str(id+1)]
    faces = human_faces_dict[str(id+1)]
    path = os.path.join(save_path_mesh, str(id+1)+'_person.obj')
    save_obj(path, torch.from_numpy(verts), torch.from_numpy(faces))
   

def save_meshes(meshes, save_folder, video_name, title):
    global obj_verts_dict
    global obj_faces_dict
    global human_verts_dict
    global human_faces_dict
    global save_path_mesh

    save_path_mesh = os.path.join(save_folder, title)
    if not os.path.exists(save_path_mesh):
        os.makedirs(save_path_mesh)

    obj_meshes = meshes['obj_mesh']
    spin_meshes = meshes['spin_mesh']

    #  merge object + SPIN meshes
    obj_verts = {}
    obj_faces = {}
    human_verts = {}
    human_faces = {}

    for idx in range(len(obj_meshes)):
        path = os.path.join(save_path_mesh, str(idx+1)+'_person.obj')
        save_obj(path, spin_meshes[idx].verts_list()[0], spin_meshes[idx].faces_list()[0])
        path = os.path.join(save_path_mesh, str(idx+1)+'_object.obj')
        save_obj(path, obj_meshes[idx].verts_list()[0], obj_meshes[idx].faces_list()[0])
    
    eft_cmd = 'python -m demo.demo_bodymocapnewnew --render solid --videoname '+video_name+' --vPath '+save_folder
    os.chdir('/local-scratch/projects/d3dhoi/eft')
    os.system(eft_cmd)
    os.chdir('/local-scratch/projects/d3dhoi')
    '''
    save_path = os.path.join(save_folder, 'eft', 'front')
    ffmpeg_cmd = 'ffmpeg -r 3 -i '+save_path+'/scene_%08d.jpg '+save_folder+'/frontview.mp4'
    os.system(ffmpeg_cmd)
    '''
    
    return



def save_parameters(model, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    obj_offset = model.obj_offset.detach().cpu().numpy()
    x_dim =  model.x_dim.item() 
    y_dim =  model.y_dim.item()     
    z_dim =  model.z_dim.item()     
    obj_rot_angle = model.obj_rot_angle.detach().cpu().numpy()     #(3,3)
    part_motion = model.part_motion.detach().cpu().numpy()
    
    parameters = {}
    parameters['obj_offset'] = obj_offset
    parameters['obj_dim'] = [x_dim, y_dim, z_dim]
    parameters['obj_rot_angle'] = obj_rot_angle
    parameters['part_motion'] = part_motion

    np.save(os.path.join(save_path, 'params.npy'), parameters)
    return



def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = axis / torch.sqrt(torch.dot(axis, axis))
    a = torch.cos(theta / 2.0)
    b, c, d = -axis * torch.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rot_mat = torch.empty(3,3)

    rot_mat[0,0] = aa + bb - cc - dd
    rot_mat[0,1] = 2 * (bc + ad)
    rot_mat[0,2] = 2 * (bd - ac)

    rot_mat[1,0] = 2 * (bc - ad)
    rot_mat[1,1] = aa + cc - bb - dd
    rot_mat[1,2] = 2 * (cd + ab)

    rot_mat[2,0] = 2 * (bd + ac)
    rot_mat[2,1] = 2 * (cd - ab)
    rot_mat[2,2] = aa + dd - bb - cc

    return rot_mat


def rotation_matrix_batch(axis, theta, device):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = axis / torch.sqrt(torch.dot(axis, axis))
    a = torch.cos(theta / 2.0)
    b = -axis[0] * torch.sin(theta / 2.0)
    c = -axis[1] * torch.sin(theta / 2.0)
    d = -axis[2] * torch.sin(theta / 2.0)

    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

    rot_mat = torch.empty(aa.shape[0],3,3).to(device)

    rot_mat[:,0,0] = aa + bb - cc - dd
    rot_mat[:,0,1] = 2 * (bc + ad)
    rot_mat[:,0,2] = 2 * (bd - ac)

    rot_mat[:,1,0] = 2 * (bc - ad)
    rot_mat[:,1,1] = aa + cc - bb - dd
    rot_mat[:,1,2] = 2 * (cd + ab)

    rot_mat[:,2,0] = 2 * (bd + ac)
    rot_mat[:,2,1] = 2 * (cd - ab)
    rot_mat[:,2,2] = aa + dd - bb - cc

    return rot_mat


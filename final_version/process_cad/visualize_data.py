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
import glob
import natsort
from torch.autograd import Variable
import trimesh
import copy
import re

# io utils
from pytorch3d.io import load_obj, save_obj, save_ply, load_ply

# datastructures
from pytorch3d.structures import Meshes

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, HardFlatShader, DirectionalLights, cameras
)

import json
import csv
import open3d as o3d

device = torch.device("cuda:0")
torch.cuda.set_device(device)


# helper function for computing roation matrix in 3D
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



# helper function for loading and merging meshes
def merge_meshes(obj_path):
    verts_list = torch.empty(0,3)
    faces_list = torch.empty(0,3).long()
    num_vtx = [0]

    # merge meshes, load in ascending order
    meshes = natsort.natsorted(glob.glob(obj_path+'/final/*_rescaled_sapien.obj'))

    for part_mesh in meshes:
        print('loading %s' %part_mesh)
        mesh = o3d.io.read_triangle_mesh(part_mesh)
        verts = torch.from_numpy(np.asarray(mesh.vertices)).float()
        faces = torch.from_numpy(np.asarray(mesh.triangles)).long()
        faces = faces + verts_list.shape[0]
        verts_list = torch.cat([verts_list, verts])
        faces_list = torch.cat([faces_list, faces])
        num_vtx.append(verts_list.shape[0])

    verts_list = verts_list.to(device)
    faces_list = faces_list.to(device)

    return verts_list, faces_list, num_vtx



cad_folder = '/home/xuxiangx/research/johmr/dataset/final'
cad_classes = [f.name for f in os.scandir(cad_folder)]

for cad_category in cad_classes:

    folder_path = os.path.join(cad_folder, cad_category)
    object_paths = [f.path for f in os.scandir(folder_path)]

    for obj_path in object_paths:

        print('processing %s' % obj_path)

        # load merged mesh and number of vtx for each part
        verts_list, faces_list, num_vtx = merge_meshes(obj_path)

        # load motion json file
        with open(os.path.join(obj_path, 'motion.json')) as json_file:
            motion = json.load(json_file)

        # create gif writer
        filename_output = os.path.join(obj_path, 'motion.gif')
        writer = imageio.get_writer(filename_output, mode='I', duration=0.3)
        vis = o3d.visualization.Visualizer()
        vis.create_window(height=500, width=500)

        distance = 2.4   # distance from camera to the object
        elevation = 25   # angle of elevation in degrees
        azimuth = 20  # No rotation so the camera is positioned on the +Z axis.

        # at least render one frame
        if len(motion) == 0:
            motion['placeholder'] = {}

        # rotate or translate individual part
        for idx, key in enumerate(motion.keys()):

            jointData = motion[key]

            # rotation part
            if jointData and jointData['type'] == 'revolute':

                start = num_vtx[idx]
                end = num_vtx[idx+1]

                rot_orig = torch.FloatTensor(jointData['axis']['origin']).to(device)
                rot_axis = torch.FloatTensor(jointData['axis']['direction']).to(device)

                aa = math.pi*jointData['limit']['a'] / 180.0
                bb = math.pi*jointData['limit']['b'] / 180.0
                print(aa)
                print(bb)

                rot_angles = np.linspace(aa, bb, num=5)
                rot_angles_rev = np.linspace(bb, aa, num=5)
                angles = np.concatenate((rot_angles, rot_angles_rev),0)

                for angle in angles:

                    verts = verts_list.clone()
                    faces = faces_list.clone()

                    # world coordinate to local coordinate (rotation origin)
                    verts[start:end, 0] -= rot_orig[0]
                    verts[start:end, 1] -= rot_orig[1]
                    verts[start:end, 2] -= rot_orig[2]

                    # rotate around local axis [-1 0 0]
                    init_value = torch.tensor([angle])
                    theta = Variable(init_value.cuda())
                    rot_mat = rotation_matrix(rot_axis, theta).float()  # 3x3
                    verts[start:end,:] = torch.t(torch.mm(rot_mat.to(device),
                                                 torch.t(verts[start:end,:])))

                    # local coordinate to world coordinate
                    verts[start:end, 0] += rot_orig[0]
                    verts[start:end, 1] += rot_orig[1]
                    verts[start:end, 2] += rot_orig[2]

                    R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
                    T = Translate(T, device=T.device)
                    R = Rotate(R, device=R.device)
                    MM = R.compose(T)

                    opt_mesh = o3d.geometry.TriangleMesh()

                    # transform
                    tmp = MM.transform_points(verts).detach().cpu().numpy()
                    tmp[:,0] *= -1
                    tmp[:,2] *= -1

                    # visualize
                    opt_mesh.vertices = o3d.utility.Vector3dVector(tmp)
                    opt_mesh.triangles = o3d.utility.Vector3iVector(faces_list.cpu().numpy())
                    opt_mesh.compute_vertex_normals()
                    vis.clear_geometries()
                    vis.add_geometry(opt_mesh)
                    vis.poll_events()
                    img = np.asarray(vis.capture_screen_float_buffer(True))
                    image = img_as_ubyte(img)
                    writer.append_data(image)


            # translation part
            elif jointData and jointData['type'] == 'prismatic':

                start = num_vtx[idx]
                end = num_vtx[idx+1]

                trans_orig = torch.FloatTensor(jointData['axis']['origin']).to(device)
                trans_axis = torch.FloatTensor(jointData['axis']['direction']).to(device)

                aa = jointData['limit']['a']
                bb = jointData['limit']['b']

                trans_len = np.linspace(aa, bb, num=5)
                trans_len_rev = np.linspace(bb, aa, num=5)
                trans_lens = np.concatenate((trans_len, trans_len_rev),0)

                for tran_len in trans_lens:

                    verts = verts_list.clone()
                    faces = faces_list.clone()

                    # world coordinate to local coordinate (rotation origin)
                    verts[start:end, 0] -= trans_orig[0]
                    verts[start:end, 1] -= trans_orig[1]
                    verts[start:end, 2] -= trans_orig[2]

                    # add value in translation direction
                    verts[start:end, 0] += (trans_axis[0] * tran_len)
                    verts[start:end, 1] += (trans_axis[1] * tran_len)
                    verts[start:end, 2] += (trans_axis[2] * tran_len)

                    # local coordinate to world coordinate
                    verts[start:end, 0] += trans_orig[0]
                    verts[start:end, 1] += trans_orig[1]
                    verts[start:end, 2] += trans_orig[2]

                    R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
                    T = Translate(T, device=T.device)
                    R = Rotate(R, device=R.device)
                    MM = R.compose(T)

                    opt_mesh = o3d.geometry.TriangleMesh()

                    # transform
                    tmp = MM.transform_points(verts).detach().cpu().numpy()
                    tmp[:,0] *= -1
                    tmp[:,2] *= -1

                    # visualize
                    opt_mesh.vertices = o3d.utility.Vector3dVector(tmp)
                    opt_mesh.triangles = o3d.utility.Vector3iVector(faces_list.cpu().numpy())
                    opt_mesh.compute_vertex_normals()
                    vis.clear_geometries()
                    vis.add_geometry(opt_mesh)
                    vis.poll_events()
                    img = np.asarray(vis.capture_screen_float_buffer(True))
                    image = img_as_ubyte(img)
                    writer.append_data(image)

            # no motion
            else:
                assert not jointData

                # world --> view coordinate
                R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
                T = Translate(T, device=T.device)
                R = Rotate(R, device=R.device)
                MM = R.compose(T)

                opt_mesh = o3d.geometry.TriangleMesh()

                # transform
                tmp = MM.transform_points(verts_list).detach().cpu().numpy()
                tmp[:,0] *= -1
                tmp[:,2] *= -1

                # visualize
                opt_mesh.vertices = o3d.utility.Vector3dVector(tmp)
                opt_mesh.triangles = o3d.utility.Vector3iVector(faces_list.cpu().numpy())
                opt_mesh.compute_vertex_normals()
                vis.clear_geometries()
                vis.add_geometry(opt_mesh)
                vis.poll_events()
                img = np.asarray(vis.capture_screen_float_buffer(True))
                image = img_as_ubyte(img)
                writer.append_data(image)


        vis.destroy_window()
        writer.close()

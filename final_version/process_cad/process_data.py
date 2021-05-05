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
from tqdm import tqdm
import re
import open3d as o3d
import itertools


# io utils
from pytorch3d.io import load_obj, save_obj

# datastructures
from pytorch3d.structures import Meshes

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights
)

import json
import csv


# helper function for computing roation matrix in 3D
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



# helper function for traversing a tree
def traverse_tree(current_node, mesh_dict):
    # further traverse the tree if not at leaf node yet
    if 'children' in current_node.keys():
        for idx in range(len(current_node['children'])):
            traverse_tree(current_node['children'][idx], mesh_dict)
    else:
        # insert meshes associated with an unique part id
        assert current_node['id'] not in mesh_dict.keys()
        mesh_dict[current_node['id']] = current_node['objs']
    return



# helper function for loading and merging meshes
def merge_meshes(save_folder, ids, mesh_dict):

    for count, part_ids in enumerate(ids):
        part_meshes = [mesh_dict[x] for x in part_ids]
        part_meshes = list(itertools.chain(*part_meshes))

        verts_list = np.empty((0,3))
        faces_list = np.empty((0,3))#.long()

        for part_mesh in part_meshes:
            obj_path = os.path.join(part_folder, 'textured_objs', part_mesh,)+'.obj'
            # check if mesh exist
            if not os.path.exists(obj_path):
                print(obj_path)
                continue
            mesh = o3d.io.read_triangle_mesh(obj_path)
            verts = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            faces = faces + verts_list.shape[0]
            verts_list = np.concatenate([verts_list, verts])
            faces_list = np.concatenate([faces_list, faces])

        mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(verts_list),
                                         triangles=o3d.utility.Vector3iVector(faces_list))
        mesh.compute_vertex_normals()
        save_path = os.path.join(save_folder, 'parts_ply')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        o3d.io.write_triangle_mesh(save_path+'/'+str(count)+'.ply', mesh)

    return


part_home = '/home/xuxiangx/research/johmr/dataset/mobility_v1_alpha5'
save_home = '/home/xuxiangx/research/johmr/dataset/final'
'''
classes_full = ['Refrigerator','Microwave','Laptop','Box','WashingMachine','TrashCan','Oven',
            'Safe','Dishwasher','Table','StorageFurniture','Lamp','Scissors','Chair',
            'FoldingChair','Kettle','KitchenPot','Printer','Toilet','Eyeglasses']  # 20
'''
classes = ['StorageFurniture']#,'Microwave','Laptop','WashingMachine','TrashCan','Oven',
         #   'Dishwasher','Toilet']  # 340 models





# Manually verify the part category
careParts = {}
careParts['Refrigerator'] = ['door', 'other_leaf', 'display_panel', 'door_frame',
                             'control_panel', 'glass']
careParts['Microwave'] = ['door']
careParts['Laptop'] =  ['shaft', 'other_leaf', 'screen_side', 'screen', 'screen_frame']
#careParts['Box'] =  ['rotation_lid', 'drawer', 'countertop', 'lid_surface']  # font on top
careParts['WashingMachine'] =  ['door']
careParts['TrashCan'] =  ['opener', 'lid', 'drawer', 'cover', 'cover_lid',
                          'frame_vertical_bar', 'container', 'other_leaf']
careParts['Oven'] = ['door', 'door_frame']
careParts['Dishwasher'] = ['door', 'shelf', 'display_panel', 'door_frame']
#careParts['Table'] = ['drawer', 'cabinet_door_surface', 'drawer_box', 'handle',
                      #'drawer_front', 'board', 'cabinet_door', 'shelf', 'keyboard_tray_surface']
careParts['StorageFurniture'] =  ['cabinet_door', 'mirror', 'drawer', 'drawer_box',
                                  'door', 'shelf', 'handle', 'glass', 'cabinet_door_surface',
                                  'other_leaf', 'countertop']
#careParts['FoldingChair'] = ['seat']
careParts['Toilet'] = ['lid', 'seat']
#careParts['Suitcase'] =  ['lid', 'pull-out_handle']


count = 0

# all dirIDs within this class
with open('partnetsim.models.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        if row['category'] in classes:
            #print(count)
            part_dir = row['category']
            part_id = row['dirId']
            part_folder = os.path.join(part_home, str(part_id))
            save_folder = os.path.join(save_home, part_dir, str(part_id))
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            count+=1

            # load meshes referenced json file
            with open(os.path.join(part_folder, 'result.json')) as json_file:
                part_meshes = json.load(json_file)
                # traverse through a tree
                mesh_dict = {}
                root = part_meshes[0]
                traverse_tree(root, mesh_dict)

            types = []
            with open(os.path.join(part_folder, 'mobility.urdf')) as f:
                our_lines = f.readlines()
                for line in our_lines:
                    myString = re.sub('\s+',' ',line)
                    if '<joint name=' in myString:
                        m_type = myString.split("type=",1)[1][1:-3]
                        types.append(m_type)
            type_idx = 0
            details = {}
            details_saved = {}

            # load mobility_v2 json file
            with open(os.path.join(part_folder, 'mobility_v2.json')) as json_file:

                mobility_parts = json.load(json_file)
                print('processing %s' % part_folder)

                part_div = []
                for idx, joint_part in enumerate(mobility_parts):

                    # visual names belonging to one joint part
                    joint_part_names = joint_part['parts']
                    assert(joint_part_names) # make sure not empty

                    # parse ids for each part
                    ids = [x['id'] for x in joint_part_names]

                    part_div.append(ids)

                    # save motion information
                    details[str(idx)] = joint_part['jointData'].copy()
                    details_saved[str(idx)] = joint_part['jointData'].copy()

                    # set type for care part
                    if type_idx<len(types):
                        if joint_part['name'] in careParts[part_dir]:
                            details[str(idx)]['type'] = types[type_idx]
                            details_saved[str(idx)]['type'] = types[type_idx]
                        type_idx += 1
                    else:
                        if details[str(idx)]:
                            assert type_idx>=len(types)
                            assert joint_part['name'] not in careParts[part_dir]


                    # remove non-care part
                    if not joint_part['jointData'] or joint_part['name'] not in careParts[part_dir]:
                        details[str(idx)] = {}
                        details_saved.pop(str(idx), None)


                with open(os.path.join(save_folder, 'motion.json'), 'w') as outfile:
                    json.dump(details_saved, outfile)

                assert len(details) == len(part_div)
                part_idx = 0
                fix_part = []
                parts = []

                for key, value in details.items():
                    if value == {}:
                        fix_part.append(part_div[part_idx])
                    else:
                        parts.append(part_div[part_idx])
                    part_idx += 1

                fix_part = list(itertools.chain(*fix_part))
                parts.append(fix_part)

                # load, merge, and save part mesh file
                merge_meshes(save_folder, parts, mesh_dict)


print(count)
print('all done...')

import os
import pdb
import numpy as np
import argparse
import subprocess
import multiprocessing
from model import JOHMRLite
import re
import torch
from pytorch3d.transforms import (
    euler_angles_to_matrix
)
from utils import (
    initialize_render2, merge_meshes,
    load_motion
)
import json
import matplotlib.pyplot as plt

global cad_path
global data_path

cad_path = '/home/xuxiangx/research/johmr/code/JOHMR_data/cad_final/dishwasher'
data_path = "/home/xuxiangx/research/johmr/code/JOHMR_data/data_v5.2/dishwasher"



def render(video):
    global cad_path
    global data_path

    vidname = video[0]
    vidpath = video[1]
    gt_mask_path = os.path.join(vidpath, 'gt_mask')
    gt_maskvisual_path = os.path.join(vidpath, 'gt_maskvisual')
    if not os.path.exists(gt_mask_path):
        os.makedirs(gt_mask_path)
    if not os.path.exists(gt_maskvisual_path):
        os.makedirs(gt_maskvisual_path)


    # load gt rotation
    gt_partrot = []
    fp = open(os.path.join(data_path, vidname, 'jointstate.txt'))
    for i, line in enumerate(fp):
        line = line.strip('\n')
        if line.isdigit() == True:
            gt_partrot.append(float(line))

    # load gt data annotaiton
    with open(os.path.join(data_path, vidname, '3d_info.txt')) as myfile:
        gt_data = [next(myfile).strip('\n') for x in range(14)]

    # gt orientation
    gt_alpha = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[3])[0])
    gt_beta =  float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[4])[0])
    gt_gamma = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[5])[0])
    gt_alpha_tensor = torch.tensor(gt_alpha).reshape(-1)
    gt_beta_tensor =  torch.tensor(gt_beta).reshape(-1)
    gt_gamma_tensor =  torch.tensor(gt_gamma).reshape(-1)
    euler_angle = torch.cat((gt_alpha_tensor,gt_beta_tensor,gt_gamma_tensor),0).reshape(1,3)
    rot_mat_gt = euler_angles_to_matrix(euler_angle, ["X","Y","Z"])

    # gt offset
    gt_x = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[6])[0])
    gt_y =  float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[7])[0])
    gt_z = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[8])[0])

    # gt obj size
    gt_xdim = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[9])[0])
    gt_ydim =  float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[9])[1])
    gt_zdim = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[9])[2])

    cadname = int(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[10])[0])
    partid = int(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[11])[0])
    focal_len = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[12])[0])


    # load cad model
    device = torch.device("cuda:0")
    obj_path = os.path.join(cad_path, str(cadname))
    verts, faces, vertexSegs, faceSegs = merge_meshes(obj_path, device)
    verts[:,1:] *= -1  # pytorch3d -> world coordinate
    obj_verts = verts.to(device)
    obj_faces = faces.to(device)
    obj_size = np.asarray([gt_xdim, gt_ydim, gt_zdim]) # length, height, width (x, y, z), cm

    focal_x = focal_len
    focal_y = focal_len
    img_square = 1280
    img_small = 256

    # load motion
    with open(os.path.join(obj_path, 'motion.json')) as json_file:
        motions = json.load(json_file)
    assert len(motions) + 2 == len(vertexSegs)
    rot_origin, rot_axis, rot_type, limit_a, limit_b, contact_list = load_motion(motions, device)

    # render
    renderer, _ = initialize_render2(device, focal_x, focal_y, img_square, img_small)

    # model
    model = JOHMRLite(obj_verts, obj_faces, renderer, rot_origin, rot_axis, vertexSegs, faceSegs, obj_size, gt_x, gt_y, gt_z, rot_mat_gt)

    # render and save
    for idx, partrot in enumerate(gt_partrot):
        silhouette = model(partrot, partid)
        silhouette = silhouette.astype(bool)
        np.save(os.path.join(gt_mask_path, str(idx+1).zfill(4)+'_'+'object_mask'), silhouette)
        plt.imsave(os.path.join(gt_maskvisual_path, str(idx+1).zfill(4)+'_'+'object_mask.jpg'), silhouette, cmap='gray')


videos = sorted([(f.name, f.path) for f in os.scandir(data_path)])
for video in videos:
    render(video)

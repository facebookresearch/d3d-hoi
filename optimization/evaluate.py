# Copyright (c) Facebook, Inc. and its affiliates.
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from pytorch3d.transforms import (
    so3_relative_angle,
    euler_angles_to_matrix
)
from scipy.spatial.distance import cdist
import json
from utils import (
    load_motion,
)
import re 
import argparse

from pytorch3d.transforms.rotation_conversions import (
    rotation_6d_to_matrix
)


def isfloat(x):
    try:
        a = float(x)
    except (TypeError, ValueError):
        return False
    else:
        return True

def isint(x):
    try:
        a = float(x)
        b = int(a)
    except (TypeError, ValueError):
        return False
    else:
        return a == b
        

parser = argparse.ArgumentParser()
parser.add_argument('--cad_path', type=str, help="experiment cad folder")
parser.add_argument('--result_folder', type=str, help="experiment result folder")
parser.add_argument('--data_path', type=str, help="experiment data folder")
parser.add_argument('--scale', type=float)
args = parser.parse_args()
cad_path = args.cad_path
result_folder = args.result_folder
anno_path = args.data_path

videos = sorted([(f.name, f.path) for f in os.scandir(result_folder)])
results = {}


# loop through all videos
for idx, video in enumerate(videos):
    vidname = video[0]
    vidpath = video[1]
    cads = sorted([(f.name, f.path) for f in os.scandir(vidpath)])

    if(vidname[:4]=='b001'):
        category = 'dishwasher'
    elif(vidname[:4]=='b003'):
        category = 'laptop'
    elif(vidname[:4]=='b004'):
        category = 'microwave'
    elif(vidname[:4]=='b005'):
        category = 'refrigerator'
    elif(vidname[:4]=='b006'):
        category = 'trashcan'
    elif(vidname[:4]=='b007'):
        category = 'washingmachine'
    elif(vidname[:4]=='b008'):
        category = 'storage_revolute'
    elif(vidname[:4]=='b108'):
        category = 'storage_prismatic'
    elif(vidname[:4]=='b009'):
        category = 'oven'

    # loop through all cad models
    for cad in cads:
        cadname = cad[0]
        cadpath = cad[1]
        settings = sorted([(f.name, f.path) for f in os.scandir(cadpath)])

        # loop through all settings
        for setting in settings:
            
            expname = setting[0]
            exppath = setting[1]
            partid = int(setting[0][0])
            
            # load experiment meta
            if not os.path.exists(os.path.join(exppath, 'params.npy')):
                print('missing '+vidname +' for setting '+expname)
                continue

            expmeta = np.load(os.path.join(exppath, 'params.npy'), allow_pickle=True)
            expmeta = expmeta.item()

            # load estimated global object rotation 
            exp_rot_angle = torch.from_numpy(expmeta['obj_rot_angle'])
            exp_rot_mat = rotation_6d_to_matrix(exp_rot_angle)  

            # load estimated global object translation (cm)
            exp_t = expmeta['obj_offset'] * args.scale
           
            # load estimated object dimension (cm)
            exp_dim = expmeta['obj_dim']

            # load estimated part motion (degree or cm)
            if cadname == '45132' or cadname == '45261':
                exp_partmotion = expmeta['part_motion'] * args.scale
            else:
                exp_partmotion = expmeta['part_motion'] * 57.296

            # load gt part motion values (degree or cm)
            gt_partmotion = []
            fp = open(os.path.join(anno_path, category, vidname, 'jointstate.txt'))
            for i, line in enumerate(fp):
                line = line.strip('\n')
                if isfloat(line) or isint(line):
                    gt_partmotion.append(float(line))
            gt_partmotion = np.asarray(gt_partmotion)
           
            with open(os.path.join(anno_path, category, vidname, '3d_info.txt')) as myfile:
                gt_data = [next(myfile).strip('\n') for x in range(14)]

            # GT global object rotation 
            gt_alpha = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[3])[0])
            gt_beta =  float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[4])[0])
            gt_gamma = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[5])[0])
            gt_alpha_tensor = torch.tensor(gt_alpha).reshape(-1)
            gt_beta_tensor =  torch.tensor(gt_beta).reshape(-1)
            gt_gamma_tensor =  torch.tensor(gt_gamma).reshape(-1)
            euler_angle = torch.cat((gt_alpha_tensor,gt_beta_tensor,gt_gamma_tensor),0).reshape(1,3)
            rot_mat_gt = euler_angles_to_matrix(euler_angle, ["X","Y","Z"])

            # GT global object translation (cm)
            gt_x = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[6])[0])*100.0
            gt_y =  float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[7])[0])*100.0
            gt_z = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[8])[0])*100.0

            # GT object dimension (cm)
            gt_xdim = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[9])[0])
            gt_ydim =  float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[9])[1])
            gt_zdim = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[9])[2])

            gt_cad = re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[10])[0]
            gt_part = int(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[11])[0])

            # CAD model correctness
            correctness = gt_cad==cadname #and gt_part == partid
            
            # Avg part motion abs error (degree or cm)
            motion_error = np.mean(np.abs(gt_partmotion - exp_partmotion))
                
            # Global object rotation error [relative angle (in degree) between the rotation matrixs in so3 space]
            R_dist = (so3_relative_angle(rot_mat_gt, exp_rot_mat, cos_angle=False).numpy()*57.296)[0]

            # Global object translation error (in cm)
            x_error = np.square(gt_x - exp_t[0])
            y_error = np.square(gt_y - exp_t[1])
            z_error = np.square(gt_z - exp_t[2])
            T_dist = np.sqrt(x_error+y_error+z_error)
            
            # Avg object dimension abs error (in cm)
            xdim_error = np.abs(gt_xdim - exp_dim[0])
            ydim_error = np.abs(gt_ydim - exp_dim[1])
            zdim_error = np.abs(gt_zdim - exp_dim[2])
            dim_error = (xdim_error + ydim_error + zdim_error)/3.0

            # print per video result 
            with open(os.path.join(os.path.dirname(result_folder),"result.txt"), 'a') as f:
                print(vidname+': ', file=f)
                print('model: '+str(cadname)+', part: '+str(partid), file=f)
                print('correctness: '+str(correctness), file=f)
                print('orientation (degree): '+str(round(R_dist,4)), file=f)
                print('location (cm): '+str(round(T_dist,4)), file=f)
                if cadname == '45132' or cadname == '45261':
                    print('motion (cm): '+str(round(motion_error,4)), file=f)
                else:
                    print('motion (degree): '+str(round(motion_error,4)), file=f)
                print('dimension (cm): '+str(round(dim_error,4)), file=f)
                print('--------------------------', file=f)

            if not category in results:
                results[category] = {}
                results[category]['correctness'] = []
                results[category]['orientation'] = []
                results[category]['location'] = []
                results[category]['motion'] = []
                results[category]['dimension'] = []
        
            results[category]['correctness'].append(int(correctness))
            if not correctness:
                continue 
            results[category]['orientation'].append(R_dist)
            results[category]['location'].append(T_dist)
            results[category]['motion'].append(motion_error)
            results[category]['dimension'].append(dim_error)


# per-category results:
for key, value in results.items():
    correct_percent = sum(value['correctness'])/len(value['correctness'])*100.0
    motion_mean = sum(value['motion'])/len(value['motion'])
    oriens_mean = sum(value['orientation'])/len(value['orientation'])
    locs_mean = sum(value['location'])/len(value['location'])
    dims_mean = sum(value['dimension'])/len(value['dimension'])

    with open(os.path.join(os.path.dirname(result_folder),"result.txt"), 'a') as f:
        print('--------------------------', file=f)
        print(key+' model correctness: '+str(correct_percent)+'%', file=f)
        print('motion_mean: '+str(motion_mean), file=f)
        print('orientation_mean: '+str(oriens_mean), file=f)
        print('location_mean: '+str(locs_mean), file=f)
        print('dimension_mean: '+str(dims_mean), file=f)
        print('--------------------------', file=f)



import os
import pdb
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
    get_3d_box,
    box3d_iou
)
import re 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cad_path', type=str, help="experiment cad folder")
parser.add_argument('--result_folder', type=str, help="experiment result folder")
parser.add_argument('--data_path', type=str, help="experiment data folder")
args = parser.parse_args()
cad_path = args.cad_path
result_folder = args.result_folder
anno_path = args.data_path

videos = sorted([(f.name, f.path) for f in os.scandir(result_folder)])

video_best_loss = 10000.0*np.ones(len(videos))
video_best_setting = {}
video_best_rot = {}
video_best_location = {}
video_best_orientation = {}
video_best_xdim = {}
video_best_ydim = {}
video_best_zdim = {}
video_best_iou = {}
video_best_cad = {}
video_best_category = {}


# loop through all videos
for idx, video in enumerate(videos):
    vidname = video[0]
    vidpath = video[1]
    cads = sorted([(f.name, f.path) for f in os.scandir(vidpath)])

    # loop through all cad models
    for cad in cads:
        cadname = cad[0]
        cadpath = cad[1]
        settings = sorted([(f.name, f.path) for f in os.scandir(cadpath)])


        # loop through all settings
        for setting in settings:
            expname = setting[0]
            exppath = setting[1]

            # load experiment meta
            if not os.path.exists(os.path.join(exppath, 'loss_meta.npy')):
                print('missing '+vidname +' for setting '+expname)
                continue
            expmeta = np.load(os.path.join(exppath, 'loss_meta.npy'), allow_pickle=True)
            expmeta = expmeta.item()
            exp_loss = expmeta['overall_loss']  # best loss for comparison

            # load object orientation result
            exp_alpha = torch.from_numpy(expmeta['obj_alpha']).reshape(-1)
            exp_beta = torch.from_numpy(expmeta['obj_beta']).reshape(-1)
            exp_gamma = torch.from_numpy(expmeta['obj_gamma']).reshape(-1)
            euler_angle = torch.cat((exp_alpha,exp_beta,exp_gamma),0).reshape(1,3)
            exp_rot_mat = euler_angles_to_matrix(euler_angle, ["X","Y","Z"])

            # load part articulation result
            exp_partrot = np.load(os.path.join(exppath, 'partRot.npy'), allow_pickle=True)

            # load object location result
            exp_obj_x = expmeta['obj_x']*100.0
            exp_obj_y = expmeta['obj_y']*100.0
            exp_obj_z = expmeta['obj_z']*100.0

            # load object dimension result
            exp_obj_xdim = expmeta['obj_xdim']
            exp_obj_ydim = expmeta['obj_ydim']
            exp_obj_zdim = expmeta['obj_zdim']
            
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
                category = 'storageFurniture'
            elif(vidname[:4]=='b009'):
                category = 'oven'
            else:
                print('not available category...')

            # load gt cad model limit
            with open(os.path.join(cad_path, category, cadname, 'motion.json')) as json_file:
                motions = json.load(json_file)
            device='cpu'
            _, _, _, limit_a, limit_b, _ = load_motion(motions, device)
            partid = int(setting[0][-1])


            # load gt rotation
            gt_partrot = np.zeros(len(exp_partrot))
            fp = open(os.path.join(anno_path, category, vidname, 'jointstate.txt'))
            for i, line in enumerate(fp):
                line = line.strip('\n')
                if line.isdigit() == True:
                    gt_partrot[i] = float(line)
           
            # load gt data annotaiton
            with open(os.path.join(anno_path, category, vidname, '3d_info.txt')) as myfile:
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
            gt_x = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[6])[0])*100.0
            gt_y =  float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[7])[0])*100.0
            gt_z = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[8])[0])*100.0

            # gt obj size  
            gt_xdim = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[9])[0])
            gt_ydim =  float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[9])[1])
            gt_zdim = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[9])[2])


            # compute part rotation abs error (in degree)
            partrot_error = np.abs(gt_partrot - exp_partrot*57.296)
                
            # compute object orientation error [relative angle (in degree) between the rotation matrixs in so3 space]
            R_distance = so3_relative_angle(rot_mat_gt, exp_rot_mat, cos_angle=False).numpy()*57.296

            # compute object location mse error (in cm)
            x_error = np.square(gt_x - exp_obj_x)
            y_error = np.square(gt_y - exp_obj_y)
            z_error = np.square(gt_z - exp_obj_z)
            location_mse_error = np.sqrt(x_error+y_error+z_error)
            
            # compute size dimension abs error (in cm)
            xdim_error = np.abs(gt_xdim - exp_obj_xdim)
            ydim_error = np.abs(gt_ydim - exp_obj_ydim)
            zdim_error = np.abs(gt_zdim - exp_obj_zdim)

            # compute 3D IoU (without part articulation)
            R = rot_mat_gt.numpy()[0]
            corners_3d_ground  = get_3d_box((gt_xdim, gt_zdim, gt_ydim), R, (gt_x, gt_y, gt_z)) 
            corners_3d_predict  = get_3d_box((exp_obj_xdim, exp_obj_ydim, exp_obj_zdim), R, (exp_obj_x, exp_obj_y, exp_obj_z))
            (iou_3d,_) = box3d_iou(corners_3d_predict,corners_3d_ground)


            if exp_loss < video_best_loss[idx]:
                video_best_loss[idx] = exp_loss
                video_best_setting[vidname] = setting
                video_best_orientation[vidname] = R_distance
                video_best_rot[vidname] = partrot_error
                video_best_location[vidname] = location_mse_error
                video_best_xdim[vidname] = xdim_error
                video_best_ydim[vidname] = ydim_error
                video_best_zdim[vidname] = zdim_error
                video_best_iou[vidname] = iou_3d
                video_best_cad[vidname] = cadname
                video_best_category[vidname] = category

rots = {}
oriens = {}
locs = {}
ious = {}
x_dims = {}
y_dims = {}
z_dims = {}

text_file = open(os.path.join(result_folder,"file_list.txt"), "w")

for idx, video in enumerate(videos):
    vidname = video[0]
    loss = video_best_loss[idx]
    model = video_best_cad[vidname]
    partid = int(video_best_setting[vidname][0][-1])
    orien = video_best_orientation[vidname][0]
    partrot_error = video_best_rot[vidname]
    location = video_best_location[vidname]
    iou = video_best_iou[vidname]
    xdim = video_best_xdim[vidname]
    ydim = video_best_ydim[vidname]
    zdim = video_best_zdim[vidname]

    partrot_mean = np.mean(partrot_error)
    partrot_std = np.std(partrot_error)

    # print per video result 
    with open(os.path.join(result_folder,'result.txt'), 'a') as f:
        print(vidname+', best loss: '+str(round(loss,4)), file=f)
        print('best model: '+str(model)+', best part: '+str(partid), file=f)
        print('best obj orientation (degree, so3): '+str(round(orien,4))+', best obj location (cm, mse): '+str(round(location,4)), file=f)
        print('best obj articulation mean (degree, abs): '+str(round(partrot_mean,4))+', best obj articulation std (degree, abs): '+str(round(partrot_std,4)), file=f)
        print('best obj iou: '+str(round(iou,4))+', best obj dimension (cm, abs, xyz): '+str(round(xdim,4))+ ', '+str(round(ydim,4))+ ', '+str(round(zdim,4)), file=f)
        print('-------------', file=f)
    
    category = video_best_category[vidname]
    if category not in rots.keys():
        rots[category] = []
    if category not in oriens.keys():
        oriens[category] = []
    if category not in locs.keys():
        locs[category] = []
    if category not in ious.keys():
        ious[category] = []
    if category not in x_dims.keys():
        x_dims[category] = []
    if category not in y_dims.keys():
        y_dims[category] = []
    if category not in z_dims.keys():
        z_dims[category] = []
    rots[category] += partrot_error.tolist()
    oriens[category].append(round(orien,4))
    locs[category].append(round(location,4))
    ious[category].append(round(iou,4))
    x_dims[category].append(round(xdim,4))
    y_dims[category].append(round(ydim,4))
    z_dims[category].append(round(zdim,4))

    text_file.write(video_best_setting[vidname][1]+'\n')
    
text_file.close()


for category in rots.keys():
    rots_mean = np.mean(np.asarray(rots[category]))
    rots_std = np.std(np.asarray(rots[category]))
    oriens_mean = np.mean(np.asarray(oriens[category]))
    oriens_std = np.std(np.asarray(oriens[category]))
    locs_mean = np.mean(np.asarray(locs[category]))
    locs_std = np.std(np.asarray(locs[category]))
    ious_mean = np.mean(np.asarray(ious[category]))
    ious_std = np.std(np.asarray(ious[category]))
    x_dims_mean = np.mean(np.asarray(x_dims[category]))
    x_dims_std = np.std(np.asarray(x_dims[category]))
    y_dims_mean = np.mean(np.asarray(y_dims[category]))
    y_dims_std = np.std(np.asarray(y_dims[category]))
    z_dims_mean = np.mean(np.asarray(z_dims[category]))
    z_dims_std = np.std(np.asarray(z_dims[category]))

    with open(os.path.join(result_folder,'result.txt'), 'a') as f:
        print(category+': ', file=f)
        print('rots_mean: '+str(rots_mean)+', rots_std: '+str(rots_std), file=f)
        print('oriens_mean: '+str(oriens_mean)+', oriens_std: '+str(oriens_std), file=f)
        print('locs_mean: '+str(locs_mean)+', locs_std: '+str(locs_std), file=f)
        print('ious_mean: '+str(ious_mean)+', ious_std: '+str(ious_std), file=f)
        print('x_dims_mean: '+str(x_dims_mean)+', x_dims_std: '+str(x_dims_std), file=f)
        print('y_dims_mean: '+str(y_dims_mean)+', y_dims_std: '+str(y_dims_std), file=f)
        print('z_dims_mean: '+str(z_dims_mean)+', z_dims_std: '+str(z_dims_std), file=f)
        print('--------------------------', file=f)


# zip test.zip $(cat file_list.txt) -r

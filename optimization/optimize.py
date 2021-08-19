# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import os
from model import JOHMRModel
from utils import (
    initialize_render, merge_meshes,
    load_motion,
    save_meshes, save_parameters
)
import json
import tqdm
from matplotlib.image import imsave
import matplotlib.pyplot as plt
import cv2
import re
import numpy as np
from PIL import Image
import glob
from dataloader import MyOwnDataset
import torch.nn as nn
import torch.optim as optim
import argparse
import itertools


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

#################
# training code #
#################
def run_exp(inputvideo):

    # Initialize gpu device
    assert torch.cuda.is_available()
    device = torch.device("cuda:"+str(args.device))

    global available_category
    vidpath = inputvideo[1]
    video_name = inputvideo[0]
    if(video_name[:4]=='b001'):
        video_category = 'dishwasher'
    elif(video_name[:4]=='b003'):
        video_category = 'laptop'
    elif(video_name[:4]=='b004'):
        video_category = 'microwave'
    elif(video_name[:4]=='b005'):
        video_category = 'refrigerator'
    elif(video_name[:4]=='b006'):
        video_category = 'trashcan'
    elif(video_name[:4]=='b007'):
        video_category = 'washingmachine'
    elif(video_name[:4]=='b008'):
        video_category = 'storage_revolute'
    elif(video_name[:4]=='b108'):
        video_category = 'storage_prismatic'
    elif(video_name[:4]=='b009'):
        video_category = 'oven'
    else:
        print('not available category...')
    print('processing '+video_name+' for category '+video_category)

    # load gt annotation, find the correct object size, cad model, part id, and focal len
    with open(os.path.join(vidpath, '3d_info.txt')) as myfile:
        gt_data = [next(myfile).strip('\n') for x in range(14)]


    # Initialize object scale (x, y, z)
    obj_sizeX = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[9])[0]) 
    obj_sizeY = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[9])[1]) 
    obj_sizeZ = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[9])[2]) 
    obj_dimension = [obj_sizeX, obj_sizeY, obj_sizeZ]  # in cm

    # initialize object cad model and part id
    if args.use_gt_objmodel:    

        if args.use_gt_objpart:
            cad_object = os.path.join(args.cadpath, video_category, re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[10])[0])
            cad_part = int(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[11])[0])
            cad_models = [(cad_object, cad_part)]

        else:
            cad_models = []
            cad_object = os.path.join(args.cadpath, video_category, re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[10])[0])
            with open(os.path.join(cad_object, 'motion.json')) as json_file:
                cad_parts = len(json.load(json_file))
            for cad_part in range(cad_parts):
                cad_models.append((cad_object,cad_part))
       
    else:
        # iter through all cad models in that category
        cad_models = []
        cad_objects = [f.path for f in os.scandir(os.path.join(args.cadpath, video_category))] 
        for cad_object in cad_objects:
            with open(os.path.join(cad_object, 'motion.json')) as json_file:
                cad_parts = len(json.load(json_file))
            for cad_part in range(cad_parts):
                cad_models.append((cad_object,cad_part))

    # initialize focal len 
    focal_len = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[12])[0]) 
    assert(focal_len ==  float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[13])[0])) # in pixel (for 1280x720 only)
    
    # initalize data loader (use all frames at once)
    dataset = MyOwnDataset(inputvideo[1])
    img_square, img_small = dataset.correct_image_size(200,300)
    if img_small <= 0:
        print('can not find small image size')
        return False

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), pin_memory=True, shuffle=False, num_workers=4)
    
    # initialize render 
    silhouette_renderer, phong_renderer = initialize_render(device, focal_len, focal_len, img_square, img_small)
    
    # load all data per video 
    for idx, data in enumerate(trainloader):
        imgs = data['image'].permute(0,2,3,1).to(device)
        batch_size = imgs.shape[0]
        img_h = imgs.shape[1]
        img_w = imgs.shape[2]
        points = data['smplv2d'].to(device).float()
        smpl_verts = data['ver'].to(device).float()
        smpl_faces = data['f'].to(device).float()
        joints = data['joint3d'].to(device).float()
        normal = data['normal'].to(device).float()
        normal2 =  data['normal2'].to(device).float()
        objmask = data['objmask'].permute(0,2,3,1).to(device).float()
    print('data loaded...')
    
    # load gt part motion
    gt_partmotion = []
    fp = open(os.path.join(vidpath, 'jointstate.txt'))
    for i, line in enumerate(fp):
        line = line.strip('\n')
        if isfloat(line) or isint(line):
            gt_partmotion.append(float(line))
    gt_partmotion = np.asarray(gt_partmotion)

    # Infer HOI snippet from GT part motions
    diff = gt_partmotion[:-1] - gt_partmotion[1:]  # (i - i+1)
    if video_category == 'storage_prismatic':
        large_diff = np.where(abs(diff)>0.5)[0]
    else:
        large_diff = np.where(abs(diff)>2)[0]
    care_idx = np.union1d(large_diff, large_diff+1)
    care_idx = np.clip(care_idx, 0, len(gt_partmotion)-1)

    # compute object mask center
    obj_x_center = 0
    obj_y_center = 0
    count = 1e-5
    for mask in objmask:
        if torch.sum(mask) > 0:
            count += 1
            small_img = mask.squeeze().detach().cpu().numpy()
            large_img = cv2.resize(small_img, dsize=(img_square, img_square), interpolation=cv2.INTER_NEAREST)
            x, y, w, h = cv2.boundingRect(np.uint8(large_img))
            obj_x_center += int(x+0.5*w)
            obj_y_center += int(y+0.5*h)
    obj_x_center /= count
    obj_y_center /= count    


    ###############################################
    # optimize different cad model configurations #
    ###############################################
    final_losses = []
    folders = []

    for (obj_path, part_idx) in cad_models:

        cad_name = re.findall(r'\d+', obj_path)[-1]

        # load object mesh
        verts, faces, vertexSegs, faceSegs = merge_meshes(obj_path)
        verts[:,1:] *= -1  # pytorch3d -> world coordinate

        if args.use_gt_objscale:
            # compute object rescale value if using gt dimension  (cm)
            x_diff = torch.max(verts[:,0]) - torch.min(verts[:,0])
            x_ratio = obj_dimension[0] / x_diff
            y_diff = torch.max(verts[:,1]) - torch.min(verts[:,1])
            y_ratio = obj_dimension[1] / y_diff
            z_diff = torch.max(verts[:,2]) - torch.min(verts[:,2])
            z_ratio = obj_dimension[2] / z_diff
        else:
            if video_category == 'laptop':
                initial_dim = 5.0
            elif cad_name == '10797':  
                initial_dim = 20.0  # small fridge
            elif video_category == 'refrigerator':
                initial_dim = 100.0 # large fridge 
            else:
                initial_dim = 50.0
            x_diff = torch.max(verts[:,0]) - torch.min(verts[:,0])
            x_ratio =  x_diff * initial_dim
            y_diff = torch.max(verts[:,1]) - torch.min(verts[:,1])
            y_ratio =  y_diff * initial_dim
            z_diff = torch.max(verts[:,2]) - torch.min(verts[:,2])
            z_ratio = z_diff * initial_dim

        obj_verts = verts.to(device)
        obj_faces = faces.to(device)

        # load motion json file
        with open(os.path.join(obj_path, 'motion.json')) as json_file:
            motions = json.load(json_file)
        assert len(motions) + 2 == len(vertexSegs)
        rot_origin, rot_axis, rot_type, limit_a, limit_b, contact_list = load_motion(motions, device)
        
        # Hand, object contact vertex id
        handcontact = [2005, 5466]  # left, right hand from SMPL 
        objcontact = contact_list[part_idx]

        # Optimize for all possible settings
        for handcontact_v in handcontact:
            for objcontact_v in objcontact:
                meta_info = str(part_idx)+'_'+str(objcontact_v)+'_'+str(handcontact_v)
                
                # initalize model      
                model = JOHMRModel(imgs.detach(), obj_verts.detach(), obj_faces.detach(),
                           smpl_verts.detach(), smpl_faces.detach(), points.detach(), 
                           silhouette_renderer, phong_renderer, normal.detach(), normal2.detach(), objmask.detach(),
                           rot_origin, rot_axis, rot_type, vertexSegs, faceSegs, limit_a, limit_b,
                           img_small ,focal_len, joints.detach())

                # initialize optimizer
                optimizer = optim.Adam(model.parameters(), lr=0.05)  # 0.05
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(0.75*args.iter), gamma=0.1)

                # Start optimizing
                for iteration in range(args.iter):
                    loss, loss_meta = model(iteration, args, cad_name, care_idx, part_idx, 
                                            handcontact_v, objcontact_v, obj_x_center, obj_y_center, 
                                            x_ratio, y_ratio, z_ratio)
                    if loss_meta is not None:
                        print('Iteration %d lr %.4f, total loss %.4f, smpl %.4f, mask %.4f, hfacing %.4f, depth %.4f, gamma %.4f, alpha %.4f, size %.3f, contact %.4f' 
                            % (iteration, optimizer.param_groups[0]['lr'], loss.data, loss_meta['l_points'], loss_meta['l_mask'], loss_meta['l_direction'], 
                            loss_meta['l_depth'],loss_meta['l_gamma'],loss_meta['l_alpha'], loss_meta['l_prior'],loss_meta['l_contact']))
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                # save results
                param_path = os.path.join(args.exp_name, 'params', video_name,cad_name, meta_info)
                save_parameters(model, param_path)
                final_losses.append(loss_meta['final_loss']) 
                folders.append(param_path) 
                
    # Only keep best result
    best_run = final_losses.index(min(final_losses)) 
    folders.remove(folders[best_run])
    for folder in folders:
        os.system('rm -r '+folder)

            

if __name__ == "__main__":
    global available_category
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', type=int)
    parser.add_argument('--use_gt_objscale', action='store_true')
    parser.add_argument('--use_gt_objmodel', action='store_true')
    parser.add_argument('--use_gt_objpart', action='store_true')
    parser.add_argument('--objmask', type=float)
    parser.add_argument('--hfacing', type=float)
    parser.add_argument('--depth', type=float)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--range', type=float)
    parser.add_argument('--smpl', type=float)
    parser.add_argument('--contact', type=float)
    parser.add_argument('--size', type=float)
    parser.add_argument('--center', type=float)
    parser.add_argument('--smooth', type=float)
    parser.add_argument('--scale', type=float)
    parser.add_argument('--category', type=str, help="which category to run")
    parser.add_argument('--exp_name', type=str, help="experiment main folder")
    parser.add_argument('--datapath', type=str, help="experiment data folder")
    parser.add_argument('--cadpath', type=str, help="experiment data folder")
    parser.add_argument("--device", type=int, help="CUDA Device Index")
    args = parser.parse_args()

    available_category = ['dishwasher', 'laptop', 'microwave', 'refrigerator', 'trashcan', 'washingmachine', 'oven', 'storage_revolute', 'storage_prismatic']
    if args.category not in available_category and args.category!='all':
        print('please choose a vaild category')

    # create main exp folder
    args.exp_path = os.path.join(os.getcwd(), args.exp_name)
    if not os.path.exists(args.exp_path):
        os.makedirs(args.exp_path)

    videopath = []
    # run on single object category
    if args.category != 'all':
        videopath = sorted([(f.name,f.path) for f in os.scandir(os.path.join(args.datapath, args.category))])
    # run on all object categories
    else:
        videopath = []
        for obj_class in available_category:
            videopath.append(sorted([(f.name, f.path) for f in os.scandir(os.path.join(args.datapath, obj_class))]))
        videopath = sorted(list(itertools.chain.from_iterable(videopath)))
    print('total of '+str(len(videopath))+' experiments...')

    # run locally
    for i in range(len(videopath)):
       run_exp(videopath[i])
    
    
    

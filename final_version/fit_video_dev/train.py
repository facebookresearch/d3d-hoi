import torch
import os
from model import JOHMRModel
from utils import (
    initialize_render, merge_meshes,
    load_motion, save_imgs,
    save_meshes, save_parameters
)
import pdb
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
import submitit

####################
# training code
def run_exp(args, inputvideo):

    global available_category
    vidname = inputvideo[0]
    vidpath = inputvideo[1]
    assert torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_category = inputvideo[1].split('/')[1]
    video_name = inputvideo[0]
    print('processing '+video_name+' for category '+video_category)

    # load gt annotation, find the correct object size, cad model, part id, and focal len
    with open(os.path.join(vidpath, '3d_info.txt')) as myfile:
        gt_data = [next(myfile).strip('\n') for x in range(14)]

    # initialize object scale (x, y, z)
    if args.use_gt_objscale:
        obj_sizeX = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[9])[0]) 
        obj_sizeY = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[9])[1]) 
        obj_sizeZ = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[9])[2]) 
        obj_dimension = [obj_sizeX, obj_sizeY, obj_sizeZ]  # in cm

    # initialize object cad model and part id
    if args.use_gt_objmodel:
        cad_object = os.path.join(args.cadpath, video_category, re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[10])[0])
        cad_part = int(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[11])[0])
        cad_models = [(cad_object, cad_part)]
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
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=8)

    # initialize render 
    silhouette_renderer, phong_renderer = initialize_render(device, focal_len, img_square, img_small)
    
    # load all frames data per video (take long time)
    for idx, data in enumerate(trainloader):
        # data from loader
        imgs = data['image'].permute(0,2,3,1).to(device)
        batch_size = imgs.shape[0]
        img_h = imgs.shape[1]
        img_w = imgs.shape[2]
        points = data['smplv2d'].to(device)
        smpl_verts = data['ver'].to(device)
        smpl_faces = data['f'].to(device)
        joints = data['joint3d'].to(device)
        normal = data['normal'].to(device)
        normal2 =  data['normal2'].to(device)
        #hmask = data['hmask'].permute(0,2,3,1).to(device)
        objmask = data['objmask'].permute(0,2,3,1).to(device)
        #hmask_square = data['hmask_square'].permute(0,2,3,1).to(device)
        #objmask_square = data['objmask_square'].permute(0,2,3,1).to(device)
    print('data loaded...')

    # compute object mask center
    obj_x_center = 0
    obj_y_center = 0
    count = 1e-5
    for mask in objmask:
        if torch.sum(mask) > 0:
            count += 1
            x, y, w, h = cv2.boundingRect(np.uint8(mask.squeeze().detach().cpu().numpy()))
            obj_x_center += int(x+0.5*w)
            obj_y_center += int(y+0.5*h)
    obj_x_center /= count
    obj_y_center /= count
    obj_x_center *= 5.0
    obj_y_center *= 5.0

    # load gt rotation to find action start / end frame
    gt_partrot = []
    fp = open(os.path.join(vidpath, 'jointstate.txt'))
    for i, line in enumerate(fp):
        line = line.strip('\n')
        if line.isdigit() == True:
            gt_partrot.append(float(line))
    gt_partrot = np.asarray(gt_partrot)
    set1 = np.where(gt_partrot!=gt_partrot[0])[0].tolist()
    set2 = np.where(gt_partrot!=0.0)[0].tolist()
    def intersection(lst1, lst2): 
        lst3 = [value for value in lst1 if value in lst2] 
        return lst3 
    frames = intersection(set1, set2)
    start_idx = min(frames)
    end_idx = max(frames)

    #####################################
    # optimize different cad model configurations
    for (obj_path, part_idx) in cad_models:

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

        obj_verts = verts.to(device)
        obj_faces = faces.to(device)

        # load motion json file
        with open(os.path.join(obj_path, 'motion.json')) as json_file:
            motions = json.load(json_file)
        assert len(motions) + 2 == len(vertexSegs)
        rot_origin, rot_axis, rot_type, limit_a, limit_b, contact_list = load_motion(motions, device)

        # create folder
        cad_name = re.findall(r'\d+', obj_path)[-1]

        if args.contact == 0:
            handcontact = [5510]
            objcontact = [-1]  # default value
        else:
            handcontact = [5510, 2049]
            objcontact = contact_list[part_idx]

        for handcontact_v in handcontact:
            for objcontact_v in objcontact:
                meta_info = str(part_idx)+'_'+str(objcontact_v)+'_'+str(handcontact_v)
                save_folder = os.path.join(args.exp_path, video_name, cad_name, meta_info)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                # initalize model      
                model = JOHMRModel(imgs.detach(), obj_verts.detach(), obj_faces.detach(),
                           smpl_verts.detach(), smpl_faces.detach(), points.detach(), 
                           silhouette_renderer, phong_renderer, normal.detach(), normal2.detach(),
                           objmask.detach(),
                           rot_origin, rot_axis, rot_type, vertexSegs, faceSegs, limit_a, limit_b,
                           img_small ,focal_len, joints.detach())

                # initialize optimizer
                optimizer = optim.Adam(model.parameters(), lr=0.1)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=110, gamma=0.1)

                # optimize each for this many iterations
                for iteration in range(args.iter):
                    optimizer.zero_grad()

                    if args.use_gt_objscale:
                        loss, loss_meta = model(iteration, args, cad_name, start_idx, end_idx, part_idx, 
                                                handcontact_v, objcontact_v,
                                                obj_x_center, obj_y_center, x_ratio, y_ratio, z_ratio)
                    else:
                        loss, loss_meta = model(iteration, args, cad_name, start_idx, end_idx, part_idx, 
                                                handcontact_v, objcontact_v,
                                                obj_x_center, obj_y_center)
                                                
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    
                print('---------------')
                print(model.obj_scale)
                print(model.obj_offset)
                print(model.part_rot_params)
                print('Iteration %d loss %.4f' % (iteration, loss.data))
        
                np.save(os.path.join(save_folder, 'loss_meta'),loss_meta)
                part_rot_np = model.part_rot_params.detach().cpu().numpy()
                np.save(os.path.join(save_folder, 'partRot'), part_rot_np)

                # save mesh results
                meshes = model.render()
                mesh_path = os.path.join(save_folder.split('fit_video_dev')[0],'fit_video_dev', 'meshes')
                save_folder_mesh = mesh_path + save_folder.split('fit_video_dev')[1]
                if not os.path.exists(save_folder_mesh):
                    os.makedirs(save_folder_mesh)
                save_meshes(meshes, save_folder_mesh, video_name, '')

                print('done')
                print('-----------')




if __name__ == "__main__":

    global available_category

    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', type=int)
    parser.add_argument('--use_gt_objscale', action='store_true')
    parser.add_argument('--use_gt_objmodel', action='store_true')
    parser.add_argument('--objmask', type=float)
    parser.add_argument('--flowmask', type=float)
    parser.add_argument('--hfacing', type=float)
    parser.add_argument('--depth', type=float)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--correctrot', type=float)
    parser.add_argument('--smpl', type=float)
    parser.add_argument('--contact', type=float)
    parser.add_argument('--synflow', type=float)
    parser.add_argument('--size', type=float)
    parser.add_argument('--center', type=float)
    parser.add_argument('--category', type=str, help="which category to run")
    parser.add_argument('--exp_name', type=str, help="experiment main folder")
    parser.add_argument('--datapath', type=str, help="experiment data folder")
    parser.add_argument('--cadpath', type=str, help="experiment data folder")
    parser.add_argument('--log', type=str, help="experiment log")
    args = parser.parse_args()

    available_category = ['dishwasher', 'laptop', 'microwave', 'refrigerator', 'trashcan', 'washingmachine', 'oven', 'storageFurniture']
    if args.category not in available_category and args.category!='all':
        print('please choose a vaild category')

    # create main exp folder
    args.exp_path = os.path.join(os.getcwd(), args.exp_name)
    if not os.path.exists(args.exp_path):
        os.makedirs(args.exp_path)

    # load video path
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
    #for i in range(len(videopath)):
        #run_exp(args, videopath[i])  
    #pdb.set_trace()

    
    # run on devfair
    executor = submitit.AutoExecutor(folder=args.log+"/%j")
    executor.update_parameters(timeout_min=30, slurm_partition="learnfair", cpus_per_task=10,
                               gpus_per_node=1, slurm_comment="siggraph", name=args.exp_name)#"priority" "learnfair"
    for idx in range(len(videopath)):
        print(idx)
        executor.submit(run_exp, args, videopath[idx])
    

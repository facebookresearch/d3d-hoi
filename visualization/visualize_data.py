# Copyright (c) Facebook, Inc. and its affiliates.
import cv2
import sys
import numpy as np
from utils import (
    initialize_render, merge_meshes,
    load_motion
)
import torch
from PIL import Image
from model import JOHMRLite
import os
import glob
import json
from pathlib import Path
import argparse
import re 
import matplotlib.pyplot as plt 


global model, index, alpha 
index = 0
alpha = 0.5

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


def find_files(folder, extension):
    return sorted([Path(os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith(extension)])


def read_data():
    """
    Load all annotated data for visualization
    """
    # load gt part motion values (degree or cm)
    gt_partmotion = []
    fp = open(os.path.join(args.data_folder, 'jointstate.txt'))
    for i, line in enumerate(fp):
        line = line.strip('\n')
        if isfloat(line) or isint(line):
            gt_partmotion.append(float(line))
    gt_partmotion = np.asarray(gt_partmotion)
           
    with open(os.path.join(args.data_folder, '3d_info.txt')) as myfile:
        gt_data = [next(myfile).strip('\n') for x in range(14)]
    
    # GT global object rotation 
    gt_pitch = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[3])[0])
    gt_yaw =  float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[4])[0])
    gt_roll = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[5])[0])

    # GT global object translation (cm)
    gt_x = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[6])[0])
    gt_y =  float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[7])[0])
    gt_z = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[8])[0])

    # GT object dimension (cm)
    gt_xdim = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[9])[0])
    gt_ydim =  float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[9])[1])
    gt_zdim = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[9])[2])

    gt_cad = re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[10])[0]
    gt_part = int(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[11])[0])

    gt_focalX = int(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[-2])[0])
    gt_focalY = int(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[-1])[0])

    assert gt_focalX == gt_focalY

    data = {'part_motion': gt_partmotion,
            'pitch': gt_pitch,
            'yaw': gt_yaw,
            'roll': gt_roll,
            'x_offset': gt_x,
            'y_offset': gt_y,
            'z_offset': gt_z,
            'obj_size': [gt_xdim, gt_ydim, gt_zdim],
            'cad': gt_cad,
            'part': gt_part,
            'focal': gt_focalX}

    return data 


def create_model(gt_data):
    """
    create initial models
    """
    global model, index, alpha 

    x_offset = gt_data['x_offset']
    y_offset = gt_data['y_offset']
    z_offset = gt_data['z_offset']
    yaw = gt_data['yaw']
    pitch = gt_data['pitch']
    roll = gt_data['roll']
    part_motion = gt_data['part_motion']
    obj_size = gt_data['obj_size'] # length, height, width (x, y, z), cm
    focal_x = gt_data['focal']
    focal_y = gt_data['focal']

    device = torch.device("cuda:0")
    obj_path = os.path.join(args.cad_folder, gt_data['cad'])
    verts, faces, vertexSegs, faceSegs = merge_meshes(obj_path, device)
    verts[:,1:] *= -1  # pytorch3d -> world coordinate
    obj_verts = verts.to(device)
    obj_faces = faces.to(device)

    # load motion json file
    with open(os.path.join(args.cad_folder, gt_data['cad'], 'motion.json')) as json_file:
        motions = json.load(json_file)
    assert len(motions) + 2 == len(vertexSegs)
    rot_o, rot_axis, rot_type, limit_a, limit_b, contact_list = load_motion(motions, device)

    frames = find_files(os.path.join(args.data_folder, 'frames'), '.jpg')
    image_bg = np.array(Image.open(frames[index]))/255.0
    img_h = image_bg.shape[0]
    img_w = image_bg.shape[1]
    img_square = max(img_h, img_w)
    img_small = 256

    # render
    _, phong_renderer = initialize_render(device, focal_x, focal_y, img_square, img_small)

    # Model >_<
    model =  JOHMRLite(x_offset, y_offset, z_offset, yaw, pitch, roll, part_motion, obj_size, \
                       obj_verts, obj_faces, phong_renderer, gt_data['part'], rot_o, rot_axis, \
                       vertexSegs, rot_type)

    return len(frames)


def display_img():
    global model, index, alpha 

    frames = find_files(os.path.join(args.data_folder, 'frames'), '.jpg')
    image_bg = np.array(Image.open(frames[index]))/255.0
    img_h = image_bg.shape[0]
    img_w = image_bg.shape[1]
    img_square = max(img_h, img_w)
    img_small = 256

    with torch.no_grad():
        image = model(index)
    rgb_mask = image_bg.astype(np.float32) #cv2.addWeighted(objmask.astype(np.float32), 0.5, image_bg.astype(np.float32), 0.5, 0.0)
    
    frame_img = np.zeros((img_square, img_square,3))
    start = int((max(img_h, img_w) - min(img_h, img_w))/2) - 1
    end = start + min(img_h, img_w)
    if img_h > img_w:
        frame_img[:, start:end,  :] = rgb_mask
    else:
        frame_img[start:end, :, :] = rgb_mask
    rgb_mask = frame_img
    alpha = min(1.0, max(0.0,alpha))
    img_blend = cv2.addWeighted(image.astype(np.float32), alpha, rgb_mask.astype(np.float32), 1-alpha, 0.0)
    img_blend = cv2.resize(img_blend, dsize=(800, 800), interpolation=cv2.INTER_NEAREST)
    return img_blend






parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, help="annotation data folder")
parser.add_argument("--cad_folder", type=str, help="cad data folder")
args = parser.parse_args()

gt_data = read_data()
num_frames = create_model(gt_data)

for index in range(num_frames):
    img_blend = display_img()
    plt.imshow(img_blend)
    plt.show()

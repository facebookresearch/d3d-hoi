# Copyright (c) Facebook, Inc. and its affiliates.
from skimage import io
from torch.utils.data import Dataset
import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import cv2
from natsort import natsorted
from utils import planeFit
from numpy.linalg import norm
import glob
from pytorch3d.io import load_obj

class MyOwnDataset(Dataset):
    """ My Own data loader. """

    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the images, masks and meta file.
            category (string): Category class.
        """
        self.img_paths = sorted(glob.glob(os.path.join(root_dir, 'frames', '*.jpg')))
        self.smplv2d_paths = sorted(glob.glob(os.path.join(root_dir, 'smplv2d', '*.npy')))
        self.smplmesh_paths = sorted(glob.glob(os.path.join(root_dir, 'smplmesh', '*.obj')))
        self.joint3d_paths = sorted(glob.glob(os.path.join(root_dir, 'joints3d', '*.npy')))
        self.objmask_paths = sorted(glob.glob(os.path.join(root_dir, 'gt_mask', '*object_mask.npy')))
        
        # transformations
        transform_list = [transforms.ToTensor()]
        self.transforms = transforms.Compose(transform_list)


    def correct_image_size(self,low,high):
        # automatically finds a good ratio in the given range
        image = np.array(Image.open(self.img_paths[0]))
        img_h = image.shape[0]
        img_w = image.shape[1]
        img_square = max(img_h,img_w)
        img_small = -1
        for i in range(low, high):
            if img_square % i == 0:
                img_small = i
                break
        return img_square, img_small


    def __len__(self):
        return len(self.img_paths)


    def getImgPath(self):
        return self.img_paths


    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = np.array(Image.open(self.img_paths[idx]))       # load image
        img_h = image.shape[0]
        img_w = image.shape[1]

        square_len = max(img_h, img_w)  # compute add in region for square
        new_s = int((max(img_h, img_w) - min(img_h, img_w))/2)-1
        add_l = min(img_h, img_w)

        objmask = np.load(self.objmask_paths[idx]).astype(np.uint8)
        smplv2d = np.load(self.smplv2d_paths[idx])                  # load 2D points
        joint3d = np.load(self.joint3d_paths[idx])

        if (joint3d.shape[0] == 147):
            pdb.set_trace()
            # no avaiable frame
            normal = np.zeros((3))
            normal2 = np.zeros((3))
        else:
            # estimate the body fitting plane and its normal vector
            joints_np = np.transpose(joint3d) # (3xN)
            lhip_to_rShoulder = joint3d[33] - joint3d[28]
            rhip_to_lShoulder = joint3d[34] - joint3d[27]
            normal = np.cross(lhip_to_rShoulder, rhip_to_lShoulder)
            normal = normal / np.sqrt(np.sum(normal**2))
            arm = joint3d[31,:] -  joint3d[33,:]
            cos_sim = np.inner(normal, arm)/(norm(normal)*norm(arm))
            if cos_sim < 0:
                normal *= -1

            lankle_to_rtoe = joint3d[22] - joint3d[30]
            rankle_to_ltoe = joint3d[19] - joint3d[25]
            normal2 = np.cross(lankle_to_rtoe, rankle_to_ltoe)
            normal2 = normal2 / np.sqrt(np.sum(normal2**2))
            leg = joint3d[29,:] -  joint3d[30,:]
            cos_sim2 = np.inner(normal2, leg)/(norm(normal2)*norm(leg))
            if cos_sim2 < 0:
                normal2 *= -1

        # SMPL mesh
        verts, faces, aux = load_obj(self.smplmesh_paths[idx])
        faces = faces.verts_idx
        verts = verts.float()
        faces = faces.long()
        joints = torch.from_numpy(joint3d).float()   
        normal = torch.from_numpy(normal).float()
        normal2 = torch.from_numpy(normal2).float()

        # apply transformations
        image = self.transforms(np.uint8(image))
        objmask = self.transforms(np.uint8(objmask))
        objmask[objmask>0.0] = 1.0
      
        data = {'image': image, 'objmask': objmask,
                'smplv2d': smplv2d, 'ver': verts, 'f': faces,
                'normal': normal, 'normal2': normal2, 'joint3d': joints}

        return data








# Copyright (c) Facebook, Inc. and its affiliates.
import torch.nn as nn
import torch
import numpy as np
from pytorch3d.renderer import (
    look_at_view_transform, TexturesVertex
)
import math
from pytorch3d.structures import Meshes
import cv2
import matplotlib.pyplot as plt
from utils import rotation_matrix_batch
from scipy.ndimage.filters import gaussian_filter1d
from pytorch3d.io import save_obj
from pytorch3d.transforms import (
    RotateAxisAngle, matrix_to_euler_angles, euler_angles_to_matrix
)
from pytorch3d.transforms.rotation_conversions import (
    rotation_6d_to_matrix, matrix_to_rotation_6d
)
import time
from matplotlib.image import imsave
import os
from torch.autograd import Variable
import open3d as o3d 


class JOHMRModel(nn.Module):
    """ Differentiable render for fitting CAD model based on silhouette and human. """
    def __init__(self, imgs, obj_verts, obj_faces, smpl_verts, smpl_faces, points,
                 diff_render, vis_render, normal, normal2, objmask,
                 rot_o, axis, rot_type, vertexSegs, faceSegs, limit_a, limit_b,
                 img_size_small ,focal_len, joints):
    
        super(JOHMRModel, self).__init__()
        self.imgs = imgs
        self.objmask = objmask[..., 0]
        self.objmask.requires_grad = False
        self.device = smpl_verts.device
        self.diff_render = diff_render
        self.vis_render = vis_render
        self.obj_verts_orig = obj_verts
        self.obj_faces = obj_faces
        self.smpl_verts_orig = smpl_verts
        self.smpl_faces = smpl_faces
        self.points = points
        self.rot_origs = rot_o
        self.rot_axises = axis
        self.vertexSegs = vertexSegs
        self.faceSegs = faceSegs
        self.limit_as = limit_a
        self.limit_bs = limit_b
        self.rot_type = rot_type
        self.bs = self.imgs.shape[0]
        self.normal = normal
        self.normal2 = normal2
        self.img_h = self.imgs.shape[1]
        self.img_w = self.imgs.shape[2]
        self.new_s = int((max(self.img_h, self.img_w) - min(self.img_h, self.img_w))/2)-1
        self.img_small = img_size_small
        self.focal = focal_len
        self.joints = joints
        self.normalize = 1.0/(0.5*(self.img_h+self.img_w))

        K = torch.from_numpy(np.array([[self.focal, 0, self.img_w/2],
                                       [0, self.focal, self.img_h/2],
                                       [0,0,1]]))
        self.K = K.float().to(self.device)

        # camera is at the center
        self.R, self.T = look_at_view_transform(0.1, 0.0, 0.0, device=self.device)
        self.T[0,2] = 0.0  # manually set to zero

        # object CAD x, y, z offset in 3D
        obj_offset = np.array([0.0, 0.0, 2.5], dtype=np.float32)
        self.obj_offset = nn.Parameter(torch.from_numpy(obj_offset).to(self.device))

        # object CAD scale in 3D
        obj_scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.obj_scale = nn.Parameter(torch.from_numpy(obj_scale).to(self.device))

        # SPIN mesh x, y, z offset in 3D
        smpl_offset = np.zeros((self.bs,3), dtype=np.float32)
        smpl_offset[:,0] = 0.0
        smpl_offset[:,1] = 0.0
        smpl_offset[:,2] = 2.5
        self.smpl_offset = nn.Parameter(torch.from_numpy(smpl_offset).to(self.device))

        # local rotation angle or translation offset for the parts
        part_motion = 0.0*np.ones(self.bs, dtype=np.float32)
        self.part_motion = nn.Parameter(torch.from_numpy(part_motion).to(self.device))

        # global rotation angle for the object CAD
        yaw_degree = 0.0 * 180/np.pi #-20.0# * 180/np.pi #0.0* 180/np.pi
        rot_mat = RotateAxisAngle(yaw_degree, axis='X').get_matrix()
        rot_mat = rot_mat[0,:3,:3].unsqueeze(0)
        ortho6d = matrix_to_rotation_6d(rot_mat)
        self.obj_rot_angle = nn.Parameter(ortho6d.to(self.device))
       
        # curve rotation in 3D
        yaw_degree2 = 0.0 * 180/np.pi #0.0* 180/np.pi
        rot_mat2 = RotateAxisAngle(yaw_degree2, axis='Y').get_matrix()
        rot_mat2 = rot_mat2[0,:3,:3].unsqueeze(0)
        ortho6d2 = matrix_to_rotation_6d(rot_mat2)
        self.curve_rot_angle = nn.Parameter(ortho6d2.to(self.device))
        curve_offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.curve_offset = nn.Parameter(torch.from_numpy(curve_offset).to(self.device))
 
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.relu = nn.ReLU()
        return



    def forward(self, iteration, args, cad_name, care_idx, part_idx, handcontact_v, objcontact_v, 
                obj_x_center, obj_y_center, x_ratio, y_ratio, z_ratio):

        # Predefined CAD segmentation and motion axis from SAPIEN
        self.vertexStart = self.vertexSegs[part_idx]
        self.vertexEnd = self.vertexSegs[part_idx+1]
        faceStart = self.faceSegs[part_idx]
        faceEnd = self.faceSegs[part_idx+1]
        self.rot_o = self.rot_origs[part_idx].clone().to(self.device).detach()
        self.axis = self.rot_axises[part_idx].clone().to(self.device).detach()
        limit_a = self.limit_as[part_idx]
        limit_b = self.limit_bs[part_idx]

        self.rot_o.requires_grad = False 
        self.axis.requires_grad = False 

        # Transform pytorch3d -> world coordinate 
        self.rot_o[1:] *= -1
        self.axis[1:] *= -1

        ####################
        ## fit human mesh ##
        ####################
        self.smpl_verts = self.smpl_verts_orig.clone()

        # Resize human mesh 
        smplmesh_calibrate_path = 'smplmesh-calibrate.obj'
        smplmesh_calibrate =  o3d.io.read_triangle_mesh(smplmesh_calibrate_path) # load smpl mesh
        hverts_cal = torch.from_numpy(np.asarray(smplmesh_calibrate.vertices)).float()
        human_height = 175 #cm
        h_diff = torch.max(hverts_cal[:,1]) - torch.min(hverts_cal[:,1])
        h_ratio = (human_height / h_diff).detach()
        self.smpl_verts *= h_ratio
 
        # Add x y z offsets to SMPL mesh (camera looking at positive depth z)
        smpl_offset = self.smpl_offset.reshape(-1,1,3).repeat(1,self.smpl_verts_orig.shape[1],1) # (bs, 6890, 3)
        self.smpl_verts[:,:,0] += args.scale*smpl_offset[:,:,0]
        self.smpl_verts[:,:,1] += args.scale*smpl_offset[:,:,1]
        self.smpl_verts[:,:,2] += args.scale*smpl_offset[:,:,2] #smpl_offset[:,:,2]  #smpl_offset[0,:,2]

        # Compute projection matrix K
        K_batch = self.K.expand(self.smpl_verts.shape[0],-1,-1)

        # Prespective projection
        points_out_v = torch.bmm(self.smpl_verts, K_batch.permute(0,2,1))
        self.smpl_2d = points_out_v[...,:2] / points_out_v[...,2:]

        # Human fitting error 
        l_points = torch.mean(self.normalize*(self.points - self.smpl_2d)**2)

       
        #####################
        ## optimize object ##
        #####################
        self.obj_rot_mat = rotation_6d_to_matrix(self.obj_rot_angle)[0].to(self.device)

        # pitch, yaw, roll
        alpha,beta,gamma = matrix_to_euler_angles(rotation_6d_to_matrix(self.obj_rot_angle), ["X","Y","Z"])[0]

        obj_verts_batch = self.obj_verts_orig.reshape(1,-1,3).repeat(self.bs,1,1) # (bs, ver, 3)

        # Step 1: rescale object and rotation orig
        if not args.use_gt_objscale:
            sx = self.obj_scale[0] * x_ratio  
            sy = self.obj_scale[1] * y_ratio
            sz = self.obj_scale[2] * z_ratio
        else:
            sx = x_ratio 
            sy = y_ratio
            sz = z_ratio

        obj_verts_batch[:,:,0] *= sx  
        obj_verts_batch[:,:,1] *= sy
        obj_verts_batch[:,:,2] *= sz
        self.rot_o[0] *= sx
        self.rot_o[1] *= sy
        self.rot_o[2] *= sz

        # Oject real-world dimension after scaling
        self.x_dim = torch.max(obj_verts_batch[0,:,0]) - torch.min(obj_verts_batch[0,:,0])
        self.y_dim = torch.max(obj_verts_batch[0,:,1]) - torch.min(obj_verts_batch[0,:,1])
        self.z_dim = torch.max(obj_verts_batch[0,:,2]) - torch.min(obj_verts_batch[0,:,2])

        # Step 2: add part motion (prismatic or revolute)
        if cad_name == '45261' or cad_name == '45132':
            obj_verts_batch_t1 = obj_verts_batch[:, self.vertexStart:self.vertexEnd, :] - self.rot_o
            self.part_motion_scaled = self.part_motion * args.scale
            batch_offset = self.axis.unsqueeze(0).repeat(self.bs,1) * self.part_motion_scaled.unsqueeze(-1).repeat(1,3)
            obj_verts_batch_t2 = obj_verts_batch_t1 + batch_offset.unsqueeze(1).repeat(1,obj_verts_batch_t1.shape[1], 1)  
            obj_verts_batch[:, self.vertexStart:self.vertexEnd, :] = obj_verts_batch_t2 + self.rot_o
        else:
            self.part_rot_mat = rotation_matrix_batch(self.axis, self.part_motion, self.device)
            obj_verts_batch_t1 = obj_verts_batch[:, self.vertexStart:self.vertexEnd, :] - self.rot_o
            obj_verts_batch_t2 = torch.bmm(self.part_rot_mat, obj_verts_batch_t1.permute(0,2,1)).permute(0,2,1) 
            obj_verts_batch[:, self.vertexStart:self.vertexEnd, :] = obj_verts_batch_t2 + self.rot_o
        
        # Step 3: add global object rotation
        obj_verts_batch = torch.bmm(self.obj_rot_mat.reshape(1,3,3).repeat(self.bs,1,1),
                                    obj_verts_batch.permute(0,2,1)).permute(0,2,1)

        # Step 4: add global object translation
        self.obj_verts_batch =  obj_verts_batch + args.scale*self.obj_offset

        # Object center error 
        obj_2d = torch.bmm(self.obj_verts_batch, K_batch.permute(0,2,1))
        self.obj_2d = (obj_2d[...,:2] / (obj_2d[...,2:]))  # (bs, objV, 2)
        obj_2d_x_center = torch.mean(self.obj_2d[:,:,0])
        obj_2d_y_center = torch.mean(self.obj_2d[:,:,1])
        if self.img_w > self.img_h:
            obj_2d_y_center += self.new_s
        else:
            obj_2d_x_center += self.new_s
        l_mask_center = self.normalize*(obj_y_center - obj_2d_y_center)**2 +  self.normalize*(obj_x_center - obj_2d_x_center)**2
        
        # Object & human orientation error
        if '10213' in cad_name or '9968' in cad_name:
            # Difficult to predefine orientation for laptop
            # Use CAD base part 
            front_vertex = self.obj_verts_orig[645+581].detach()
            top_vertex = self.obj_verts_orig[645+285].detach()
            base_center = self.obj_verts_orig[self.vertexSegs[-2]:self.vertexSegs[-1]].detach()
        
            obj_norm = torch.mean(base_center, 0) - front_vertex
            obj_norm_rot = torch.mm(self.obj_rot_mat, obj_norm.float().reshape(-1,1)).permute(1,0)
            output = self.cos(self.normal, obj_norm_rot.repeat(self.bs, 1))
            l_direction = torch.mean((1.0 - output)[care_idx])

            obj_norm2 =  top_vertex - torch.mean(base_center, 0) 
            obj_norm_rot2 = torch.mm(self.obj_rot_mat, obj_norm2.float().reshape(-1,1)).permute(1,0)
            output2 = self.cos(self.normal2, obj_norm_rot2.repeat(self.bs, 1))
            l_direction2 = torch.mean((1.0 - output2))

        else:
            obj_norm = torch.from_numpy(np.asarray([0,0,1])).to(self.device)
            obj_norm2 = torch.from_numpy(np.asarray([0,-1,0])).to(self.device)
            obj_norm_rot = torch.mm(self.obj_rot_mat, obj_norm.float().reshape(-1,1)).permute(1,0)
            obj_norm_rot2 = torch.mm(self.obj_rot_mat, obj_norm2.float().reshape(-1,1)).permute(1,0)
            output = self.cos(self.normal, obj_norm_rot.repeat(self.bs, 1))
            output2 = self.cos(self.normal2, obj_norm_rot2.repeat(self.bs, 1))
            l_direction = torch.mean((1.0 - output)[care_idx])   
            l_direction2 = torch.mean((1.0 - output2))
        
        
        # Differentiable mask error
        diff_images = []
        for index in range(self.bs):
            # convert object mesh for diff render, opengl -> pytorch3d
            p3d_obj_verts = self.obj_verts_batch[index].clone()
            p3d_obj_verts[:,1] *= -1
            p3d_obj_verts[:,2] *= -1
            # pass through diff render
            tex = torch.ones_like(p3d_obj_verts).unsqueeze(0)
            textures = TexturesVertex(verts_features=tex).to(self.device)
            obj_mesh = Meshes(verts=[p3d_obj_verts],faces=[self.obj_faces],textures=textures)
            diff_img = self.diff_render(meshes_world=obj_mesh, R=self.R, T=self.T)  
            diff_img = diff_img[..., 3:]
            diff_img = diff_img.permute(0,3,1,2)[0,0,:,:] #(h,w)
            diff_images.append(diff_img)
        
        diff_images = torch.stack(diff_images)  #(bs,h,w)
        mask2 = (diff_images>0).detach() 
        l_gtMask =  torch.mean(self.objmask*(diff_images-self.objmask)**2)
        l_rendMask =  torch.mean(mask2*((diff_images-self.objmask)**2)) 
        mask_diff = torch.mean((diff_images-self.objmask)**2)
        l_mask = 0.3*l_rendMask + 0.6*l_gtMask + 0.1*mask_diff
        
        # Hand & object 3D contact error 
        self.curve_rot_angle_mat = rotation_6d_to_matrix(self.curve_rot_angle)[0].to(self.device)
        l_contact = torch.zeros(1).to(self.device)        
        if '102156' not in cad_name and '103635' not in cad_name:  
            obj_contact_curve = self.obj_verts_batch[care_idx, objcontact_v, :].clone()
            smpl_contact_curve = self.smpl_verts[care_idx, handcontact_v, :].clone().detach()
            obj_contact_curve_after = torch.t(torch.mm(self.curve_rot_angle_mat, torch.t(obj_contact_curve))) + 5.0*self.curve_offset
            l_contact =  self.normalize * torch.mean((obj_contact_curve_after- smpl_contact_curve)**2)

        # Smoothing error
        nomotion_idx = list(set(list(range(0, len(self.part_motion)-1))) - set(care_idx.tolist()))
        partrot_first = self.part_motion[:-1]
        partrot_second = self.part_motion[1:]
        l_smooth = torch.mean((partrot_first - partrot_second)[np.array(nomotion_idx)]**2)

        # Motion range error 
        l_range = torch.mean(self.relu(limit_a - self.part_motion) + self.relu(self.part_motion-limit_b))
        
        # Roll, pitch constraint (except for laptop)
        if '10213' in cad_name or '9968' in cad_name:
            l_gamma = torch.zeros(1).to(self.device)
            l_alpha = torch.zeros(1).to(self.device)
        else:
            l_alpha = self.relu(-alpha-0.2)**2
            l_gamma = self.relu(torch.abs(gamma)-0.2)**2
        
        # Depth constraint
        l_depth = torch.mean((self.smpl_offset[care_idx,2].detach() - self.obj_offset[2])**2) 

        # Object size constraint
        #l_size = torch.sum(self.relu(0.1 - self.obj_scale))
        l_size = torch.mean(self.relu(self.obj_scale-0.1)**2)
        
        # Overall error
        overall_loss =  args.smpl*l_points + args.objmask*l_mask +\
                        args.depth*l_depth + args.smooth*l_smooth + args.range*l_range  +\
                        args.gamma*l_gamma + args.alpha*l_alpha +\
                        args.hfacing*(l_direction+l_direction2) +  args.contact*l_contact

        if iteration <= int(0.5*args.iter):
            overall_loss += args.center*l_mask_center
           
        if iteration > int(0.5*args.iter):
            overall_loss += (args.size*l_size )
            
        loss_meta = {}
        loss_meta['l_mask'] = args.objmask*l_mask.data.detach().cpu().numpy()
        loss_meta['l_center'] =   args.center*l_mask_center.data.detach().cpu().numpy()
        loss_meta['l_contact'] =  args.contact*l_contact.data.detach().cpu().numpy()
        loss_meta['l_depth'] =  args.depth*l_depth.data.detach().cpu().numpy()
        loss_meta['l_gamma'] =  args.gamma*l_gamma.data.detach().cpu().numpy()
        loss_meta['l_alpha'] =  args.alpha*l_alpha.data.detach().cpu().numpy()
        loss_meta['l_range'] =  args.alpha*l_range.data.detach().cpu().numpy()
        loss_meta['l_smooth'] =  args.alpha*l_smooth.data.detach().cpu().numpy()
        loss_meta['l_prior'] =  args.size*l_size.data.detach().cpu().numpy()
        loss_meta['l_direction'] = args.hfacing*(l_direction.data.detach().cpu().numpy() + l_direction2.data.detach().cpu().numpy() )
        loss_meta['l_points'] = args.smpl*l_points.data.detach().cpu().numpy()
        loss_meta['overall_loss'] = overall_loss.data.detach().cpu().item()
        loss_meta['final_loss'] = loss_meta['l_mask'] + 0.3*loss_meta['l_contact'] + loss_meta['l_depth'] + loss_meta['l_range'] + loss_meta['l_smooth'] +\
                                  loss_meta['l_gamma'] + loss_meta['l_alpha'] + loss_meta['l_prior'] + 0.3*loss_meta['l_direction']
    
    
        return overall_loss, loss_meta



    def render(self, save_folder=None):

        obj_meshes = []
        smpl_meshes = []

        for index in range(self.bs):
            smpl_verts = self.smpl_verts[index]
            obj_verts = self.obj_verts_batch[index]

            # create SPIN mesh  (opengl)
            tex = torch.ones_like(smpl_verts).unsqueeze(0)
            textures = TexturesVertex(verts_features=tex).to(self.device)
            smpl_mesh = Meshes(verts=[smpl_verts],faces=[self.smpl_faces[index]],textures=textures).detach()
            smpl_meshes.append(smpl_mesh)

            # create object mesh for diff render and visualization
            tex = torch.ones_like(obj_verts).unsqueeze(0)
            textures = TexturesVertex(verts_features=tex).to(self.device)
            obj_mesh = Meshes(verts=[obj_verts],faces=[self.obj_faces],textures=textures).detach()
            obj_meshes.append(obj_mesh)

        meshes = {'obj_mesh':obj_meshes, 'spin_mesh':smpl_meshes}
        return meshes

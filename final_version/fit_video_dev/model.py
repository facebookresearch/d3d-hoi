import torch.nn as nn
import torch
import pdb
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
from utils import (
    flow_to_image, flow_confidence
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
        #self.hmask = hmask[..., 0]
        self.objmask = objmask[..., 0]
        #self.hmask_square = hmask_square[..., 0]
        #self.objmask_square = objmask_square[..., 0]
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
        self.add_l = min(self.img_h, self.img_w)
        self.img_small = img_size_small
        self.focal = focal_len
        self.joints = joints
        self.normalize = 1.0/(0.5*(self.img_h+self.img_w))

        K = torch.from_numpy(np.array([[self.focal, 0, self.img_w/2],
                                       [0, self.focal, self.img_h/2],
                                       [0,0,1]]))
        self.K = K.float().to(self.device)

        mask_sum = torch.sum(self.objmask, [1,2])
        self.mask_skip = (mask_sum < 1.0).type(torch.float).reshape(-1,1,1).repeat(1,self.img_small,self.img_small)

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

        # local rotation angle for the parts
        #part_rot_angle = np.random.uniform(low=0.05, high=0.15, size=self.bs).astype(np.float32)
        part_rot_angle = 0.0*np.ones(self.bs, dtype=np.float32)
        self.part_rot_angle = nn.Parameter(torch.from_numpy(part_rot_angle).to(self.device))

        # global rotation angle for the object CAD
        yaw_degree = 0.0 * 180/np.pi #0.0* 180/np.pi
        rot_mat = RotateAxisAngle(yaw_degree, axis='Y').get_matrix()
        rot_mat = rot_mat[0,:3,:3].unsqueeze(0)
        ortho6d = matrix_to_rotation_6d(rot_mat)
        self.obj_rot_angle = nn.Parameter(ortho6d.to(self.device))

        # curve rotation in 3D
        yaw_degree2 = 0.0 * 180/np.pi #0.0* 180/np.pi
        rot_mat2 = RotateAxisAngle(yaw_degree2, axis='Y').get_matrix()
        rot_mat2 = rot_mat2[0,:3,:3].unsqueeze(0)
        ortho6d2 = matrix_to_rotation_6d(rot_mat2)
        self.curve_rot_angle = nn.Parameter(ortho6d2.to(self.device))
        
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        return




    def forward(self, iter, args, cad_name, start_idx, end_idx, part_idx, handcontact_v, objcontact_v, 
                obj_x_center, obj_y_center, x_ratio=50.0, y_ratio=50.0, z_ratio=50.0):

        # predefined object CAD part and axis
        self.vertexStart = self.vertexSegs[part_idx]
        self.vertexEnd = self.vertexSegs[part_idx+1]
        faceStart = self.faceSegs[part_idx]
        faceEnd = self.faceSegs[part_idx+1]
        self.rot_o = self.rot_origs[part_idx].clone()
        self.axis = self.rot_axises[part_idx].clone()
        limit_a = self.limit_as[part_idx]
        limit_b = self.limit_bs[part_idx]

        # pytorch3d -> world coordinate 
        self.rot_o[1:] *= -1
        self.axis[1:] *= -1
        
        # get object global rotation matrix from ortho6d
        self.obj_rot_angle_mat = rotation_6d_to_matrix(self.obj_rot_angle)
        self.curve_rot_angle_mat = rotation_6d_to_matrix(self.curve_rot_angle)
       
        # pitch, yaw, roll
        alpha,beta,gamma = matrix_to_euler_angles(self.obj_rot_angle_mat, ["X","Y","Z"])[0]

        # clamp part rotation according to predefined limits
        self.part_rot_params = self.part_rot_angle.clamp(-0.2, limit_b) # or replace with (-0.2, 2*np.pi)

        #self.part_rot_params[0:start_idx+1] = 0.0
        #self.part_rot_params[end_idx+1:] = 0.0


        ####################
        ## fit human mesh ##
        ####################
        self.smpl_verts = self.smpl_verts_orig.clone()

        # step 1: resize human mesh 
        smplmesh_calibrate_path = 'smplmesh-calibrate.obj'
        smplmesh_calibrate =  o3d.io.read_triangle_mesh(smplmesh_calibrate_path) # load smpl mesh
        hverts_cal = torch.from_numpy(np.asarray(smplmesh_calibrate.vertices)).float()
        human_height = 176.5 #cm
        h_diff = torch.max(hverts_cal[:,1]) - torch.min(hverts_cal[:,1])
        h_ratio = human_height / h_diff
        self.smpl_verts *= h_ratio
 
        # add x y z offsets to SMPL mesh (camera looking at positive depth z)
        smpl_offset = self.smpl_offset.reshape(-1,1,3).repeat(1,self.smpl_verts_orig.shape[1],1) # (bs, 6890, 3)
        self.smpl_verts[:,:,0] += 100.0*smpl_offset[:,:,0]
        self.smpl_verts[:,:,1] += 100.0*smpl_offset[:,:,1]
        self.smpl_verts[:,:,2] += 100.0*smpl_offset[:,:,2]  #smpl_offset[0,:,2]

    
        # compute projection matrix K
        K_batch = self.K.expand(self.smpl_verts.shape[0],-1,-1)

        # smpl prespective projection
        points_out_v = torch.bmm(self.smpl_verts, K_batch.permute(0,2,1))
        self.smpl_2d = points_out_v[...,:2] / points_out_v[...,2:]

        # loss for fitting human according to 2D points
        l_points = torch.mean(self.normalize*(self.points - self.smpl_2d)**2)


        #####################
        ## fit object mesh ##
        #####################
        self.part_rot_mat = rotation_matrix_batch(self.axis, self.part_rot_params)
        obj_verts_batch = self.obj_verts_orig.reshape(1,-1,3).repeat(self.part_rot_mat.shape[0],1,1) # (bs, ver, 3)

        # step 1: rescale object and rotation orig
        if not args.use_gt_objscale:
            assert(x_ratio == 50.0)
            assert(y_ratio == 50.0)
            assert(z_ratio == 50.0)
            sx = self.obj_scale[0] * x_ratio  # multiply by 50 prevents object scale becomes too big
            sy = self.obj_scale[0] * y_ratio
            sz = self.obj_scale[0] * z_ratio
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
        x_dim = torch.max(obj_verts_batch[0,:,0]) - torch.min(obj_verts_batch[0,:,0])
        y_dim = torch.max(obj_verts_batch[0,:,1]) - torch.min(obj_verts_batch[0,:,1])
        z_dim = torch.max(obj_verts_batch[0,:,2]) - torch.min(obj_verts_batch[0,:,2])

        # step 2: part motion
        obj_verts_batch_t1 = obj_verts_batch[:, self.vertexStart:self.vertexEnd, :] - self.rot_o
        obj_verts_batch_t2 = torch.bmm(self.part_rot_mat.to(self.device), obj_verts_batch_t1.permute(0,2,1)).permute(0,2,1) 
        obj_verts_batch[:, self.vertexStart:self.vertexEnd, :] = obj_verts_batch_t2 + self.rot_o

        # step 3: object orientation
        obj_verts_batch = torch.bmm(self.obj_rot_angle_mat[0].to(self.device).reshape(1,3,3).repeat(self.bs,1,1),
                                    obj_verts_batch.permute(0,2,1)).permute(0,2,1)

        # step 4: object offset
        self.obj_verts_batch =  obj_verts_batch + 100.0*self.obj_offset  #cm

        # compute object center loss
        obj_2d = torch.bmm(self.obj_verts_batch, K_batch.permute(0,2,1))
        self.obj_2d = (obj_2d[...,:2] / obj_2d[...,2:])  # (bs, objV, 2)
        l_mask_center = torch.zeros(1).cuda()
        if obj_x_center > 0 and obj_y_center > 0:
            obj_2d_x_center = torch.mean(self.obj_2d[:,:,0])
            obj_2d_y_center = torch.mean(self.obj_2d[:,:,1])
            l_mask_center = self.normalize*(obj_x_center - obj_2d_x_center)**2 +  self.normalize*(obj_y_center - obj_2d_y_center)**2

        # compute object + human orientation loss
        obj_norm = torch.from_numpy(np.asarray([0,0,1])).to(self.device)
        obj_norm2 = torch.from_numpy(np.asarray([0,-1,0])).to(self.device)
        obj_norm_rot = torch.mm(self.obj_rot_angle_mat[0], obj_norm.float().reshape(-1,1)).permute(1,0)
        obj_norm_rot2 = torch.mm(self.obj_rot_angle_mat[0], obj_norm2.float().reshape(-1,1)).permute(1,0)

        relunn = nn.ReLU()
        output = self.cos(self.normal, obj_norm_rot.repeat(self.bs, 1))
        output2 = self.cos(self.normal2, obj_norm_rot2.repeat(self.bs, 1))
        l_direction = torch.mean((1.0 - output)[start_idx:end_idx+1])   # look at middle interval
        l_direction2 = torch.mean((1.0 - output2)[start_idx:end_idx+1])

        # compute object mask loss
        diff_images = []
        for index, part_theta in enumerate(self.part_rot_params):
            # convert object mesh for diff render, opengl -> pytorch3d
            p3d_obj_verts = self.obj_verts_batch[index].clone()
            p3d_obj_verts[:,1] *= -1
            p3d_obj_verts[:,2] *= -1

            # pass through diff render
            tex = torch.ones_like(p3d_obj_verts).unsqueeze(0)
            textures = TexturesVertex(verts_features=tex).to(self.device)
            obj_mesh = Meshes(verts=[p3d_obj_verts],faces=[self.obj_faces],textures=textures)
            # !!! remove pix_to_face !!!
            # diff_img, pix_to_face = self.diff_render(meshes_world=obj_mesh, R=self.R, T=self.T)  # !!! remove pix_to_face !!!
            diff_img = self.diff_render(meshes_world=obj_mesh, R=self.R, T=self.T)  # !!! remove pix_to_face !!!
            # !!! remove pix_to_face !!!
            diff_img = diff_img[..., 3:]
            diff_img = diff_img.permute(0,3,1,2)[0,0,:,:] #(h,w)
            diff_images.append(diff_img)
           
        diff_images = torch.stack(diff_images)  #(bs,h,w)
        l_mask = torch.mean((1-self.mask_skip)*(diff_images-self.objmask)**2)


        ###################################
        ## hand & object 3D contact loss ##
        ###################################
        l_contact = torch.zeros(1).cuda()
        #l_curve = torch.zeros(1).cuda()
        
        if objcontact_v > 0 and '12477' not in cad_name and '102156' not in cad_name and '103635' not in cad_name:  # skip cad 10584
            
            #l_contact = torch.mean((self.smpl_verts[start_idx:end_idx+1, handcontact_v-1, :].detach()/100.0 -\
                                    #self.obj_verts_batch[start_idx:end_idx+1, objcontact_v-1, :]/100.0)**2)


            ##########################################
            ## hand & object contact curve matching ##
            ##########################################

            obj_contact_curve = self.obj_verts_batch[start_idx:end_idx+1, objcontact_v-1, :]/100.0
            smpl_contact_curve = self.smpl_verts[start_idx:end_idx+1, handcontact_v-1, :]/100.0
            #smpl_contact_curve = self.smpl_verts_orig[start_idx:end_idx+1, handcontact_v-1, :] - self.joints[start_idx:end_idx+1, 39, :].squeeze()
            #smpl_contact_curve = self.smpl_verts[start_idx:end_idx+1, handcontact_v-1, :]/100.0 - self.smpl_verts[start_idx:end_idx+1, 1330-1, :]/100.0
            #smpl_contact_curve *= h_ratio

            #obj_contact_trans = torch.t(torch.mm(self.curve_rot_angle_mat[0].cuda(), torch.t(obj_contact_curve)))  # rotate curve
            offset = smpl_contact_curve[0].detach() - obj_contact_curve[0]
            obj_contact_aligned = obj_contact_curve + offset.detach()
            l_contact =  torch.mean((obj_contact_aligned - smpl_contact_curve.detach())**2)
            
            #offset = smpl_contact_curve[0].detach() - obj_contact_curve[0]
            #obj_contact_aligned = obj_contact_curve + offset.detach()
            #obj_contact_trans = torch.t(torch.mm(self.curve_rot_angle_mat[0].cuda(), torch.t(obj_contact_aligned)))  # rotate curve
            #l_contact =  torch.mean((obj_contact_trans - smpl_contact_curve.detach())**2)


        # correct part rotation direction loss
        l_rot_possible = torch.sum(relunn(-self.part_rot_params))

        # roll pitch constraint (except for laptop)
        if '10213' in cad_name:
            l_gamma = torch.zeros(1).cuda()
            l_alpha = torch.zeros(1).cuda()
        else:
            l_alpha = relunn(torch.abs(alpha)-0.3)**2
            l_gamma = relunn(torch.abs(gamma)-0.3)**2

        # depth constraint
        l_depth = torch.mean((self.smpl_offset[start_idx:end_idx+1,2] - self.obj_offset[2])**2) #+\
                  #torch.mean((self.smpl_offset[start_idx:end_idx+1,1] - self.obj_offset[1])**2) +\
                  #torch.mean((self.smpl_offset[start_idx:end_idx+1,0] - self.obj_offset[0])**2)

        # object size constraint
        l_prior = torch.sum(relunn(0.1 - self.obj_scale[0]))
        
        # overall loss
        overall_loss = args.objmask*l_mask + args.smpl*l_points + args.hfacing*(l_direction+ l_direction2) +\
                       args.center*l_mask_center + args.correctrot*l_rot_possible + args.contact*l_contact +\
                       args.depth*l_depth + args.gamma*l_gamma + args.alpha*l_alpha + args.size*l_prior  
        
        loss_meta = {}
        loss_meta['l_direction1'] = args.hfacing*l_direction.data.detach().cpu().numpy() # object loss
        loss_meta['l_direction2'] = args.hfacing*l_direction2.data.detach().cpu().numpy()
        loss_meta['l_points'] = args.smpl*l_points.data.detach().cpu().numpy()
        loss_meta['overall_loss'] = overall_loss.data.detach().cpu().numpy()
        loss_meta['obj_gamma'] = gamma.data.detach().cpu().numpy()   # object orientation
        loss_meta['obj_alpha'] = alpha.data.detach().cpu().numpy()
        loss_meta['obj_beta'] = beta.data.detach().cpu().numpy()
        loss_meta['obj_x'] = self.obj_offset[0].data.detach().cpu().numpy() # object location
        loss_meta['obj_y'] = self.obj_offset[1].data.detach().cpu().numpy()
        loss_meta['obj_z'] = self.obj_offset[2].data.detach().cpu().numpy()
        loss_meta['obj_xdim'] = x_dim.data.detach().cpu().numpy() # object size
        loss_meta['obj_ydim'] = y_dim.data.detach().cpu().numpy()
        loss_meta['obj_zdim'] = z_dim.data.detach().cpu().numpy()  

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

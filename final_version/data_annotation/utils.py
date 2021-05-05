import numpy as np
import torch
import natsort
import glob
import open3d as o3d
# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras,RasterizationSettings,
    MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, DirectionalLights,
    PerspectiveCameras
)
from pytorch3d.io import save_obj
import math
import cv2
import matplotlib.pyplot as plt
import os
import imageio
from decimal import Decimal
import pdb
import json
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d
from scipy.ndimage.filters import gaussian_filter1d
from numpy.linalg import svd
from multiprocessing import Pool, Manager, cpu_count
from pytorch3d.transforms import Rotate, Translate
from matplotlib.image import imsave
from pathlib import Path




def planeFit(points):
    """
    p, n = planeFit(points)
    Given an array, points, of shape (d,...)
    representing points in d-dimensional space,
    fit an d-dimensional plane to the points.
    Return a point, p, on the plane (the point-cloud centroid),
    and the normal, n.
    """
    points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
    ctr = points.mean(axis=1)
    x = points - ctr[:,np.newaxis]
    M = np.dot(x, x.T) # Could also use np.cov(x) here.
    return ctr, svd(M)[0][:,-1]




def initialize_render2(device, focal_x, focal_y, img_square_size, img_small_size):
    """ initialize camera, rasterizer, and shader. """
    # Initialize an OpenGL perspective camera.
    #cameras = FoVPerspectiveCameras(znear=1.0, zfar=9000.0, fov=20, device=device)
    #cameras = FoVPerspectiveCameras(device=device)
    #cam_proj_mat = cameras.get_projection_transform()
    img_square_center = int(img_square_size/2)
    shrink_ratio = int(img_square_size/img_small_size)
    focal_x_small = int(focal_x/shrink_ratio)
    focal_y_small = int(focal_y/shrink_ratio)
    img_small_center = int(img_small_size/2)

    camera_sfm = PerspectiveCameras(
                focal_length=((focal_x, focal_y),),
                principal_point=((img_square_center, img_square_center),),
                image_size = ((img_square_size, img_square_size),),
                device=device)

    camera_sfm_small = PerspectiveCameras(
                focal_length=((focal_x_small, focal_y_small),),
                principal_point=((img_small_center, img_small_center),),
                image_size = ((img_small_size, img_small_size),),
                device=device)

    # To blend the 100 faces we set a few parameters which control the opacity and the sharpness of
    # edges. Refer to blending.py for more details.
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 256x256. To form the blended image we use 100 faces for each pixel. We also set bin_size and max_faces_per_bin to None which ensure that
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
    # the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=img_small_size,
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
        faces_per_pixel=100,
    )

    # Create a silhouette mesh renderer by composing a rasterizer and a shader.
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera_sfm_small,
            raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )


    # We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=img_square_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # We can add a point light in front of the object.
    lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
    #lights = DirectionalLights(device=device, direction=((0, 0, 1),))
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera_sfm,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(device=device, cameras=camera_sfm, lights=lights)
    )

    return silhouette_renderer, phong_renderer



def initialize_render(device, focal_len, img_square_size):
    """ initialize camera, rasterizer, and shader. """
    # Initialize an OpenGL perspective camera.
    #cameras = FoVPerspectiveCameras(znear=1.0, zfar=9000.0, fov=20, device=device)
    #cameras = FoVPerspectiveCameras(device=device)
    #cam_proj_mat = cameras.get_projection_transform()
    img_square_center = int(img_square_size/2)

    camera_sfm = PerspectiveCameras(
                focal_length=((focal_len, focal_len),),
                principal_point=((img_square_center, img_square_center),),
                image_size = ((img_square_size, img_square_size),),
                device=device)

    # We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=img_square_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # We can add a point light in front of the object.
    #lights = PointLights(device=device, location=((2.0, 2.0, -5.0),))
    lights = DirectionalLights(device=device, direction=((0, 0, 1),))
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera_sfm,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(device=device, cameras=camera_sfm, lights=lights)
    )

    return phong_renderer





def merge_meshes(obj_path, device):
    """ helper function for loading and merging meshes. """
    verts_list = torch.empty(0,3)
    faces_list = torch.empty(0,3).long()
    num_vtx = [0]
    num_faces = [0]

    # merge meshes, load in ascending order
    meshes = natsort.natsorted(glob.glob(obj_path+'/final/*_rescaled_sapien.obj'))

    for part_mesh in meshes:
        #print('loading %s' %part_mesh)
        mesh = o3d.io.read_triangle_mesh(part_mesh)
        verts = torch.from_numpy(np.asarray(mesh.vertices)).float()
        faces = torch.from_numpy(np.asarray(mesh.triangles)).long()
        faces = faces + verts_list.shape[0]
        verts_list = torch.cat([verts_list, verts])
        faces_list = torch.cat([faces_list, faces])
        num_vtx.append(verts_list.shape[0])
        num_faces.append(faces_list.shape[0])

    verts_list = verts_list.to(device)
    faces_list = faces_list.to(device)

    return verts_list, faces_list, num_vtx, num_faces



def load_motion(motions, device):
    """ load rotation axis, origin, and limit. """
    rot_origin = []
    rot_axis = []
    rot_type = []
    limit_a = []
    limit_b = []
    contact_list = []

    # load all meta data
    for idx, key in enumerate(motions.keys()):
        jointData = motions[key]

        # if contains movable parts
        if jointData is not None:
            origin = torch.FloatTensor(jointData['axis']['origin']).to(device)
            axis = torch.FloatTensor(jointData['axis']['direction']).to(device)
            mobility_type = jointData['type']
            contact_list.append(jointData['contact'])

            # convert to radians if necessary
            if mobility_type == 'revolute':
                mobility_a = math.pi*jointData['limit']['a'] / 180.0
                mobility_b = math.pi*jointData['limit']['b'] / 180.0
            else:
                assert mobility_type == 'prismatic'
                mobility_a = jointData['limit']['a']
                mobility_b = jointData['limit']['b']

            rot_origin.append(origin)
            rot_axis.append(axis)
            rot_type.append(mobility_type)
            limit_a.append(mobility_a)
            limit_b.append(mobility_b)

    return rot_origin, rot_axis, rot_type, limit_a, limit_b, contact_list




def visualize(mask, image, alpha):
    #mask = np.repeat(mask[:,:,np.newaxis], 3, axis=2)
    mask_img_blend = cv2.addWeighted(mask, alpha, image.astype(np.float32), 1.0-alpha, 0)
    mask_img_blend = mask_img_blend*mask + image*(1-mask)
    return mask_img_blend



def visualize_curve(data, output, save_folder, title):

    mask_model = output['obj_mask']
    spin_points = output['spin_points']

    # plot curve
    obj_curve = output['obj_curve']
    spin_curve = output['spin_curve']

    x_offset = spin_curve[0,0] - obj_curve[0,0]
    y_offset = spin_curve[0,1] - obj_curve[0,1]
    z_offset = spin_curve[0,2] - obj_curve[0,2]
    obj_curve[:,0] += x_offset
    obj_curve[:,1] += y_offset
    obj_curve[:,2] += z_offset


    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #obj_curves = obj_curve_norm
    ax.scatter(spin_curve[0,0], spin_curve[0,1], spin_curve[0,2], color='red')
    ax.scatter(obj_curve[0,0], obj_curve[0,1], obj_curve[0,2], color='red')
    ax.plot(obj_curve[:,0],  obj_curve[:,1], obj_curve[:,2], label='object curve')
    ax.plot(spin_curve[:,0],  spin_curve[:,1], spin_curve[:,2], label='hand curve')
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def save_mesh(id):
    global obj_verts_dict
    global obj_faces_dict
    global human_verts_dict
    global human_faces_dict
    global save_path_mesh

    verts1 = obj_verts_dict[str(id+1)]
    verts2 = human_verts_dict[str(id+1)]
    faces1 = obj_faces_dict[str(id+1)]
    faces2 = human_faces_dict[str(id+1)]
    verts = np.concatenate((verts1, verts2), axis=0)
    faces = np.concatenate((faces1, faces2 + verts1.shape[0]), axis=0)

    path = os.path.join(save_path_mesh, str(id+1)+'_object.obj')
    save_obj(path, torch.from_numpy(verts1), torch.from_numpy(faces1))
    path = os.path.join(save_path_mesh, str(id+1)+'_person.obj')
    save_obj(path, torch.from_numpy(verts2), torch.from_numpy(faces2))
    path = os.path.join(save_path_mesh, str(id+1)+'_joint.obj')
    save_obj(path, torch.from_numpy(verts), torch.from_numpy(faces))


def save_meshes(meshes, save_folder, video_name, title):
    global obj_verts_dict
    global obj_faces_dict
    global human_verts_dict
    global human_faces_dict
    global save_path_mesh

    save_path_mesh = os.path.join(save_folder, title)
    if not os.path.exists(save_path_mesh):
        os.makedirs(save_path_mesh)

    obj_meshes = meshes['obj_mesh']
    spin_meshes = meshes['spin_mesh']

    #  merge object + SPIN meshes
    obj_verts = {}
    obj_faces = {}
    human_verts = {}
    human_faces = {}

    for idx in range(len(obj_meshes)):
        obj_verts[str(idx+1)] = obj_meshes[idx].verts_list()[0].detach().cpu().numpy()
        obj_faces[str(idx+1)] = obj_meshes[idx].faces_list()[0].detach().cpu().numpy()
        human_verts[str(idx+1)] = spin_meshes[idx].verts_list()[0].detach().cpu().numpy()
        human_faces[str(idx+1)] = spin_meshes[idx].faces_list()[0].detach().cpu().numpy()

    manager = Manager()
    obj_verts_dict = manager.dict(obj_verts)
    obj_faces_dict =  manager.dict(obj_faces)
    human_verts_dict = manager.dict(human_verts)
    human_faces_dict =  manager.dict(human_faces)

    ids = [item for item in range(len(obj_meshes))]
    pool = Pool(processes=12)
    pool.map(save_mesh, ids)


    '''
    eft_cmd = 'python -m demo.demo_bodymocap --render wire --bg rgb --videoname '+video_name+' --vPath '+save_folder
    os.chdir('/home/xuxiangx/research/eft')
    os.system(eft_cmd)

    save_path = os.path.join(save_folder, 'eft', 'front')
    ffmpeg_cmd = 'ffmpeg -r 3 -i '+save_path+'/scene_%08d.jpg '+save_folder+'/frontview.mp4'
    os.system(ffmpeg_cmd)

    save_path = os.path.join(save_folder, 'eft', 'side')
    ffmpeg_cmd = 'ffmpeg -r 3 -i '+save_path+'/scene_%08d.jpg '+save_folder+'/sideview.mp4'
    os.system(ffmpeg_cmd)
    '''

    return



def save_parameters(model, save_folder, title):
    save_path = os.path.join(save_folder, title)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    obj_offset = 1000.0*model.obj_offset.detach().cpu().numpy()         #(3,)
    smpl_offset = 1000.0*model.smpl_offset.detach().cpu().numpy()       #(bs,3)
    obj_scale = 3000.0*model.obj_scale
    smpl_scale = 3000.0
    focal_len = model.focal
    part_rot_angle = model.part_rot_params.detach().cpu().numpy()       #(bs,)
    obj_rot_mat = model.obj_rot_angle_mat[0].detach().cpu().numpy()     #(3,3)
    part_rot_mat = model.part_rot_mat.detach().cpu().numpy()            #(bs,3,3)
    K_mat = model.K.detach().cpu().numpy()                              #(3,3)
    rot_o = model.rot_o.detach().cpu().numpy()                          #(3,)
    rot_axis = model.axis.detach().cpu().numpy()                        #(3,)


    parameters = {}
    parameters['obj_offset'] = obj_offset
    parameters['smpl_offset'] = smpl_offset
    parameters['obj_scale'] = obj_scale
    parameters['smpl_scale'] = smpl_scale
    parameters['focal_length'] = focal_len
    parameters['part_rot_angle'] = part_rot_angle
    parameters['obj_rot_matrix'] = obj_rot_mat
    parameters['part_rot_matrix'] = part_rot_mat
    parameters['K_matrix'] = K_mat
    parameters['rot_origin'] = rot_o
    parameters['rot_axis'] = rot_axis

    np.save(os.path.join(save_path, 'parameters.npy'), parameters)
    return





def save_img(idx):

    global shared_dict1
    global shared_dict2
    global save_path

    roi_image = shared_dict1['image'].permute(0,2,3,1)[idx].numpy()
    silhouette = shared_dict1['objmask'].permute(0,2,3,1)[idx]
    mask_model = shared_dict2['obj_mask']
    gt_points = shared_dict1['smplv2d']
    spin_points = shared_dict2['spin_points']

    silhouette_init = mask_model.detach().cpu().squeeze()[idx].numpy()
    mask_img_blend = visualize(silhouette_init, roi_image, 0.8)


    # save image
    #plt.subplots_adjust(hspace = 0.2, left=0.01, right=0.99, top=0.95, bottom=0.05)
    imsave(os.path.join(save_path, str(idx)+'.png'), mask_img_blend)

    return



def save_imgs(data, output, save_folder):

    global shared_dict1
    global shared_dict2
    global save_path

    save_path = save_folder

    manager = Manager()
    shared_dict1 = manager.dict(data)
    shared_dict1  = data
    shared_dict2 = manager.dict(output)
    shared_dict2 = output

    ids = [item for item in range(data['image'].shape[0])]
    pool = Pool(processes=12)
    pool.map(save_img, ids)


    #sceneflow = shared_dict2['sceneflow']
    #objSurfaceFlow = shared_dict2['objSurfaceFlow']
    #synFlow = shared_dict2['synFlow']
    #sceneflowMaskSquareShrink = shared_dict2['sceneflowMaskSquareShrink']
    #part_diff_images = shared_dict2['part_diff_images']


    # save object part suface raft flow visualization
    #save_path = os.path.join(save_folder, title, 'objSurfaceFlow')
    #if not os.path.exists(save_path):
        #os.makedirs(save_path)
    #for idx in range(objSurfaceFlow.shape[0]):
        #imsave(os.path.join(save_path, str(idx)+'.png'), objSurfaceFlow[idx])
    #ffmpeg_cmd = 'ffmpeg -r 3 -i '+save_path+'/%d.png '+save_folder+'/objSurfaceFlow.mp4'
    #os.system(ffmpeg_cmd)

    # save synthetic rendering flow visualization
    #save_path = os.path.join(save_folder, title, 'synFlow')
    #if not os.path.exists(save_path):
        #os.makedirs(save_path)
    #for idx in range(synFlow.shape[0]):
        #imsave(os.path.join(save_path, str(idx)+'.png'), synFlow[idx])
    #ffmpeg_cmd = 'ffmpeg -r 3 -i '+save_path+'/%d.png '+save_folder+'/synFlow.mp4'
    #os.system(ffmpeg_cmd)

    # save visualize images
    #for idx in range(data['image'].shape[0]):
        #save_img(idx, shared_dict1, shared_dict2)
    #save_path = os.path.join(save_folder, title, 'render')
    #ffmpeg_cmd = 'ffmpeg -r 3 -i '+save_path+'/%d.png '+save_folder+'/render.mp4'
    #os.system(ffmpeg_cmd)


    #vid1 = os.path.join(save_folder, 'objSurfaceFlow.mp4')
    #vid2 = os.path.join(save_folder, 'synFlow.mp4')
    #vid3 = os.path.join(save_folder, 'visual.mp4')
    #ffmpeg_cmd = 'ffmpeg -i '+vid1+' -i '+vid2+' -filter_complex hstack=inputs=2 '+vid3
    #os.system(ffmpeg_cmd)


    return






def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = axis / torch.sqrt(torch.dot(axis, axis))
    a = torch.cos(theta / 2.0)
    b, c, d = -axis * torch.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rot_mat = torch.empty(3,3)

    rot_mat[0,0] = aa + bb - cc - dd
    rot_mat[0,1] = 2 * (bc + ad)
    rot_mat[0,2] = 2 * (bd - ac)

    rot_mat[1,0] = 2 * (bc - ad)
    rot_mat[1,1] = aa + cc - bb - dd
    rot_mat[1,2] = 2 * (cd + ab)

    rot_mat[2,0] = 2 * (bd + ac)
    rot_mat[2,1] = 2 * (cd - ab)
    rot_mat[2,2] = aa + dd - bb - cc

    return rot_mat


def rotation_matrix_batch(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = axis / torch.sqrt(torch.dot(axis, axis))
    a = torch.cos(theta / 2.0)
    b = -axis[0] * torch.sin(theta / 2.0)
    c = -axis[1] * torch.sin(theta / 2.0)
    d = -axis[2] * torch.sin(theta / 2.0)

    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

    rot_mat = torch.empty(aa.shape[0],3,3)

    rot_mat[:,0,0] = aa + bb - cc - dd
    rot_mat[:,0,1] = 2 * (bc + ad)
    rot_mat[:,0,2] = 2 * (bd - ac)

    rot_mat[:,1,0] = 2 * (bc - ad)
    rot_mat[:,1,1] = aa + cc - bb - dd
    rot_mat[:,1,2] = 2 * (cd + ab)

    rot_mat[:,2,0] = 2 * (bd + ac)
    rot_mat[:,2,1] = 2 * (cd - ab)
    rot_mat[:,2,2] = aa + dd - bb - cc

    return rot_mat



def flow_confidence(threshold, forwardFlow, backwardFlow, img_w, img_h):
    # I_t -> I_(t+1), wrap with forward flow
    It1 = forwardFlow.clone()
    It1[:,:,0] += torch.arange(img_w)
    It1[:,:,1] += torch.arange(img_h).unsqueeze(1) # (u, v) coordinate
    It1 = torch.round(It1)
    withinFrameMask = (It1[:,:,0] < img_w) * (It1[:,:,0] > 0) *\
                      (It1[:,:,1] < img_h) * (It1[:,:,1] > 0)
    pdb.set_trace()
    withinFrameCoord = torch.nonzero(withinFrameMask==1) # (x, y) coordinate of within frame flow
    nextCoord = It1[withinFrameCoord[:, 0], withinFrameCoord[:,1]].astype(int)  # u, v order

    # I_(t+1) -> I_t, wrap back with backward flow
    nextCoordBackwardFlow =  backwardFlow[nextCoord[:,1], nextCoord[:,0],:]
    nextbackCoord = nextCoord + nextCoordBackwardFlow  # u, v coord
    nextbackCoord[:,[1,0]] = nextbackCoord[:,[0,1]] # swap to x,y coord

    # filter out noisy flow
    stableFlowMask = np.sum(np.abs(nextbackCoord - withinFrameCoord), 1) < threshold
    stableFlowCoord = withinFrameCoord[stableFlowMask]  # (x,y) coord

    return stableFlowCoord




def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = 40.0#,
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)

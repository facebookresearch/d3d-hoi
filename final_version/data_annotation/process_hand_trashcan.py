import os
import pdb
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import re
import glob
from PIL import Image
import open3d as o3d
import argparse





parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int)
parser.add_argument('--end', type=int)
parser.add_argument('--data_path', type=str)
args = parser.parse_args()


data_path = args.data_path



global obj_x_center, obj_y_center, obj_diameter, obj_size, valid_obj, flow, predictor, coco_metadata, hmask
global righthand, righthand3d, lefthand, lefthand3d
global current_imgname
global objmask_merge

def calculate_center(bb):
    return [(bb[0] + bb[2])/2, (bb[1] + bb[3])/2]


def filter_object(obj_dets, hand_dets):
    object_cc_list = [] # object center list
    for j in range(obj_dets.shape[0]):
        object_cc_list.append(calculate_center(obj_dets[j,:4]))
    object_cc_list = np.array(object_cc_list)

    img_obj_id = [] # matching list
    for i in range(hand_dets.shape[0]):
        if hand_dets[i, 5] <= 0: # if hand is non-contact
            img_obj_id.append(-1)
            continue
        else: # hand is in-contact
            hand_cc = np.array(calculate_center(hand_dets[i,:4])) # hand center points
            point_cc = np.array([(hand_cc[0]+hand_dets[i,6]*10000*hand_dets[i,7]), (hand_cc[1]+hand_dets[i,6]*10000*hand_dets[i,8])]) # extended points (hand center + offset)
            dist = np.sum((object_cc_list - point_cc)**2,axis=1)
            dist_min = np.argmin(dist) # find the nearest
            img_obj_id.append(dist_min)

    return img_obj_id



def draw_bbox(img, dets, bboxsize, insertext=None, insertext2=None):

    img_h = img.shape[0]
    img_w = img.shape[1]

    center_y = int(0.5*(dets[1]+dets[3]))
    center_x = int(0.5*(dets[0]+dets[2]))
    center_x_low = max(0, center_x - bboxsize)
    center_x_high = min(img_w-1, center_x + bboxsize)
    center_y_low = max(0, center_y - bboxsize)
    center_y_high = min(img_h-1, center_y + bboxsize)

    img[center_y_low:center_y_high, center_x_low:center_x_high, 0] = 255
    img[center_y_low:center_y_high, center_x_low:center_x_high, 1] = 0
    img[center_y_low:center_y_high, center_x_low:center_x_high, 2] = 0

    if insertext is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale  = 0.7
        fontColor  = (0,0,255)
        lineType = 2

        textLocation = (center_x+bboxsize+5, center_y+bboxsize+5)
        cv2.putText(img, insertext,
            textLocation,
            font,
            fontScale,
            fontColor,
            lineType)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale  = 0.7
    fontColor  = (255,0,0)
    lineType = 2
    textLocation = (10, 40)
    cv2.putText(img, insertext2,
        textLocation,
        font,
        fontScale,
        fontColor,
        lineType)

    frame_idx = int(current_imgname.split('/images-')[1].split('.jpg')[0])
    video_folder = current_imgname.split('/frames')[0]
    saved_folder = os.path.join(video_folder, '100DOH')
    if not os.path.exists(saved_folder):
        os.makedirs(saved_folder)

    txt_file = os.path.join(saved_folder, '100DOH.txt')
    file1 = open(txt_file, "a")  # append mode
    file1.write('frame_id:{:d} hand:{:s}\n'.format(frame_idx, insertext))
    file1.close()

    img_file = os.path.join(saved_folder, str(frame_idx).zfill(4)+'.jpg')
    plt.imsave(img_file, img)

    return



def gather_info(image, handdet, objdet, bboxsize):
    img_h = image.shape[0]
    img_w = image.shape[1]

    # hand mask center
    hcenter_y = int(0.5*(handdet[1]+handdet[3]))
    hcenter_x = int(0.5*(handdet[0]+handdet[2]))
    hcenter_x_low = max(0, hcenter_x - bboxsize)
    hcenter_x_high = min(img_w-1, hcenter_x + bboxsize)
    hcenter_y_low = max(0, hcenter_y - bboxsize)
    hcenter_y_high = min(img_h-1, hcenter_y + bboxsize)

    # distacne to two hand (smpl)
    d_rightwrist = np.sqrt((hcenter_y - righthand[1])**2 + (hcenter_x - righthand[0])**2)
    d_leftwritst = np.sqrt((hcenter_y - lefthand[1])**2 + (hcenter_x - lefthand[0])**2)
    wrist_diff = np.abs(d_rightwrist-d_leftwritst)

    if wrist_diff > 30.0:
        if (d_leftwritst < d_rightwrist):
            lr = 'left'
        else:
            lr = 'right'
    else:
        if(righthand3d[2] < lefthand3d[2]):
            lr ='right'
        else:
            lr ='left'


    # 2d mask ratios
    objcenter_y = int(0.5*(objdet[1]+objdet[3]))
    objcenter_x = int(0.5*(objdet[0]+objdet[2]))
    obj_len_w = objdet[2] - objdet[0]
    obj_len_h = objdet[3] - objdet[1]
    objcenter_ratio = -1
    objsize_ratio = -1
    withinmask = 0
    mask_coverage = 0

    if valid_obj:
        objcenter_offset = np.sqrt((obj_x_center - objcenter_x)**2 +\
                                   (obj_y_center - objcenter_y)**2)
        objcenter_ratio = objcenter_offset/obj_diameter
        objsize_ratio = (obj_len_w*obj_len_h)/obj_size
        mask_coverage = np.mean(objmask_merge[hcenter_y_low:hcenter_y_high,hcenter_x_low:hcenter_x_high])

    objMask = np.repeat((1-hmask)[:,:,np.newaxis], 2, axis=2)
    flowSum = np.sum(np.abs(flow),2)
    largeFlow = flowSum>0.5
    largeFlow = np.repeat(largeFlow[:,:,np.newaxis], 2, axis=2)
    objFlow = flow * objMask * largeFlow # ignore small flow
    handFlow = flow[hcenter_y, hcenter_x]
    handFlow_len = np.sqrt(handFlow[0]**2 +handFlow[1]**2)
    objFlow_bbox_sum = np.sum(objFlow, (0,1))
    objFlow_bbox_nonzero = np.sum(np.sum(np.abs(objFlow),2)>0)
    objFlow_avg = objFlow_bbox_sum / (1e-5+objFlow_bbox_nonzero)
    flow_dist = np.sqrt((handFlow[0] - objFlow_avg[0])**2 +\
                        (handFlow[1] - objFlow_avg[1])**2)
    flow_dist /= handFlow_len
    if valid_obj:
        #print(mask_coverage)
        #print(flow_dist)
        #print(objcenter_ratio)
        #print(objsize_ratio)
        #print('-----------')
        withinmask = mask_coverage >= 0.1

    return objcenter_ratio, objsize_ratio, flow_dist, withinmask, lr, mask_coverage




videos = sorted([(f.name, f.path) for f in os.scandir(data_path)])[args.start:args.end]

# loop through all videos
for idx, video in enumerate(videos):
    vidname = video[0]
    vidpath = video[1]
    viddir = os.path.join(vidpath, 'frames')
    maskdir = os.path.join(vidpath, 'masks')
    detdir = os.path.join(vidpath, 'hand_det')
    meshdir = os.path.join(vidpath, 'smplv2d')
    meshdir3d = os.path.join(vidpath, 'smplmesh')
    flowdir = os.path.join(vidpath, 'flow')

    imglist = sorted(glob.glob(os.path.join(viddir, '*.jpg')))
    handdetlist = sorted(glob.glob(os.path.join(detdir, '*handdet.npy')))
    objdetlist = sorted(glob.glob(os.path.join(detdir, '*objdet.npy')))
    hmasklist = sorted(glob.glob(os.path.join(maskdir, '*person_mask.npy')))
    objmasklist = sorted(glob.glob(os.path.join(maskdir, '*object_mask.npy')))
    meshlist = sorted(glob.glob(os.path.join(meshdir, '*.npy')))
    mesh3dlist = sorted(glob.glob(os.path.join(meshdir3d, '*.obj')))
    flowlist = sorted(glob.glob(os.path.join(flowdir, '*.npy')))

    # load overall mask
    tmp = np.array(Image.open(imglist[0]))
    img_h = tmp.shape[0]
    img_w = tmp.shape[1]

    obj_x_center = -1
    obj_y_center = -1
    obj_diameter = -1
    objmask_merge = np.zeros((img_h, img_w))
    obj_size = -1
    count = 1e-5
    for objmask in objmasklist:
        mask = np.load(objmask, allow_pickle=True)
        objmask_merge += mask
        if np.sum(mask) > 0:
            count += 1
            x, y, w, h = cv2.boundingRect(np.uint8(mask))
            obj_x_center += int(x+0.5*w)
            obj_y_center += int(y+0.5*h)
            obj_diameter += np.sqrt(w**2+h**2)
            obj_size += np.sum(mask)
    obj_x_center /= count
    obj_y_center /= count
    obj_diameter /= count
    obj_size /= count
    valid_obj = count>1e-4
    objmask_merge[objmask_merge>0] = 1

    flow_threshold =  1.0
    center_threshold =  0.9
    size_threshold =  0.1


    for idx, img in enumerate(imglist):
        '''
        omask = np.load(objmasklist[idx], allow_pickle=True).astype(np.uint8)
        if np.sum(omask)>0 and np.sum(omask)>0.8*obj_size:
            objmask_merge = omask
            x, y, w, h = cv2.boundingRect(np.uint8(omask))
            obj_x_center = int(x+0.5*w)
            obj_y_center = int(y+0.5*h)
            obj_diameter = np.sqrt(w**2+h**2)
            obj_size = np.sum(omask)
            valid_obj = True
            objmask_merge[objmask_merge>0] = 1
        '''




        current_imgname = img
        flow = np.load(flowlist[idx], allow_pickle=True).astype(np.float)
        hmask = np.load(hmasklist[idx], allow_pickle=True).astype(np.uint8)
        # expand human mask
        kernel = np.ones((10,10), np.float32)/50
        dst = cv2.filter2D(hmask, -1, kernel)
        dst[dst>0] = 1.0
        dst[dst<=0] = 0.0
        for j in range(10):
            dst = cv2.filter2D(dst, -1, kernel)
            dst[dst>0] = 1.0
            dst[dst<=0] = 0.0
        hmask = dst

        print(img)

        image = np.array(Image.open(img))
        points =  np.load(meshlist[idx], allow_pickle=True)
        righthand = points[5509]
        lefthand = points[2048]
        smpl = o3d.io.read_triangle_mesh(mesh3dlist[idx])
        verts = np.asarray(smpl.vertices)
        #faces = np.asarray(smpl.triangles)
        righthand3d = verts[5509]
        lefthand3d = verts[2048]
        handdets = np.load(handdetlist[idx], allow_pickle=True)
        objdets = np.load(objdetlist[idx], allow_pickle=True)


        if objdets.any() == None or handdets.any() == None:
            print('case 1')
        else:
            img_obj_id = filter_object(objdets, handdets) # match bboxes

            if len(handdets) == 1:
                state = handdets[0, 5]
                if state <= 2:
                    print('case 2')
                else:
                    assert(len(objdets)>=1)
                    obj_bbox = objdets[img_obj_id[0]]
                    center_ratio, size_ratio, flow_dist, withinmask, lr, mask_coverage = gather_info(image, handdets[0], obj_bbox, 80)

                    if not valid_obj:
                        print('case 3')
                    else:
                        if center_ratio < center_threshold and  size_ratio > size_threshold and withinmask and flow_dist<flow_threshold:
                            txt2 = 'center:{:.3f} size:{:3f} flow:{:3f} within:{:d} cover:{:2f}'.format(center_ratio, size_ratio, flow_dist, withinmask, mask_coverage)
                            draw_bbox(image, handdets[0], 10, lr, txt2)
                            print('case 4')
                        else:
                            print('case 5')

            else:
                assert len(handdets) == 2  # two hands

                detect_hands = []
                for hand_id, handdet in enumerate(handdets):
                    hand_obj_id = img_obj_id[hand_id]
                    if hand_obj_id >= 0 and handdet[5]>2:
                        obj_bbox = objdets[hand_obj_id]
                        center_ratio, size_ratio, flow_dist, withinmask, lr, mask_coverage = gather_info(image, handdet, obj_bbox, 80)
                        annotate={}
                        annotate['hand_id'] = hand_id
                        annotate['flow_dist'] = flow_dist
                        annotate['mask_coverage'] = mask_coverage
                        annotate['center_ratio'] = center_ratio
                        annotate['size_ratio'] = size_ratio
                        annotate['handdet'] = handdet
                        annotate['objdet'] = obj_bbox
                        annotate['withinmask'] = withinmask
                        annotate['lr'] = lr
                        detect_hands.append(annotate)

                if len(detect_hands) == 0:
                    print('case 6')

                elif len(detect_hands) == 1:
                    center_ratio = detect_hands[0]['center_ratio']
                    size_ratio = detect_hands[0]['size_ratio']
                    withinmask = detect_hands[0]['withinmask']
                    flow_dist = detect_hands[0]['flow_dist']
                    mask_coverage = detect_hands[0]['mask_coverage']
                    lr = detect_hands[0]['lr']
                    state = detect_hands[0]['handdet'][5]

                    if not valid_obj:
                        print('case 7')
                    else:
                        if center_ratio < center_threshold and  size_ratio > size_threshold and withinmask and flow_dist<flow_threshold and state>2:
                            txt2 = 'center:{:.3f} size:{:3f} flow:{:3f} within:{:d} cover:{:2f}'.format(center_ratio, size_ratio, flow_dist, withinmask, mask_coverage)
                            draw_bbox(image, detect_hands[0]['handdet'], 10, lr, txt2)
                            print('case 8')
                        else:
                            print('case 9')

                else:
                    assert len(detect_hands) == 2

                    withinmask0 = detect_hands[0]['withinmask']
                    withinmask1 = detect_hands[1]['withinmask']
                    mask_coverage0 = detect_hands[0]['mask_coverage']
                    mask_coverage1 = detect_hands[1]['mask_coverage']

                    if not valid_obj:
                        print('case 10')
                    else:
                        if withinmask0 and not withinmask1:
                            care_id = [0]
                            print('case 11')
                        elif withinmask1 and not withinmask0:
                            care_id = [1]
                            print('case 12')
                        elif not withinmask0 and not withinmask1:
                            care_id = []
                            print('case 13')
                        else:
                            assert (withinmask0 and withinmask1)
                            if mask_coverage0 > mask_coverage1:
                                care_id = [0]
                                print('case 14')
                            else:
                                care_id = [1]
                                print('case 15')


                        for ids in care_id:
                            center_ratio = detect_hands[ids]['center_ratio']
                            size_ratio = detect_hands[ids]['size_ratio']
                            state = detect_hands[ids]['handdet'][5]
                            flow_dist = detect_hands[ids]['flow_dist']
                            withinmask = detect_hands[ids]['withinmask']
                            mask_coverage = detect_hands[ids]['mask_coverage']
                            if center_ratio < center_threshold and size_ratio > size_threshold and flow_dist<flow_threshold and state>2:
                                txt2 = 'center:{:.3f} size:{:3f} flow:{:3f} within:{:d} cover:{:2f}'.format(center_ratio, size_ratio, flow_dist, withinmask, mask_coverage)
                                draw_bbox(image, detect_hands[ids]['handdet'], 10, detect_hands[ids]['lr'], txt2)




        #plt.figure(figsize=(50,50))
        #plt.imshow(image)
        #plt.show()

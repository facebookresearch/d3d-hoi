
import os
from natsort import natsorted
import pdb
import subprocess
import cv2

video_path = "/home/xuxiangx/research/johmr/code/JOHMR_data/washingmachine"
videos = natsorted([f.path for f in os.scandir(video_path)])



for video in videos:
    img_paths = natsorted([(f.name,f.path) for f in os.scandir(video)])
    for imgpath in img_paths:
        img = cv2.imread(imgpath[1])  # (1440, 1280, 3)
        img_h = img.shape[0]
        img_w = img.shape[1]

        if img_h > img_w:
            img_top = img[:int(img_h/2), :, :]
            img_bottom = img[int(img_h/2):, :, :]
        else:
            img_top = img[:, :int(img_w/2), :]
            img_bottom = img[:, int(img_w/2):, :]

        top_folder = os.path.join(video, 'annotate')
        bottom_folder = os.path.join(video, 'frames')
        if not os.path.exists(top_folder):
            os.makedirs(top_folder)
        if not os.path.exists(bottom_folder):
            os.makedirs(bottom_folder)

        cv2.imwrite(os.path.join(bottom_folder, imgpath[0]), img_top)
        cv2.imwrite(os.path.join(top_folder, imgpath[0]), img)

        os.system('rm '+imgpath[1])


import os
from natsort import natsorted
import pdb
import subprocess
import cv2

video_path = "/home/xuxiangx/research/johmr/code/JOHMR_data/microwave"
videos = natsorted([f.path for f in os.scandir(video_path)])



for video in videos:
    img_paths = natsorted([(f.name,f.path) for f in os.scandir(video+'/frames')])

    for imgpath in img_paths:
        img = cv2.imread(imgpath[1])  # (720, 1280, 3)

        img_rotate = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # cv2.ROTATE_90_CLOCKWISE     ROTATE_90_COUNTERCLOCKWISE
        assert(img_rotate.shape == (1280,720,3))

        os.system('rm '+imgpath[1])

        #img_h = img.shape[0]
        #img_w = img.shape[1]
        #if img_h > img_w:
            #assert((img_h == 1280 and img_w == 720) or (img_h == 1920 and img_w == 1080))
            #cmd = 'convert ' + imgpath[1] + ' -resize 720x1280 ' + imgpath[1]
        #else:
            #assert((img_h == 720 and img_w == 1280) or (img_h == 1080 and img_w == 1920))
            #cmd = 'convert ' + imgpath[1] + ' -resize 1280x720 ' + imgpath[1]
        #os.system(cmd)

        cv2.imwrite(imgpath[1], img_rotate)

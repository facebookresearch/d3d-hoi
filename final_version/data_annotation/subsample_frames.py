
import os
from natsort import natsorted
import pdb
import subprocess
import cv2

video_path = "/home/xuxiangx/research/johmr/code/JOHMR_data/final_data_v3/drawer"
videos = natsorted([f.path for f in os.scandir(video_path)])



for video in videos:
    print(video)
    img_paths = natsorted([(f.name,f.path) for f in os.scandir(video)])
    for idx, imgpath in enumerate(img_paths):
        # subsample
        if idx%3 == 0:
            img = cv2.imread(imgpath[1])

            folder = os.path.join(video, 'clip')
            if not os.path.exists(folder):
                os.makedirs(folder)

            cv2.imwrite(os.path.join(folder, imgpath[0]), img)

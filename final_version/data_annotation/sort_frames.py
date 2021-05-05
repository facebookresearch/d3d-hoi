
import os
from natsort import natsorted
import pdb
import subprocess


video_path = "/home/xuxiangx/research/johmr/code/JOHMR_data/washingmachine"
videos = natsorted([f.path for f in os.scandir(video_path)])


for video in videos:
    video = video#+'/annotate'
    img_paths = natsorted([(f.name,f.path) for f in os.scandir(video)])
    video_id = video.split('washingmachine/')[1]
    start_frame = img_paths[0][0].split('.jpg')[0][-4:]
    end_frame = img_paths[-1][0].split('.jpg')[0][-4:]

    file1 = open("order.txt", "a")  # append mode
    file1.write(video_id+': ')
    file1.write(start_frame+',')
    file1.write(end_frame+'\n')
    file1.close()

    for idx, img in enumerate(img_paths):
        source = os.path.join(video, img[0])
        dest = os.path.join(video,'images-'+str(idx+1).zfill(4)+'.jpg')
        cmd = 'mv '+source+' '+dest
        os.system(cmd)

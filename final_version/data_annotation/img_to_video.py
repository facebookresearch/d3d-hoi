
import os
from natsort import natsorted
import pdb
import subprocess
import cv2

video_path = "/home/xuxiangx/research/johmr/code/JOHMR_data/oven"
videos = natsorted([f.path for f in os.scandir(video_path)])



for video in videos:
    save_path = video+'/frames'
    save_video = video
    ffmpeg_cmd = 'ffmpeg -r 5 -i '+save_path+'/images-%04d.jpg -codec copy '+save_video+'/rgb.mp4'
    os.system(ffmpeg_cmd)

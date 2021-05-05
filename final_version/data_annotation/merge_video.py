import os
import pdb
import numpy as np
import argparse
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int)
parser.add_argument('--end', type=int)
args = parser.parse_args()


video_folder1 = './hfacing_all_videos'
video_folder2 = './hfacing_all_100_videos'
video_folder3 = './merge'



videos1 = sorted([(f.name, f.path) for f in os.scandir(video_folder1)])[args.start:args.end]
videos2 = sorted([(f.name, f.path) for f in os.scandir(video_folder2)])[args.start:args.end]

# loop through all videos
for idx, video in enumerate(videos1):
    vidname = video[0]
    vidpath1 = video[1]
    vidpath2 = videos2[idx][1]

    subprocess.call(['ffmpeg', '-i', os.path.join(vidpath1,vidname)+'.mp4', '-i', os.path.join(vidpath2,vidname)+'.mp4',
                             '-filter_complex', '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]', '-map',  '[vid]', os.path.join(video_folder3,vidname)+'.mp4'])

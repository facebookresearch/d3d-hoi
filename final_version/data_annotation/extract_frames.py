import os
import pdb
import subprocess
from multiprocessing import Pool
import os

video_folder = "./washingmachine"
start_name = 'a007-'
global videos
videos = sorted([f.name for f in os.scandir(video_folder)])

def extract_frames(video):
    global videos
    idx = videos.index(video)
    file_name = start_name + str(idx+1).zfill(4)
    file1 = open("order.txt", "a")  # append mode
    file1.write(file_name+': ')
    file1.write(video+'\n')
    file1.close()
    save_folder = os.path.join(video_folder, file_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_folder = os.path.join(save_folder, "images-%04d.jpg")
    subprocess.call(["ffmpeg", "-i", os.path.join(video_folder, video), "-qscale:v", "2", "-vf", "fps=3", save_folder])  # 5

pool = Pool(processes=12)
pool.map(extract_frames, videos)

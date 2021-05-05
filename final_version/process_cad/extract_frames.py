import os
import pdb
import subprocess
from multiprocessing import Pool

video_folder = "./videos"
frame_folder = "./frames"

#video_paths = [f.path for f in os.scandir(video_folder)]
videos = [(f.name, f.path) for f in os.scandir(video_folder)]


def extract_frames(video):
    save_folder = os.path.join(frame_folder, video[0])
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_folder = os.path.join(save_folder, "images-%04d.jpg")
    subprocess.call(["ffmpeg", "-i", video[1], "-vf", "fps=4", save_folder])


pool = Pool(processes=12)
pool.map(extract_frames, videos)

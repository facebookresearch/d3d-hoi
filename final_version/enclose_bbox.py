import os
import pdb
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=None)
args = parser.parse_args()
home_folder = args.path
videopath = sorted([(f.name, f.path) for f in os.scandir(home_folder)])

for video in videopath:
    print(video)
    x_min = 9999.0
    x_max = 0.0
    y_min = 9999.0
    y_max = 0.0

    mask_folder = os.path.join(video[1], 'masks')

    for file in os.listdir(mask_folder):
        if file.endswith("person_mask.npy"):
            mask = np.load(os.path.join(mask_folder, file))
            x,y,w,h = cv2.boundingRect(np.uint8(mask))
            if w == 0 or h == 0:  # skip empty mask
                continue
            x_min = min(x_min, x)
            x_max = max(x_max, x+w)
            y_min = min(y_min, y)
            y_max = max(y_max, y+h)

    if  x_max-x_min < 100 or y_max-y_min < 100:
        pdb.set_trace()
    bbox = np.asarray([x_min, y_min, x_max-x_min, y_max-y_min])
    print(x_min)
    print(y_min)
    print(x_max-x_min)
    print(y_max-y_min)
    print('-------')
    np.save(os.path.join(video[1], 'bbox'), bbox)

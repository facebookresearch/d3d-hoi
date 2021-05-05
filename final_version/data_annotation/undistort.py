import cv2
import numpy as np
import os
import glob
import pdb
import subprocess
from multiprocessing import Pool


mtx = np.load('final_data_v3/camera/intrinsic.npy')
dist = np.load('final_data_v3/camera/dist.npy')

print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)


video_folder = "final_data_v3/dishwasher"

#video_paths = [f.path for f in os.scandir(video_folder)]
videos = sorted([f.path for f in os.scandir(video_folder)])

for video in videos:
    images = sorted(glob.glob(os.path.join(video,'frames')+'/*.jpg'))
    for fname in images:
        print(fname)
        img = cv2.imread(fname)
        h,  w = img.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        cv2.imwrite('tmp1.jpg', dst)

        # crop the image
        img_saved = np.zeros_like(img)
        x,y,w,h = roi
        img_saved[y:y+h, x:x+w] = dst[y:y+h, x:x+w]

        #rm_cmd = 'rm '+fname
        #os.system(rm_cmd)

        cv2.imwrite('tmp2.jpg', img_saved)
        pdb.set_trace()

import os
import subprocess
from tqdm import tqdm
from multiprocessing import Pool

def convert(obj_path):
    try:
        load_folder = os.path.join(obj_path, 'parts_ply')
        save_folder = os.path.join(obj_path, 'parts_off')

        part_paths = [f.path for f in os.scandir(load_folder)]

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for part in part_paths:
            target_mesh = save_folder+'/'+part[-5:-3]+'off'
            subprocess.run(["meshlabserver", "-i", part, "-o", target_mesh])

    except Exception as ex:
        return 


cad_folder = './cad_sapien'
cad_classes = [f.name for f in os.scandir(cad_folder)]

for cad_category in cad_classes:

    folder_path = os.path.join(cad_folder, cad_category)
    object_paths = [f.path for f in os.scandir(folder_path)]

    # Parallel
    threads = 16  # number of threads in your computer
    convert_iter = Pool(threads).imap(convert, object_paths) 
    for _ in tqdm(convert_iter, total=len(object_paths)):
        pass


     

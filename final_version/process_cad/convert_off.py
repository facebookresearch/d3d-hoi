import os
import pdb
import subprocess

cad_folder = '/home/xuxiangx/research/johmr/dataset/final/'
cad_classes = [f.name for f in os.scandir(cad_folder)]

for cad_category in cad_classes:

    folder_path = os.path.join(cad_folder, cad_category)
    object_paths = [f.path for f in os.scandir(folder_path)]

    for obj_path in object_paths:

        print('processing %s' % obj_path)
        load_folder = os.path.join(obj_path, 'parts_ply')
        save_folder = os.path.join(obj_path, 'parts_off')

        part_paths = [f.path for f in os.scandir(load_folder)]

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for part in part_paths:
            target_mesh = save_folder+'/'+part[-5:-3]+'off'
            subprocess.run(["meshlabserver", "-i", part, "-o", target_mesh])

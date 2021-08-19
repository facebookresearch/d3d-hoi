import pdb
import subprocess
import scandir
from multiprocessing import Pool
import json
import common


def remesh(obj_path):
    in_dir = os.path.join(obj_path, 'parts_off/')
    scaled_dir = os.path.join(obj_path, 'parts_scaled_off/')
    depth_dir = os.path.join(obj_path, 'parts_depth_off/')
    fused_dir = os.path.join(obj_path, 'parts_watertight_off/')
    out_dir = os.path.join(obj_path, 'parts_out_off/')
    final_dir = os.path.join(obj_path, 'final/')
    rescale_dir = os.path.join(obj_path, 'rescale/')

    # scale to .5 cube
    subprocess.call(["python", "1_scale.py", "--in_dir", in_dir, "--out_dir", scaled_dir])

    # re-mesh using tsdf
    subprocess.call(["python", "2_fusion.py", "--mode", "render", "--in_dir", scaled_dir, "--depth_dir", depth_dir, "--out_dir", fused_dir])
    subprocess.call(["python", "2_fusion.py", "--mode", "fuse", "--in_dir", scaled_dir, "--depth_dir", depth_dir, "--out_dir", fused_dir])

    # simplify mesh
    subprocess.call(["python", "3_simplify.py", "--in_dir", fused_dir, "--out_dir", out_dir])



    if not os.path.exists(final_dir):
        os.makedirs(final_dir)

    for file in os.listdir(rescale_dir):
        if file.endswith("rescale.json"):
            with open(os.path.join(rescale_dir, file)) as json_file:

                # load rescale value
                rescale_dict = json.load(json_file)
                scales = (1.0/rescale_dict['scales'][0], 1.0/rescale_dict['scales'][1], 1.0/rescale_dict['scales'][2])
                translation = (-rescale_dict['translation'][2], -rescale_dict['translation'][1], -rescale_dict['translation'][0])

                # load mesh
                mesh = common.Mesh.from_off(os.path.join(out_dir, file[0]+'.off'))

                # apply rescaling
                mesh.scale(scales)
                mesh.translate(translation)
                mesh.to_off(os.path.join(final_dir, file[0]+'_rescaled.off'))

                # change axis
                apply_script = "change_axis.mlx"
                source_mesh = os.path.join(final_dir, file[0]+'_rescaled.off')
                target_mesh = os.path.join(final_dir, file[0]+'_rescaled_sapien.off')
                subprocess.call(["meshlabserver", "-i", source_mesh, "-o", target_mesh, "-s", apply_script])

                # convert to obj
                source_mesh = os.path.join(final_dir, file[0]+'_rescaled_sapien.off')
                target_mesh = os.path.join(final_dir, file[0]+'_rescaled_sapien.obj')
                subprocess.call(["meshlabserver", "-i", source_mesh, "-o", target_mesh])

    return



cad_folder = 'test' # cad data path (after convert_off)
cad_classes = [f.name for f in scandir.scandir(cad_folder)]
Processors = 10 # n of processors you want to use

for cad_category in cad_classes:

    folder_path = os.path.join(cad_folder, cad_category)
    object_paths = [f.path for f in scandir.scandir(folder_path)]
   
    pool = Pool(processes=Processors)
    pool.map(remesh, object_paths)

    print('All jobs finished...')

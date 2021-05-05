import os
import pdb
import numpy as np
import argparse
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int)
parser.add_argument('--end', type=int)
args = parser.parse_args()



cad_path = '/home/xuxiangx/research/johmr/code/JOHMR_data/final_cad'
anno_path = "/home/xuxiangx/research/johmr/code/JOHMR_data/final_data_fullv2"
meta_folder = '/home/xuxiangx/research/johmr/code/JOHMR_data/hfacing_all_100'
mesh_folder = '/home/xuxiangx/research/johmr/code/JOHMR_data/hfacing_all_100_mesh'


videos = sorted([(f.name, f.path) for f in os.scandir(mesh_folder)])[args.start:args.end]

# loop through all videos
for idx, video in enumerate(videos):
    vidname = video[0]
    vidpath = video[1]

    cads = sorted([(f.name, f.path) for f in os.scandir(vidpath)])

    # loop through best cad model
    for cad in cads:
        cadname = cad[0]
        cadpath = cad[1]
        settings = sorted([(f.name, f.path) for f in os.scandir(cadpath)])

        # loop through best setting
        for setting in settings:
            expname = setting[0]
            save_folder = setting[1]

            # load saved meta
            meta_file = os.path.join(meta_folder, vidname, cadname, expname, 'loss_meta.npy')
            rot_file = os.path.join(meta_folder, vidname, cadname, expname, 'partRot.npy')
            save_video = os.path.join('/home/xuxiangx/research/johmr/code/JOHMR_data/output', vidname)


            eft_cmd = 'python -m demo.demo_bodymocap --render solid --bg rgb --orig 1 --side 1 --annotate 0 --videoname '+vidname+' --vPath '+save_folder+' --metapath '+meta_file+' --rotpath '+rot_file +' --anno_path '+anno_path+' --cadpath '+cad_path+' --cadname '+cadname
            os.chdir('/home/xuxiangx/research/eft')
            os.system(eft_cmd)

            save_path = os.path.join('/home/xuxiangx/research/johmr/code/JOHMR_data/output', vidname, 'front')
            ffmpeg_cmd = 'ffmpeg -r 2 -i '+save_path+'/scene_%08d.jpg '+save_video+'/frontview_solid_rgb.mp4'
            os.system(ffmpeg_cmd)
            rm_cmd = 'rm -r '+save_path
            os.system(rm_cmd)

            save_path = os.path.join('/home/xuxiangx/research/johmr/code/JOHMR_data/output', vidname, 'side')
            ffmpeg_cmd = 'ffmpeg -r 2 -i '+save_path+'/scene_%08d.jpg '+save_video+'/sideview_solid_rgb.mp4'
            os.system(ffmpeg_cmd)
            rm_cmd = 'rm -r '+save_path
            os.system(rm_cmd)


            save_path = os.path.join('/home/xuxiangx/research/johmr/code/JOHMR_data/output', vidname, 'orig')
            ffmpeg_cmd = 'ffmpeg -r 2 -i '+save_path+'/scene_%08d.jpg '+save_video+'/orig_rgb.mp4'
            os.system(ffmpeg_cmd)
            rm_cmd = 'rm -r '+save_path
            os.system(rm_cmd)


            eft_cmd = 'python -m demo.demo_bodymocap --render wire --orig 0 --bg flow --side 1 --annotate 1 --videoname '+vidname+' --vPath '+save_folder+' --metapath '+meta_file+' --rotpath '+rot_file +' --anno_path '+anno_path+' --cadpath '+cad_path+' --cadname '+cadname
            os.system(eft_cmd)

            save_path = os.path.join('/home/xuxiangx/research/johmr/code/JOHMR_data/output', vidname, 'front')
            ffmpeg_cmd = 'ffmpeg -r 2 -i '+save_path+'/scene_%08d.jpg '+save_video+'/frontview_wire_flow.mp4'
            os.system(ffmpeg_cmd)
            rm_cmd = 'rm -r '+save_path
            os.system(rm_cmd)

            save_path = os.path.join('/home/xuxiangx/research/johmr/code/JOHMR_data/output', vidname, 'side')
            ffmpeg_cmd = 'ffmpeg -r 2 -i '+save_path+'/scene_%08d.jpg '+save_video+'/sideview_wire_flow.mp4'
            os.system(ffmpeg_cmd)
            rm_cmd = 'rm -r '+save_path
            os.system(rm_cmd)


            eft_cmd = 'python -m demo.demo_bodymocap --render wire --orig 0 --bg mask --side 0 --annotate 1 --videoname '+vidname+' --vPath '+save_folder+' --metapath '+meta_file+' --rotpath '+rot_file +' --anno_path '+anno_path+' --cadpath '+cad_path+' --cadname '+cadname
            os.system(eft_cmd)

            save_path = os.path.join('/home/xuxiangx/research/johmr/code/JOHMR_data/output', vidname, 'front')
            ffmpeg_cmd = 'ffmpeg -r 2 -i '+save_path+'/scene_%08d.jpg '+save_video+'/frontview_wire_mask.mp4'
            os.system(ffmpeg_cmd)
            rm_cmd = 'rm -r '+save_path
            os.system(rm_cmd)

            save_path = os.path.join('/home/xuxiangx/research/johmr/code/JOHMR_data/output', vidname, 'side')
            rm_cmd = 'rm -r '+save_path
            os.system(rm_cmd)


            subprocess.call(['ffmpeg', '-i', save_video+'/orig_rgb.mp4', '-i', save_video+'/frontview_solid_rgb.mp4',
                             '-filter_complex', '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]', '-map',  '[vid]', save_video+'/tmp1.mp4'])

            subprocess.call(['ffmpeg', '-i', save_video+'/frontview_wire_flow.mp4', '-i', save_video+'/sideview_solid_rgb.mp4',
                             '-filter_complex', '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]', '-map',  '[vid]', save_video+'/tmp2.mp4'])


            subprocess.call(['ffmpeg', '-i', save_video+'/frontview_wire_mask.mp4', '-i', save_video+'/sideview_wire_flow.mp4',
                             '-filter_complex', '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]', '-map',  '[vid]', save_video+'/tmp3.mp4'])



            subprocess.call(['ffmpeg', '-i', save_video+'/tmp1.mp4', '-vf',
                             'pad=iw:2*ih [top]; movie='+save_video+'/tmp2.mp4'+' [bottom];[top][bottom] overlay=0:main_h/2', save_video+'/tmp4.mp4'])

            subprocess.call(['ffmpeg', '-i', save_video+'/tmp4.mp4', '-vf',
                             'pad=iw:2*ih [top]; movie='+save_video+'/tmp3.mp4'+' [bottom];[top][bottom] overlay=0:main_h/2', save_video+'/'+vidname+'.mp4'])


            rm_cmd = 'rm -r '+save_video+'/tmp1.mp4'
            os.system(rm_cmd)
            rm_cmd = 'rm -r '+save_video+'/tmp2.mp4'
            os.system(rm_cmd)
            rm_cmd = 'rm -r '+save_video+'/tmp3.mp4'
            os.system(rm_cmd)
            rm_cmd = 'rm -r '+save_video+'/tmp4.mp4'
            os.system(rm_cmd)
            rm_cmd = 'rm -r '+save_video+'/frontview_solid_rgb.mp4'
            os.system(rm_cmd)
            rm_cmd = 'rm -r '+save_video+'/frontview_wire_flow.mp4'
            os.system(rm_cmd)
            rm_cmd = 'rm -r '+save_video+'/sideview_wire_flow.mp4'
            os.system(rm_cmd)
            rm_cmd = 'rm -r '+save_video+'/frontview_wire_mask.mp4'
            os.system(rm_cmd)
            rm_cmd = 'rm -r '+save_video+'/orig_rgb.mp4'
            os.system(rm_cmd)
            rm_cmd = 'rm -r '+save_video+'/sideview_solid_rgb.mp4'
            os.system(rm_cmd)

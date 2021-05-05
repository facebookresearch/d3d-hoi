import os
import pdb
import numpy as np
import argparse
import subprocess
import multiprocessing


global cad_path
global data_path

cad_path = '/home/xuxiangx/research/johmr/code/JOHMR_data/cad_final/dishwasher'
data_path = "/home/xuxiangx/research/johmr/code/JOHMR_data/data_v5.2/dishwasher"



def render(video):
    global cad_path
    global data_path

    vidname = video[0]
    vidpath = video[1]
    save_video = os.path.join('/home/xuxiangx/research/johmr/code/JOHMR_data/output', vidname)
    os.chdir('/home/xuxiangx/research/eft')
    
    ############################
    ##  solid rgb front + side
    eft_cmd = 'python -m demo.demo_bodymocapnew --render solid --bg rgb --orig 1 --side 1 --videoname '+vidname+' --vPath '+vidpath+' --cadpath '+cad_path
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


    ############################
    ##  wire flow front
    eft_cmd = 'python -m demo.demo_bodymocapnew --render wire --orig 0 --bg flow --side 0 --videoname '+vidname+' --vPath '+vidpath+' --cadpath '+cad_path
    os.system(eft_cmd)

    save_path = os.path.join('/home/xuxiangx/research/johmr/code/JOHMR_data/output', vidname, 'front')
    ffmpeg_cmd = 'ffmpeg -r 2 -i '+save_path+'/scene_%08d.jpg '+save_video+'/frontview_wire_flow.mp4'
    os.system(ffmpeg_cmd)
    rm_cmd = 'rm -r '+save_path
    os.system(rm_cmd)

    save_path = os.path.join('/home/xuxiangx/research/johmr/code/JOHMR_data/output', vidname, 'side')
    rm_cmd = 'rm -r '+save_path
    os.system(rm_cmd)
    



    ############################
    ##  wire rgb front
    eft_cmd = 'python -m demo.demo_bodymocapnew --render wire --orig 0 --bg rgb --side 0 --videoname '+vidname+' --vPath '+vidpath+' --cadpath '+cad_path
    os.system(eft_cmd)

    save_path = os.path.join('/home/xuxiangx/research/johmr/code/JOHMR_data/output', vidname, 'front')
    ffmpeg_cmd = 'ffmpeg -r 2 -i '+save_path+'/scene_%08d.jpg '+save_video+'/frontview_wire_rgb.mp4'
    os.system(ffmpeg_cmd)
    rm_cmd = 'rm -r '+save_path
    os.system(rm_cmd)

    save_path = os.path.join('/home/xuxiangx/research/johmr/code/JOHMR_data/output', vidname, 'side')
    rm_cmd = 'rm -r '+save_path
    os.system(rm_cmd)



    ############################
    ##  wire mask front
    eft_cmd = 'python -m demo.demo_bodymocapnew --render wire --orig 0 --bg mask --side 0 --videoname '+vidname+' --vPath '+vidpath+' --cadpath '+cad_path
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

    subprocess.call(['ffmpeg', '-i', save_video+'/frontview_wire_flow.mp4', '-i', save_video+'/frontview_wire_rgb.mp4',
                             '-filter_complex', '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]', '-map',  '[vid]', save_video+'/tmp2.mp4'])

    subprocess.call(['ffmpeg', '-i', save_video+'/frontview_wire_mask.mp4', '-i', save_video+'/sideview_solid_rgb.mp4',
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
    #rm_cmd = 'rm -r '+save_video+'/frontview_solid_rgb.mp4'
    #os.system(rm_cmd)
    #rm_cmd = 'rm -r '+save_video+'/frontview_wire_flow.mp4'
    #os.system(rm_cmd)
    #rm_cmd = 'rm -r '+save_video+'/frontview_wire_rgb.mp4'
    #os.system(rm_cmd)
    #rm_cmd = 'rm -r '+save_video+'/frontview_wire_mask.mp4'
    #os.system(rm_cmd)
    #rm_cmd = 'rm -r '+save_video+'/orig_rgb.mp4'
    #os.system(rm_cmd)
    #rm_cmd = 'rm -r '+save_video+'/sideview_solid_rgb.mp4'
    #os.system(rm_cmd)


videos = sorted([(f.name, f.path) for f in os.scandir(data_path)])
render(videos[0])
#pool = multiprocessing.Pool(processes=25)
#pool.map(render, videos)

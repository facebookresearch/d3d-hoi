#!/bin/bash
FILE_DIR="/home/xuxiangx/research/johmr/code/JOHMR_data/data_v5.2/oven"
echo "processing" $FILE_DIR"..."

# pointrend segmentation for human and object silhouettes
python /home/xuxiangx/research/johmr/code/point_rend/run.py --path $FILE_DIR

# find largest enclose human bbox
#python enclose_bbox.py --path $FILE_DIR

# EFT human mesh, human joints, 2D points estimation
#cd /home/xuxiangx/research/eft
#python -m demo.demo_bodymocapold --vPath $FILE_DIR

# estimate optical flow with RAFT
#cd /home/xuxiangx/research/RAFT
#python demo.py --model=models/raft-sintel.pth --path=$FILE_DIR

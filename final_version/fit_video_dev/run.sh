#!/bin/bash
python train.py --iter 150 --objmask 100.0 --flowmask 0.0 --size 100.0 \
                --hfacing 0.0 --gamma 50.0 --alpha 50.0 --correctrot 0.5 --smpl 0.1 --depth 1.0 \
                --contact 0.0 --synflow 0.0 --center 0.5 --category all --exp_name baseline --log log \
                --cadpath cad_final --datapath data_final --use_gt_objmodel #--use_gt_objscale  



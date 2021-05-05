This folder contains the optimization, visualization, and evaluation code 


## Optimization
```
sh run.sh 
```
Important parameters:

  --iter  training iterations 
  
  --objmask  hyperparam for the diff render mask loss
  
  --size  hyperparam for the size prior loss (prevent object from becoming negative size)
  
  --hfacing  hyperparam for human direction loss
  
  --gamma  hyperparam for gamma constraint
  
  --alpha  hyperparam for alpha constraint 
  
  --correctrot  hyperparam for the part motion loss (prevent out of limit)
  
  --smpl  hyperparam for human smpl projection loss
  
  --depth  hyperparam for depth constraint between human and object
  
  --center  hyperparam for matching object mask projection center (useful for diff render)
  
  --category  options: 'all' , 'laptop', 'dishwasher', and etc
  
  --use_gt_objmodel  use the gt object cad model + part id or not (if not, then iter through all configurations within this category)
  
  --use_gt_objscale  use fixed gt object dimension or not 
  
The following parameters should be used for baseline in the script:
```
python train.py --iter 150 --objmask 100.0 --flowmask 0.0 --size 100.0 \
                --hfacing 0.0 --gamma 50.0 --alpha 50.0 --correctrot 0.5 --smpl 0.1 --depth 1.0 \
                --contact 0.0 --synflow 0.0 --center 0.5 --category all --exp_name baseline --log log \
                --cadpath cad_final --datapath data_final --use_gt_objmodel #--use_gt_objscale  
```

The following parameters should be used for human in the script:
```
python train.py --iter 150 --objmask 100.0 --flowmask 0.0 --size 100.0 \
                --hfacing 50.0 --gamma 50.0 --alpha 50.0 --correctrot 0.5 --smpl 0.1 --depth 1.0 \
                --contact 0.0 --synflow 0.0 --center 0.5 --category all --exp_name human --log log \
                --cadpath cad_final --datapath data_final --use_gt_objmodel #--use_gt_objscale  
```

During optimization, the 'joint', 'object' and 'person' meshes will be saved for each setting in the exp_name folder. The loss and final values for object pose and other meta information will also be saved. 



## Evaluation
```
sh test.sh 
```
The output are two txt file in the exp_name folder. results.txt has the best per-video and per-category evaluation score.  file_list.txt has the configuraiton of the best setting. 


## Visualization
```
python visual.py
```
Use OpenGL to render the saved meshes (with original image in the background). The OpenGL render code is based on eft. Thus you need to make sure the path pointing to eft folder is correct. 

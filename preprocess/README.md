## Data 

Download the original [SAPIEN PartNet-Mobility Dataset](https://sapien.ucsd.edu/) 

The 24 CAD ID used in our paper is available [here](https://dl.fbaipublicfiles.com/d3d-hoi/processed_cads_id.zip).

## Usage 
This code preprocess the SAPIEN models and simplied meshes to roughly 2500 faces per object, modify path in each file so that it points to the correct folder

To preprocess the SAPIEN models:

    python process_data.py
    python convert_off.py

To simplied the meshes (requires meshlab and mesh-fusion):

    python re-meshing.py 

Re-meshing code is based on the [mesh-fusion](https://github.com/davidstutz/mesh-fusion) library, please follow their tutorial for installation.


## Visualization
To visualize the final post-processed CAD models:

    python visualize_data.py

Here are some rendered results:

<img src="example/example1.gif" width="200" height="200" />  <img src="example/example2.gif" width="200" height="200" />  <img src="example/example3.gif" width="200" height="200" />  <img src="example/example4.gif" width="200" height="200" />  <img src="example/example5.gif" width="200" height="200" />  <img src="example/example6.gif" width="200" height="200" /> <img src="example/example7.gif" width="200" height="200" /> <img src="example/example8.gif" width="200" height="200" />

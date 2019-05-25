PointCloud_Costmap_NN
============================

> This repository is created for the bachelorthesis under the subject "Representatie van een wereldmodel aan de hand van machine learning".

### Project structure

    .
    ├── data                     # Recorded datacollections of different size
    │   ├── datacollection
    │   ├── ...
    │   └── datacollection7
    ├── proto                    # Files needed to decode the datacollection
    │   ├── ...
    │   └── pointcloud_pc2.py    # File to help you parse datacollection
    ├── Calc_FP_FN.py            # Script to calc false positives and negatives in costmaps
    ├── NN_v1.py                 # Fist Feed Forward Network
    ├── NN_v2.py                 # Second Feed Forward Network with cross-validation
    ├── Test_rotate_data.py      # Script to test rotation of datasamples
    ├── Use_WM.py                # Script to use the trained models
    ├── Visualizer.py            # Script to show samples from collection
    └── README.md
    
    
### Setup
First step is to install all needed libraries. This porject is tested on Python 3.5 and tensorflow 1.13.
Next, you can run alle scripts with runparameter the datacollection you want to parse.

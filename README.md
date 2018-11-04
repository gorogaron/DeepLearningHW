# Deep Learning with Python
### Project: Video Frame Interpolation with Deep Convolutional Neural Network

The following scripts were created for the Deep Learning 2018 course from BUTE by József Kenderák and Árom Görög.

## Prerequisites
```
conda install -c conda-forge keras
conda install -c conda-forge tensorflow
conda install -c conda-forge opencv
conda install -c anaconda scipy
```

## Directory structure and files
```
data/             - The whole RAW dataset in subfolders 
model_weights/    - Wieghts of the best model
results/          - Save folder of the videos after testing
createDataset.py  - Put the whole dataset into the HDF5 file structuring by train, valid and test sets
imageGenerator.py - Train and Validation generators for *.fit_generator()
losses.py         - Defined some new Loss functions
networks.py       - Currently only contains U-net architecture
train.py          - Train the model after preprocessing
test.py           - After training we can test the network with videos
```

## HowTo
### 1. Step - Preprocessing


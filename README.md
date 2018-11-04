# Deep Learning with Python
### Project: Video Frame Interpolation with Deep Convolutional Neural Network

The following scripts were created for the Deep Learning 2018 course from BUTE by József Kenderák and Árom Görög.

## Prerequisites
Anaconda 5.3 contains a lot of libraries but we need the followings:
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
Copy the previously shared dataset to the root directory and just run:
```
python createDataset.py
```
In the *.py* file you can change the resize resolution of the images by *img_new_size = (384,128)*. The image resolution by default is 384x128, it is important due to input size of the neural network. After the preprocess is done, you can see the *dataset.hdf5* file in the root directory.

### 2. Step - Training
After the preprocessing is done just run:
```
python train.py
```
In the file you can change the number of epochs (*NB_EPOCHS*) and the batch size (*BATCH_SIZE*) which is 1 by default.

### 3. Step - Testing
After the training you can test your model via videos. Just download an *.mp4* video to the root directory and in the *test.py* you should add the filename in the *main()* function to the *vid_fn* variable. Make sure you add your *.hdf5* file path of your model correctly in the *.load_weights(...)* line.
```
python test.py
```
After testing is done, you can look at the predicted videos in the *results/* directory.

## TODO
 - Making the usage more comfortable by using argparse
 - Making better dir and file structure
 - Saving the history into a file after training
 - Testing with pictures, not only with videos
 - Implementing our custom pooling layer

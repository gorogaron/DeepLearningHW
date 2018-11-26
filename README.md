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
core/createDataset.py  - Put the whole dataset into the HDF5 file structuring by train, valid and test sets
core/imageGenerator.py - Train and Validation generators for *.fit_generator()
core/losses.py         - Defined some new Loss functions
core/networks.py       - Currently only contains U-net architecture
data/                  - The whole RAW dataset
model_weights/         - Wieghts of the best model
results/               - Save folder of the videos after testing
train.py               - Train the model after preprocessing
test.py                - After training we can test the network with videos
config.json            - Contain configuration of the model and training parameters
```

## HowTo
### 1. Step - Preprocessing
Copy the previously shared dataset to the root directory and just run:
```
data_processor(row,col) // now this function will do it all, you do not have to run createDataset.py directly
```
In the *.py* file you can change the resize resolution of the images by *img_new_size = (384,128)*. The image resolution by default is 384x128, it is important due to input size of the neural network. After the preprocess is done, you can see the *dataset.hdf5* file in the root directory.

### 2. Step - Training
After the preprocessing is done just run:
```
python train.py
```
In the file you can change the number of epochs (*nb_epochs*) and the batch size (*batch_size*) which is 1 by default.

### 3. Step - Testing
After the training you can test your model via videos. Just download an *.mp4* video to the root directory and in the *test.py* you should add the filename in the *main()* function to the *vid_fn* variable. Make sure you add your *.hdf5* file path of your model correctly in the *.load_weights(...)* line.
```
python test.py
```
After testing is done, you can look at the predicted videos in the *results/* directory.

## TODO
 - [x] Make the usage more comfortable
 - [x] Make better dir and file structure
 - [ ] Save the history into a file after training
 - [ ] Test with pictures, not only with videos
 - [ ] Implement new activation layer
 - [ ] Implement custom pooling layer

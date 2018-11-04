# -*- coding: utf-8 -*-
# This file contain the train and validation generator for the .fit_generator function.
# Currently only works with batch_size = 1
import numpy as np
import h5py
import cv2

# Reading dataset
hdf5_file = 'dataset.hdf5'
hf = h5py.File(hdf5_file, 'r')

def get_img_nparr(img1, img2, img3):
        '''
        Convert images from string to numpy array.

        # Arguments
        img1, img2, img3 : Strings from the HDF5 file

        # Output
        image1, image2, image3 : Numpy arrays of images
        '''

        image1 = cv2.imdecode(np.fromstring(img1, np.uint8), 1)
        image2 = cv2.imdecode(np.fromstring(img2, np.uint8), 1)
        image3 = cv2.imdecode(np.fromstring(img3, np.uint8), 1)

        return image1, image2, image3

def train_generator(batch_size = 1):
        '''
        Train generator. Assume that the yielded images format -> 
        (batch_size, channels, height, width)

        # Arguments
        batch_size: The size of one batch to yield from the generator

        # Output
        X_train, Y_train: Training batches by the defined format above
        '''
        
        input_data  = hf.get('x_train')
        output_data = hf.get('y_train')

        while True:

                X_train = []
                Y_train = []

                for i in range(input_data.shape[0]):

                        image1, image2, image3 = get_img_nparr(input_data[i][0],input_data[i][1], output_data[i])
                        X_train = np.transpose(np.concatenate((image1, image2), axis=2), (2,0,1))[np.newaxis, ...]
                        Y_train = np.transpose(image3, (2,0,1))[np.newaxis, ...]

                        yield np.array(X_train).astype('float32')/255. , np.array(Y_train).astype('float32')/255.

def valid_generator(batch_size = 1):
        '''
        Validation generator. Assume that the yielded images format -> 
        (batch_size, channels, height, width)

        # Arguments
        batch_size: The size of one batch to yield from the generator

        # Output
        X_valid, Y_valid: Training batches by the defined format above
        '''

        input_data  = hf.get('x_valid')
        output_data = hf.get('y_valid')

        while True:

                X_valid = []
                Y_valid = []

                for i in range(input_data.shape[0]):

                        image1, image2, image3 = get_img_nparr(input_data[i][0],input_data[i][1], output_data[i])
                        X_valid = np.transpose(np.concatenate((image1, image2), axis=2), (2,0,1))[np.newaxis, ...]
                        Y_valid = np.transpose(image3, (2,0,1))[np.newaxis, ...]

                        yield np.array(X_train).astype('float32')/255. , np.array(Y_train).astype('float32')/255.

        

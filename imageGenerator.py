import numpy as np
import h5py
import cv2

hdf5_file = 'dataset.hdf5'
hf = h5py.File(hdf5_file, 'r')

def get_img_nparr(img1, img2, img3):

        image1 = cv2.imdecode(np.fromstring(img1, np.uint8), 1)
        image2 = cv2.imdecode(np.fromstring(img2, np.uint8), 1)
        image3 = cv2.imdecode(np.fromstring(img3, np.uint8), 1)

        return image1, image2, image3

def train_generator(batch_size = 1):
        
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

        input_data  = hf.get('x_valid')
        output_data = hf.get('y_valid')

        while True:

                X_train = []
                Y_train = []

                for i in range(input_data.shape[0]):

                        image1, image2, image3 = get_img_nparr(input_data[i][0],input_data[i][1], output_data[i])
                        X_train = np.transpose(np.concatenate((image1, image2), axis=2), (2,0,1))[np.newaxis, ...]
                        Y_train = np.transpose(image3, (2,0,1))[np.newaxis, ...]

                        yield np.array(X_train).astype('float32')/255. , np.array(Y_train).astype('float32')/255.

        

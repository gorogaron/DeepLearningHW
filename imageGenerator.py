import numpy as np
import h5py
import cv2

hdf5_file = 'dataset.hdf5'

def get_img_nparr(img1, img2, img3):
        image1 = cv2.imdecode(np.fromstring(img1, np.uint8), 1)
        image2 = cv2.imdecode(np.fromstring(img2, np.uint8), 1)
        image3 = cv2.imdecode(np.fromstring(img3, np.uint8), 1)
        return image1, image2, image3

def train_generator(batch_size = 1):
        hf = h5py.File(hdf5_file, 'r')
        input_data  = hf.get('x_train')
        output_data = hf.get('y_train')
        
        while True:
                X_train = []
                Y_train = []
                for i in range(input_data.shape[0]):
                        image1, image2, image3 = get_img_nparr(input_data[i][0],input_data[i][1], output_data[i])
                        X_train = (image1, image2)
                        Y_train = image3
                        yield np.array(X_train), np.array(Y_train)

def valid_generator():
        hf = h5py.File(hdf5_file, 'r')
        input_data  = hf.get('x_valid')
        output_data = hf.get('y_valid')
        while True:
                X_train = 0
                Y_train = 0
                for i in range(input_data.shape[0]):
                        image1, image2, image3 = get_img_nparr(input_data[i][0],input_data[i][1], output_data[i])
                        X_train = (image1, image2)
                        Y_train = image3
                        yield np.array(X_train), np.array(Y_train)

if __name__ == '__main__':
        hf = h5py.File(hdf5_file,'r')

        ## Test database
        x_test = hf.get('x_test')
        y_test = hf.get('y_test')
        
        ## Data visualization via VALIDATION generator!
        G = valid_generator()
        for _ in G:
                x, y = next(G)
                cv2.imshow("input1", x[0])
                cv2.imshow("input2", x[1])
                cv2.imshow("output", y)
                cv2.waitKey(0)
        

from os import listdir
from os import walk
from os.path import isfile, join, exists
import matplotlib.pylab as plt 
import h5py
import numpy as np
import cv2

inputData = []
outputData = []
triplet = []


for folderName in sorted(listdir('./data')):
    print(folderName)
    count = 0
    inputData = []
    outputData = []

    for fileName in sorted(listdir('./data/' + folderName)):

        img = cv2.imread('./data/' + folderName + '/' + fileName)
        img = cv2.resize(img, (640, 480))
        triplet.append(img)
        count += 1
        if (count == 3):
            inputDataFrame = np.empty([2,480,640,3])
            inputDataFrame[0] = triplet[0]
            inputDataFrame[1] = triplet[2]

            inputData.append(inputDataFrame)
            outputData.append(triplet[1])
            
            count = 0
            triplet = []

    if exists('./data.h5'):
        with h5py.File('data.h5','a') as hdf:
            hdf['input'].resize(hdf['input'].shape[0] + np.asarray(inputData).shape[0], axis=0)
            hdf['input'][hdf['input'].shape[0]- np.asarray(inputData).shape[0]:] = inputData

            hdf['output'].resize(hdf['output'].shape[0] + np.asarray(inputData).shape[0], axis=0)
            hdf['output'][hdf['output'].shape[0]- np.asarray(inputData).shape[0]:] = outputData

            print('Input dataset shape:  ' + str(hdf['input']))
            print('Input dataset shape:  ' + str(hdf['output']))
    else:
        with h5py.File('data.h5','w') as hdf:
            dset_in = hdf.create_dataset('input', data=inputData, maxshape=(None, 2, 480, 854, 3))
            dset_out = hdf.create_dataset('output', data = outputData, maxshape=(None, 480, 854, 3))
            




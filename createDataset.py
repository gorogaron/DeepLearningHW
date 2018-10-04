from os import remove, listdir
from os.path import exists
import numpy as np
import h5py
import cv2

file_paths = dict()
num_of_samp = 0
N = 2

# Create dict of filepaths and calculate number of samples
for folder in sorted(listdir('./data')):
    file_paths[folder] = []
    for file_name in sorted(listdir('./data/' + folder)):
        file_paths[folder].append(file_name)
    num_of_samp += len(file_paths[folder]) - N

# Create HDF5 file for dataset
if exists('dataset.hdf5'):
    print('Dataset deleted.')
    remove('dataset.hdf5')

f = h5py.File('dataset.hdf5')
dt = h5py.special_dtype(vlen=np.dtype('uint8'))
input_dataset = f.create_dataset('input', (num_of_samp,2,), dtype=dt)
output_dataset = f.create_dataset('output', (num_of_samp,), dtype=dt)

# Read images
samp_idx = 0
for key in file_paths:
    print('Current folder: ' + key + '...')
    for i in range(len(file_paths[key]) - N):
        input_1_path = 'data/' + key + '/' + file_paths[key][i]
        input_2_path = 'data/' + key + '/' + file_paths[key][i+2]
        output_path  = 'data/' + key + '/' + file_paths[key][i+1]
        fin1 = open(input_1_path, 'rb')
        fin2 = open(input_2_path, 'rb')
        fout = open(output_path, 'rb')
        fin1_b64 = np.fromstring(fin1.read(), np.uint8)
        fin2_b64 = np.fromstring(fin2.read(), np.uint8)
        fout_b64 = np.fromstring(fout.read(), np.uint8)
        fin_b64  = (fin1_b64 , fin2_b64)
        input_dataset[samp_idx] = fin_b64
        output_dataset[samp_idx]= fout_b64
        
        # img = input_dataset[samp_idx][0]
        # nparr = np.fromstring(img, np.uint8)
        # img_np = cv2.imdecode(nparr, 1)
        # cv2.imshow('a', img_np)
        # cv2.waitKey(0)
        samp_idx += 1


# hf = h5py.File('foo.hdf5', 'r')
# data = hf.get('binary_data').value # `data` is now an ndarray.

from os.path import dirname, abspath
import os
import numpy as np
from predict import predict
from train_wrapper import train_wrapper
import datetime


mode = 'train'

### common parameters ###
homedir = dirname(dirname(abspath(__file__)))
data_dir = homedir + '/data/'
test_data_dir = data_dir + 'interpretation_points/riped/test/'
test_files = [test_data_dir + x for x in os.listdir(test_data_dir) if
              os.path.isfile(test_data_dir + x) and 'z_' in x]
segy_filename = [data_dir + '3d_cube_cropped_flattened.segy']
now = datetime.datetime.now()
output_dir = homedir + f'/output/unet_output/{mode}_' + now.strftime('%Y-%m-%d_%H-%M') + '/'
inp_res = np.float16
facies_list = ['ch', 'fault']
test_slices = [1016, 1036] # range

### predict mode parameters ###
model_path = dirname(dirname(output_dir)) + '/train_2020-11-11_20-11/trained.h5'

### train mode parameters ###
train_data_dir = data_dir + 'interpretation_points/riped/'
train_files = [train_data_dir + x for x in os.listdir(train_data_dir) if
               os.path.isfile(train_data_dir + x) and 'z_' in x]
n_samples = 10
create_patches = True
batch_size = 8
lr_prior = [3e-4, 1e-4, 1e-5]
layer_depth_prior = [3, 5]
filters_root_prior = [16, 32, 64]
window_size_prior = [64, 128, 256]
overlap_prior = [0, 50, 70]
epochs_prior = [150, 250]

if mode == 'train':
    train_wrapper(n_samples, lr_prior, layer_depth_prior, filters_root_prior, window_size_prior, overlap_prior,
                  epochs_prior, output_dir, segy_filename, inp_res, train_files, test_files, facies_list, test_slices,
                  create_patches, batch_size, mode)
elif mode == 'predict':
    predict(segy_filename, inp_res, test_files, facies_list, mode, model_path, output_dir, test_slices)
else:
    raise ValueError("mode should be train or predict")
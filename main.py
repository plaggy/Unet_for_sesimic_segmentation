from os.path import dirname, abspath
import os
import numpy as np
from predict import predict
from train_wrapper import train_wrapper
import datetime

# which mode to run, ‘train’ or ‘predict’
mode = 'train'

### common parameters ###
# root directory that contains directory with the code
homedir = dirname(dirname(abspath(__file__)))
# data directory inside the root directory
data_dir = homedir + '/data/'
# directory with test data points
test_data_dir = data_dir + 'interpretation_points/riped/test/'
# list of test files
test_files = [test_data_dir + x for x in os.listdir(test_data_dir) if
              os.path.isfile(test_data_dir + x) and 'z_' in x]
# path to segy
segy_filename = [data_dir + '3d_cube_cropped_flattened.segy']
now = datetime.datetime.now()
# directory to save all output files to
output_dir = homedir + f'/output/unet_output/{mode}_' + now.strftime('%Y-%m-%d_%H-%M') + '/'
# numeric precision with which to load and read segy
inp_res = np.float16
# a list of facies names, used to process train/test data files, should correspond to the facies names
# used in the train/test file names
facies_list = ['ch', 'fault']
# range of test slices
test_slices = [1016, 1036]

### predict mode parameters ###
# path to a saved model
model_path = dirname(dirname(output_dir)) + '/train_2020-11-11_20-11/trained.h5'

### train mode parameters ###
# directory with training data points
train_data_dir = data_dir + 'interpretation_points/riped/'
# list of training files
train_files = [train_data_dir + x for x in os.listdir(train_data_dir) if
               os.path.isfile(train_data_dir + x) and 'z_' in x]
# number of samples to draw from lists of hyperparameters.
# For each set of parameters a separate model will be created and trained
n_samples = 10
# whether or not to create patches from input examples. If False, examples are used as they are
create_patches = True
batch_size = 8
# a hyperparameter, a list of learning rates
lr_prior = [3e-4, 1e-4, 1e-5]
# a hyperparameter, a list of depths of the architecture (number of blocks in top-down and bottom-up paths)
layer_depth_prior = [3, 5]
# a hyperparameter, a list of numbers of filters of the first conv. block
filters_root_prior = [16, 32, 64]
# a hyperparameter, a list of sizes of patches to be extracted
window_size_prior = [64, 128, 256]
# a hyperparameter, a list of amounts of overlap between patches in %
overlap_prior = [0, 50, 70]
# a hyperparameter, a list of numbers of epochs
epochs_prior = [150, 250]

if mode == 'train':
    train_wrapper(n_samples, lr_prior, layer_depth_prior, filters_root_prior, window_size_prior, overlap_prior,
                  epochs_prior, output_dir, segy_filename, inp_res, train_files, test_files, facies_list, test_slices,
                  create_patches, batch_size, mode)
elif mode == 'predict':
    predict(segy_filename, inp_res, test_files, facies_list, mode, model_path, output_dir, test_slices)
else:
    raise ValueError("mode should be train or predict")
from typing import Tuple, List

import numpy as np
import tensorflow as tf
import segyio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from skimage.util import random_noise


def segy_read(segy_file, mode, scale=1, inp_cube=None, read_direc='xline', inp_res=np.float32):

    if mode == 'create':
        print('Starting SEG-Y decompressor')
        output = segyio.spec()

    elif mode == 'add':
        if inp_cube is None:
            raise ValueError('if mode is add inp_cube must be provided')
        print('Starting SEG-Y adder')
        cube_shape = inp_cube.shape
        data = np.empty(cube_shape[0:-1])

    else:
        raise ValueError('mode must be create or add')

    # open the segyfile and start decomposing it
    with segyio.open(segy_file, "r") as segyfile:
        segyfile.mmap()

        if mode == 'create':
            # Store some initial object attributes
            output.inl_start = segyfile.ilines[0]
            output.inl_end = segyfile.ilines[-1]
            output.inl_step = segyfile.ilines[1] - segyfile.ilines[0]

            output.xl_start = segyfile.xlines[0]
            output.xl_end = segyfile.xlines[-1]
            output.xl_step = segyfile.xlines[1] - segyfile.xlines[0]

            output.t_start = int(segyfile.samples[0])
            output.t_end = int(segyfile.samples[-1])
            output.t_step = int(segyfile.samples[1] - segyfile.samples[0])


            # Pre-allocate a numpy array that holds the SEGY-cube
            data = np.empty((segyfile.xline.length,segyfile.iline.length,\
                            (output.t_end - output.t_start)//output.t_step+1), dtype = np.float32)

        # Read the entire cube line by line in the desired direction
        if read_direc == 'inline':
            # Potentially time this to find the "fast" direction
            #start = time.time()
            for il_index in range(segyfile.xline.len):
                data[il_index,:,:] = segyfile.iline[segyfile.ilines[il_index]]
            #print(end - start)

        elif read_direc == 'xline':
            #start = time.time()
            for xl_index in range(segyfile.iline.len):
                data[:,xl_index,:] = segyfile.xline[segyfile.xlines[xl_index]]
            #end = time.time()
            #print(end - start)

        elif read_direc == 'full':
            #start = time.time()
            data = segyio.tools.cube(segy_file)
            #end = time.time()
            #print(end - start)
        else:
            print('Define reading direction(read_direc) using either ''inline'', ''xline'', or ''full''')

        factor = scale/np.amax(np.absolute(data))
        if inp_res == np.float32:
            data = (data*factor)
        else:
            data = (data*factor).astype(dtype = inp_res)

    if mode == 'create':
        output.data = data
        return output
    else:
        return output


def convert(file_list, facies_names):
    file_list_by_facie = []
    for facie in facies_names:
        facie_list = []
        for filename in file_list:
            if facie in os.path.basename(filename):
                facie_list.append(filename)
        file_list_by_facie.append(facie_list)

    z_slices = {}
    # Itterate through the list of example adresses and store the class as an integer
    for i, files in enumerate(file_list_by_facie):
        for filename in files:
            a = np.loadtxt(filename, skiprows=0, usecols=range(3), dtype=np.int32)
            a = np.append(a, i * np.ones((len(a), 1), dtype=np.int32), axis=1)
            if a[0][2] not in z_slices:
                z_slices[a[0][2]] = a
            else:
                z_slices[a[0][2]] = np.append(z_slices[a[0][2]], a, axis=0)

    # Return the list of adresses and classes as a numpy array
    return z_slices


def patches_creator(seismic_data, labels, window_size, window_step):
    seis_patches = []
    label_patches = []

    for line, label in zip(seismic_data, labels):
        window_start_x = 0
        window_end_x = window_size
        window_start_y = 0
        window_end_y = window_size
        n_steps_x = (line.shape[0] - window_size) // window_step + 1
        n_steps_y = (line.shape[1] - window_size) // window_step + 1

        for i in range(n_steps_x * n_steps_y):
            seis_ex = line[window_start_x:window_end_x, window_start_y:window_end_y, :]
            seis_patches.append(seis_ex)
            classes_ex = label[window_start_x:window_end_x, window_start_y:window_end_y]
            label_patches.append(classes_ex)

            seis_patches.append(np.flipud(seis_ex))
            label_patches.append(np.flipud(classes_ex))

            seis_patches.append(random_noise(seis_ex, mode='gaussian', var=0.001))
            label_patches.append(classes_ex)

            if window_end_x + window_step > line.shape[0]:
                window_start_x = 0
                window_end_x = window_size
                window_start_y += window_step
                window_end_y += window_step
            else:
                window_start_x += window_step
                window_end_x += window_step

    return np.array(seis_patches), np.array(label_patches)


def point_to_images(z_points, segy_obj):
    seis_data = []
    labels = []
    for z_sl, points in z_points.items():
        label = np.zeros((segy_obj.data.shape[0], segy_obj.data.shape[1]))

        idx_0 = np.where(points[:, 3] == 0)[0]
        idx_1 = np.where(points[:, 3] == 1)[0]
        label[(points[idx_0, 0] - segy_obj.inl_start) // segy_obj.inl_step, (points[idx_0, 1] - segy_obj.xl_start) // segy_obj.xl_step] = 1
        label[(points[idx_1, 0] - segy_obj.inl_start) // segy_obj.inl_step, (points[idx_1, 1] - segy_obj.xl_start) // segy_obj.xl_step] = 2
        labels.append(label[:, :, np.newaxis])

        seis_data.append(segy_obj.data[:, :, z_sl - segy_obj.t_start, :])

    return np.array(seis_data), np.array(labels)


def pad_data(layer_depth, data):
    if np.ndim(np.array(data) < 2):
        x_shape = 0
        y_shape = 0
        for ex in data:
            if ex.shape[0] > x_shape:
                x_shape = ex.shape[0]
            if ex.shape[1] > y_shape:
                y_shape = ex.shape[1]
    else:
        x_shape = data[0].shape[0]
        y_shape = data[0].shape[1]

    for i in range(layer_depth):
        x_shape = np.ceil(x_shape / 2)
        y_shape = np.ceil(y_shape / 2)

    for i in range(layer_depth):
        x_shape = int(x_shape * 2)
        y_shape = int(y_shape * 2)

    if np.ndim(np.array(data) < 2):
        data_padded = np.zeros((len(data), x_shape, y_shape, data[0].shape[2]))
        for i, d in enumerate(data):
            data_padded[i] = np.pad(d, ((0, x_shape - d.shape[0]), (0, y_shape - d.shape[1]),
                                                (0, 0)), mode='constant', constant_values=0)
    else:
        data_padded = np.pad(data, ((0, 0), (0, x_shape - data.shape[1]), (0, y_shape - data.shape[2]),
                                          (0, 0)), mode='constant', constant_values=0)

    return data_padded


def load_data(segy_filename, inp_res, test_files, facies_names, mode, layer_depth, test_slices, train_files=None,
              create_patches=None, window_size=None, window_overlap=None):
    if type(segy_filename) is str or (type(segy_filename) is list and len(segy_filename) == 1):
        # Check if the filename needs to be retrieved from a list
        if type(segy_filename) is list:
            segy_filename = segy_filename[0]

        # Make a master segy object
        segy_obj = segy_read(segy_filename, mode='create', read_direc='full', inp_res=inp_res)

        # Define how many segy-cubes we're dealing with
        segy_obj.cube_num = 1
        segy_obj.data = np.expand_dims(segy_obj.data, axis=len(segy_obj.data.shape))

    elif type(segy_filename) is list:
        # start an iterator
        i = 0

        # iterate through the list of cube names and store them in a masterobject
        for filename in segy_filename:
            # Make a master segy object
            if i == 0:
                segy_obj = segy_read(filename, mode='create', read_direc='full', inp_res=inp_res)

                # Define how many segy-cubes we're dealing with
                segy_obj.cube_num = len(segy_filename)

                # Reshape and preallocate the numpy-array for the rest of the cubes
                print('Starting restructuring to 4D arrays')
                ovr_data = np.empty((list(segy_obj.data.shape) + [len(segy_filename)]))
                ovr_data[:, :, :, i] = segy_obj.data
                segy_obj.data = ovr_data
                ovr_data = None
                print('Finished restructuring to 4D arrays')
            else:
                # Add another cube to the numpy-array
                segy_obj.data[:, :, :, i] = segy_read(segy_filename, mode='add', inp_cube=segy_obj.data,
                                                      read_direc='full', inp_res=inp_res)
            # Increase the itterator
            i += 1
    else:
        print('The input filename needs to be a string, or a list of strings')

    test_z_points = convert(test_files, facies_names)
    test_seis_data, test_labels = point_to_images(test_z_points, segy_obj)

    if type(test_slices) is list:
        test_seis_data = np.transpose(segy_obj.data[:, :, test_slices[0] - segy_obj.t_start:test_slices[1] -
                                                                           segy_obj.t_start, :], (2, 0, 1, 3))

    test_seis_data_padded = pad_data(layer_depth, test_seis_data)
    test_labels_padded = pad_data(layer_depth, test_labels)

    if mode == 'predict':
        return test_seis_data_padded, test_labels_padded, segy_obj

    z_points = convert(train_files, facies_names)
    seis_data, labels = point_to_images(z_points, segy_obj)

    if create_patches:
        window_step = int(np.ceil(window_size * (1 - window_overlap / 100)))
        seis_data, labels = patches_creator(seis_data, labels, window_size, window_step)

        seis_data_padded = pad_data(layer_depth, seis_data)
        labels_padded = pad_data(layer_depth, labels)
    else:
        seis_data_padded = pad_data(layer_depth, seis_data)
        labels_padded = pad_data(layer_depth, labels)

    x_train, x_valid, y_train, y_valid = train_test_split(seis_data_padded, labels_padded, test_size=0.1, random_state=42)
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_data = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))

    return train_data, validation_data, test_seis_data_padded, test_labels_padded, segy_obj
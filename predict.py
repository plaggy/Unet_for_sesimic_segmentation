import tensorflow as tf
from unet import ConvBlock, UpconvBlock
import numpy as np
import time
import data_loader as data
import os
from utils import printout, save_prediction


def predict(segy_filename, inp_res, test_files, facies_list, mode, model_path, output_dir, test_slices):
    model = tf.keras.models.load_model(model_path, custom_objects={'ConvBlock': ConvBlock, 'UpconvBlock': UpconvBlock})

    layer_depth = 0
    for l in model.layers:
        if type(l) is ConvBlock:
            layer_depth += 1
        elif type(l) is UpconvBlock:
            layer_depth -= 1

    test_seis_data, test_labels, segy_obj = \
        data.load_data(segy_filename, inp_res, test_files, facies_list, mode, layer_depth, test_slices)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    acc_file = open(output_dir + 'acc.txt', 'w+')
    runtime_file = open(output_dir + 'runtimes.txt', 'w+')

    start = time.time()
    prediction = model.predict(test_seis_data)
    predict_time = time.time() - start

    runtime_file.write(f"predict_time: {predict_time}\n")

    save_prediction(test_slices, prediction, segy_filename, segy_obj, output_dir)

    if len(prediction) == len(test_labels):
        acc = np.sum(prediction.flatten() == test_labels.flatten()) / test_labels.flatten().shape[0]
        acc_file.write(f"{acc}\n")

    printout(None, output_dir, test_labels.flatten(), prediction.flatten(),
             ['background'] + facies_list)


    runtime_file.close()
    acc_file.close()
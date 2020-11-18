# unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with unet.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import time
import tensorflow as tf

import unet
from trainer import Trainer
from schedulers import SchedulerType
import data_loader as data
import os
import matplotlib.pyplot as plt
from utils import printout, save_prediction


def train(lr, layer_depth, filters_root, epochs, train_data, validation_data, n_classes, batch_size):
    unet_model = unet.build_model(channels=1,
                                  num_classes=n_classes + 1,
                                  layer_depth=layer_depth,
                                  filters_root=filters_root)

    unet.finalize_model(unet_model,
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                        mean_iou=False,
                        dice_coefficient=False,
                        auc=False,
                        learning_rate=lr)

    trainer = Trainer(name="riped",
                      learning_rate_scheduler=SchedulerType.WARMUP_LINEAR_DECAY,
                      warmup_proportion=0.1,
                      learning_rate=lr)

    history, train_time = trainer.fit(unet_model,
                                      train_data,
                                      validation_data,
                                      epochs=epochs,
                                      batch_size=batch_size)

    return unet_model, history, train_time


def train_wrapper(n_samples, lr_prior, layer_depth_prior, filters_root_prior, window_size_prior, overlap_prior,
                  epochs_prior, output_dir_global, segy_filename, inp_res, train_files, test_files, facies_list, test_slices,
                  create_patches, batch_size, mode):

    # Monte Carlo sampling from priors
    lr_samples = np.array(lr_prior)[np.random.randint(0, len(lr_prior), size=n_samples)]
    layer_depth_samples = np.array(layer_depth_prior)[np.random.randint(0, len(layer_depth_prior), size=n_samples)]
    filters_root_samples = np.array(filters_root_prior)[np.random.randint(0, len(filters_root_prior), size=n_samples)]
    window_size_samples = np.array(window_size_prior)[np.random.randint(0, len(window_size_prior), size=n_samples)]
    overlap_samples = np.array(overlap_prior)[np.random.randint(0, len(overlap_prior), size=n_samples)]
    epochs_samples = np.array(epochs_prior)[np.random.randint(0, len(epochs_prior), size=n_samples)]

    if not os.path.exists(output_dir_global):
        os.makedirs(output_dir_global)

    acc_file = open(output_dir_global + 'acc.txt', 'w+')
    runtime_file = open(output_dir_global + 'runtimes.txt', 'w+')
    runtime_file.write(f" \ttrain_time\tprediction_time\n")

    for i in range(n_samples):

        if layer_depth_samples[i] > 3 and window_size_samples[i] < 64:
            continue

        train_data, validation_data, test_seis_data, test_labels, segy_obj = \
            data.load_data(segy_filename, inp_res, test_files, facies_list, mode, layer_depth_samples[i], test_slices,
                           train_files, create_patches, window_size_samples[i], overlap_samples[i])

        model, history, train_time = train(lr_samples[i], layer_depth_samples[i], filters_root_samples[i],
                                           epochs_samples[i], train_data, validation_data, len(facies_list), batch_size)

        output_dir = output_dir_global + f"set_{i}/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model.save(output_dir + 'trained.h5')

        with open(output_dir + 'params_file.txt', 'w') as pfile:
            pfile.write(f'epochs: {epochs_samples[i]}\n'
                        f'lr: {lr_samples[i]}\n'
                        f'depth: {layer_depth_samples[i]}\n'
                        f'ilters_root: {filters_root_samples[i]}\n'
                        f'window_size: {window_size_samples[i]}\n'
                        f'overlap: {overlap_samples[i]}')

        start = time.time()
        prediction = model.predict(test_seis_data)
        predict_time = time.time() - start

        runtime_file.write(f"{i}\t{train_time}\t{predict_time}\n")

        save_prediction(test_slices, prediction, segy_filename, segy_obj, output_dir)

        if len(prediction) == len(test_labels):
            acc = np.sum(prediction.flatten() == test_labels.flatten()) / test_labels.flatten().shape[0]
            acc_file.write(f"{acc}\n")

        printout(history, output_dir, test_labels.flatten(), prediction.flatten(),
                 ['background'] + facies_list)

    runtime_file.close()
    acc_file.close()

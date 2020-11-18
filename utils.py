from typing import Tuple

import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import os
import segyio
from shutil import copyfile


def crop_to_shape(data, shape: Tuple[int, int, int]):
    """
    Crops the array to the given image shape by removing the border

    :param data: the array to crop, expects a tensor of shape [batches, nx, ny, channels]
    :param shape: the target shape [batches, nx, ny, channels]
    """
    diff_nx = (data.shape[0] - shape[0])
    diff_ny = (data.shape[1] - shape[1])

    if diff_nx == 0 and diff_ny == 0:
        return data

    offset_nx_left = diff_nx // 2
    offset_nx_right = diff_nx - offset_nx_left
    offset_ny_left = diff_ny // 2
    offset_ny_right = diff_ny - offset_ny_left

    cropped = data[offset_nx_left:(-offset_nx_right), offset_ny_left:(-offset_ny_right)]

    assert cropped.shape[0] == shape[0]
    assert cropped.shape[1] == shape[1]
    return cropped


def crop_labels_to_shape(shape: Tuple[int, int, int]):
    def crop(image, label):
        return image, crop_to_shape(label, shape)
    return crop


def crop_image_and_label_to_shape(shape: Tuple[int, int, int]):
    def crop(image, label):
        return crop_to_shape(image, shape), \
               crop_to_shape(label, shape)
    return crop


def to_rgb(img: np.array):
    """
    Converts the given array into a RGB image and normalizes the values to [0, 1).
    If the number of channels is less than 3, the array is tiled such that it has 3 channels.
    If the number of channels is greater than 3, only the first 3 channels are used

    :param img: the array to convert [bs, nx, ny, channels]

    :returns img: the rgb image [bs, nx, ny, 3]
    """
    img = img.astype(np.float32)
    img = np.atleast_3d(img)

    channels = img.shape[-1]
    if channels == 1:
        img = np.tile(img, 3)

    elif channels == 2:
        img = np.concatenate((img, img[..., :1]), axis=-1)

    elif channels > 3:
        img = img[..., :3]

    img[np.isnan(img)] = 0
    img -= np.amin(img)
    if np.amax(img) != 0:
        img /= np.amax(img)

    return img


def printout(history, write_location, y_test, pred_test, facies_list):

    if not os.path.exists(write_location):
        os.makedirs(write_location)

    if history is not None:
        plt.figure()
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('batch #')
        ax1.set_ylabel('loss', color='blue')
        ax1.plot(np.array(history.batch_loss), color='blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('acc', color='red')
        ax2.plot(np.array(history.batch_acc), color='red')
        plt.savefig(write_location + 'batch_history.jpg')

        np.savetxt(write_location + 'history_batch_loss.txt', np.array(history.batch_loss))
        np.savetxt(write_location + 'history_batch_acc.txt', np.array(history.batch_acc))
        if history.val_acc[0] is not None:
            np.savetxt(write_location + 'history_val_acc.txt', np.array(history.val_acc))
            np.savetxt(write_location + 'history_val_loss.txt', np.array(history.val_loss))

    if len(y_test) == len(pred_test):
        plt.figure()
        cm = confusion_matrix(y_test, pred_test, normalize='true')
        df_cm = pd.DataFrame(cm, index=[i for i in facies_list],
                             columns=[i for i in facies_list])
        sn.heatmap(df_cm, annot=True, fmt='.2%', cmap='Blues')
        plt.savefig(write_location + "confusion_matrix.jpg")


def save_segy_prediction(predictions, test_slices, segy_filename, segy_obj, write_location, c=None):
    if c is not None:
        output_file = write_location + f'prediction_prob_{c}.sgy'
    else:
        output_file = write_location + 'test_prediction.sgy'
    il_shape = segy_obj.data.shape[0]
    xl_shape = segy_obj.data.shape[1]
    predictions = np.transpose(predictions, (1, 2, 0))
    predictions = predictions[:il_shape, :xl_shape, :]
    print(f"predictions shape: {predictions.shape}")
    # a faster way of copying all the trace headers from source
    copyfile(segy_filename[0], output_file)

    with segyio.open(output_file, "r+") as src:
        for il in src.ilines:
            il_num = il - segy_obj.inl_start
            src.iline[il] = -1 * (np.ones((src.iline[il].shape), dtype=np.float32))
            line = src.iline[il]
            line[:, test_slices[0] - segy_obj.t_start:test_slices[1] - segy_obj.t_start] = predictions[il_num]
            src.iline[il] = line


def save_prediction(test_slices, prediction, segy_filename, segy_obj, output_dir):
    if type(test_slices) is list:
        for c in range(prediction.shape[-1]):
            save_segy_prediction(prediction[..., c], test_slices, segy_filename, segy_obj, output_dir, c)
    else:
        for c in range(prediction.shape[-1]):
            plt.imsave(output_dir + f'slice_{test_slices}_prob_{c}.jpg', prediction[0, ..., c])

    prediction = np.argmax(prediction, axis=-1)

    if type(test_slices) is list:
        save_segy_prediction(prediction, test_slices, segy_filename, segy_obj, output_dir)
    else:
        plt.imsave(output_dir + f'pred_{test_slices}.jpg', prediction[0])
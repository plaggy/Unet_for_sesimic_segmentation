from datetime import datetime
from pathlib import Path
from typing import Union, List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import time
import utils
import schedulers
from schedulers import SchedulerType


class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.batch_loss = []
        self.batch_acc = []
        self.val_acc = []
        self.val_loss = []

    def on_batch_end(self, batch, logs={}):
        self.batch_loss.append(logs.get('loss'))
        self.batch_acc.append(logs.get('accuracy'))

    def on_epoch_end(self, batch, logs={}):
        self.val_acc.append(logs.get('val_accuracy'))
        self.val_loss.append(logs.get('val_loss'))


class Trainer:
    """
    Fits a given model to a datasets and configres learning rate schedulers and
    various callbacks

    :param name: Name of the model, used to build the target log directory if no explicit path is given
    :param log_dir_path: Path to the directory where the model and tensorboard summaries should be stored
    :param checkpoint_callback: Flag if checkpointing should be enabled. Alternatively a callback instance can be passed
    :param tensorboard_callback: Flag if information should be stored for tensorboard. Alternatively a callback instance can be passed
    :param tensorboard_images_callback: Flag if intermediate predictions should be stored in Tensorboard. Alternatively a callback instance can be passed
    :param callbacks: List of additional callbacks
    :param learning_rate_scheduler: The learning rate to be used. Either None for a constant learning rate, a `Callback` or a `SchedulerType`
    :param scheduler_opts: Further kwargs passed to the learning rate scheduler
    """

    def __init__(self,
                 name: Optional[str]="unet",
                 log_dir_path: Optional[Union[Path, str]]=None,
                 checkpoint_callback: Optional[Union[TensorBoard, bool]] = True,
                 tensorboard_callback: Optional[Union[TensorBoard, bool]] = True,
                 callbacks: Union[List[Callback], None]=None,
                 learning_rate_scheduler: Optional[Union[SchedulerType, Callback]]=None,
                 **scheduler_opts,
                 ):
        self.checkpoint_callback = checkpoint_callback
        self.tensorboard_callback = tensorboard_callback
        self.callbacks = callbacks
        self.learning_rate_scheduler = learning_rate_scheduler
        self.scheduler_opts=scheduler_opts

        if isinstance(log_dir_path, Path):
            log_dir_path = str(log_dir_path)

        self.log_dir_path = log_dir_path

    def fit(self,
            model: Model,
            train_dataset,
            validation_dataset=None,
            epochs=10,
            batch_size=1,
            **fit_kwargs):
        """
        Fits the model to the given data

        :param model: The model to be fit
        :param train_dataset: The dataset used for training
        :param validation_dataset: (Optional) The dataset used for validation
        :param test_dataset:  (Optional) The dataset used for test
        :param epochs: Number of epochs
        :param batch_size: Size of minibatches
        :param fit_kwargs: Further kwargs passd to `model.fit`
        """

        prediction_shape = self._get_output_shape(model, train_dataset)[1:]

        train_dataset = train_dataset.map(utils.crop_labels_to_shape(prediction_shape)).batch(batch_size)

        if validation_dataset:
            validation_dataset = validation_dataset.map(utils.crop_labels_to_shape(prediction_shape)).batch(batch_size)

        history = LossHistory()
        start = time.time()
        model.fit(train_dataset,
                  validation_data=validation_dataset,
                epochs=epochs,
                callbacks=[history],
                **fit_kwargs)

        train_time = time.time() - start

        return history, train_time

    def _get_output_shape(self,
                          model: Model,
                          train_dataset: tf.data.Dataset):
        return model.predict(train_dataset
                             .take(count=1)
                             .batch(batch_size=1)
                             ).shape

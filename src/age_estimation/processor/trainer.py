import os
import numpy as np
import pandas as pd
from imgaug import augmenters as iaa
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.utils import Sequence
from scipy.misc import imresize
from utils import read_by_pyvips


class DataGenerator(Sequence):

    def __init__(self, dataset, split_name, input_shape, batch_size, augmentations=None, shuffle=True):
        self.dataset = dataset
        self.image_names = dataset.splits[split_name]
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(len(self.image_names) / float(self.batch_size)))

    def __getitem__(self, index):
        batch_image_names = self.image_names[index * self.batch_size: (index + 1) * self.batch_size]
        batch_x = list()
        batch_y = list()

        for i, image_name in enumerate(batch_image_names):
            image = read_by_pyvips(self.dataset.get_absolute_path(image_name))
            batch_x.append(imresize(image, (self.input_shape[0], self.input_shape[1])))
            batch_y.append(self.dataset.labels_dict[image_name])

        batch_x = np.asarray(batch_x)
        if self.augmentations:
            batch_x = self.augmentations.augment_images(batch_x)

        return batch_x / 255., batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_names)


class Trainer(object):

    def __init__(self, model_wrapper, dataset, test_mode=True, epochs=10, batch_size=32, loss='mae', metrics=['mse'],
                 optimizer_type='adam', learning_rate=0.1, decay=6e-5, augmentations=None,
                 callbacks_list=['best_model_checkpoint', 'last_model_checkpoint', 'early_stopping', 'tensorboard', 'csv_logger', 'learning_rate_scheduler'],
                 monitor='val_loss', monitor_mode='min', early_stopping_patience=2, alias='',
                 nn_models_dir='../../nn_models', logs_dir='../../logs'):

        self.model_wrapper = model_wrapper
        self.dateset = dataset
        self.test_mode = test_mode

        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.metrics = metrics
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.decay = decay
        self.augmentations = augmentations

        self.callbacks_list = callbacks_list
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.early_stopping_patience = early_stopping_patience

        self.alias = alias
        self.nn_models_dir = nn_models_dir
        self.logs_dir = logs_dir

        self._initialize()

    def _initialize(self):
        if self.augmentations:
            self._augmentations = iaa.Sequential([
                iaa.Sometimes(0.5, iaa.Affine(scale=(0.8, 1.2), backend='cv2')),
                iaa.Sometimes(0.5, iaa.Fliplr(0.5))
            ])
        else:
            self._augmentations = None

        self._train_generator = DataGenerator(
            self.dateset,
            split_name='train_image_names',
            input_shape=self.model_wrapper.input_shape,
            batch_size=self.batch_size,
            augmentations=self._augmentations
        )

        self._valid_generator = DataGenerator(
            self.dateset,
            split_name='valid_image_names',
            input_shape=self.model_wrapper.input_shape,
            batch_size=self.batch_size,
        )

        if self.test_mode:
            self._test_generator = DataGenerator(
                self.dateset,
                split_name='test_image_names',
                input_shape=self.model_wrapper.input_shape,
                batch_size=self.batch_size,
            )

        self._optimizer = self._make_optimizer()
        self._callbacks = self._make_callbacks(model_maskname='{0}{1}'.format(self.alias, self.model_wrapper.name))

    def _make_optimizer(self):

        if self.optimizer_type == 'rmsprop':
            return RMSprop(lr=self.learning_rate, decay=float(self.decay))

        elif self.optimizer_type == 'adam':
            return Adam(lr=self.learning_rate, decay=float(self.decay))

        elif self.optimizer_type == 'amsgrad':
            return Adam(lr=self.learning_rate, decay=float(self.decay), amsgrad=True)

        elif self.optimizer_type == 'nesterov':
            return SGD(lr=self.learning_rate, decay=float(self.decay), momentum=0.9, nesterov=True)

        else:
            raise NotImplementedError('Unknown optimizer "{0}"'.format(self.optimizer_type))

    def _make_callbacks(self, model_maskname):
        callbacks = list()

        if 'best_model_checkpoint' in self.callbacks_list:
            best_model = os.path.join(self.nn_models_dir, 'best_{0}.h5').format(model_maskname)
            best_model_checkpoint = ModelCheckpoint(
                filepath=best_model,
                monitor=self.monitor,
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                mode=self.monitor_mode,
                period=1
            )
            callbacks.append(best_model_checkpoint)

        if 'last_model_checkpoint' in self.callbacks_list:
            last_model = os.path.join(self.nn_models_dir, 'last_{0}.h5').format(model_maskname)
            last_model_checkpoint = ModelCheckpoint(
                filepath=last_model,
                monitor=self.monitor,
                verbose=1,
                save_best_only=False,
                save_weights_only=True,
                mode=self.monitor_mode,
                period=1
            )
            callbacks.append(last_model_checkpoint)

        if 'early_stopping' in self.callbacks_list:
            early_stopping = EarlyStopping(
                monitor=self.monitor,
                min_delta=0,
                patience=self.early_stopping_patience,
                verbose=1,
                mode=self.monitor_mode
            )
            callbacks.append(early_stopping)

        if 'tensorboard' in self.callbacks_list:
            tensorboard = TensorBoard(log_dir=os.path.join(self.logs_dir, '{0}').format(model_maskname))
            callbacks.append(tensorboard)

        if 'csv_logger' in self.callbacks_list:
            csv_logger = CSVLogger(filename=os.path.join(self.logs_dir, '{0}.log').format(model_maskname))
            callbacks.append(csv_logger)

        if 'learning_rate_scheduler' in self.callbacks_list:
            exp_decay = lambda epoch: self.learning_rate * np.exp(-1 * epoch)
            callbacks.append(LearningRateScheduler(exp_decay, verbose=1))

        return callbacks

    def train(self):
        model = self.model_wrapper.model
        model.compile(optimizer=self._optimizer, loss=self.loss, metrics=self.metrics)
        model.fit_generator(
            self._train_generator,
            epochs=self.epochs,
            callbacks=self._callbacks,
            validation_data=self._valid_generator,
            verbose=1,
            workers=4,
            use_multiprocessing=False,
        )

        if self.test_mode:
            model.load_weights(os.path.join(self.nn_models_dir, 'best_{0}{1}.h5').format(self.alias, self.model_wrapper.name))

            scores = model.evaluate_generator(self._test_generator)
            results_path = os.path.join(self.logs_dir, 'test_{0}{1}.csv').format(self.alias, self.model_wrapper.name)
            pd.DataFrame({'Metrics': model.metrics_names, 'Scores': scores})\
                .to_csv(os.path.join(results_path), index=False)
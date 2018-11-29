import abc
import os
import random
import sys
import numpy as np


class Dataset(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, name, log1p_target=True):

        dataset_dir = os.path.join('../../data', name)
        self.images_dir = os.path.join(dataset_dir, 'images')
        self.image_names = os.listdir(self.images_dir)
        self.labels_dict = np.load(os.path.join(dataset_dir, 'labels_dict.npy')).item()
        self._log1p_target = log1p_target
        self.splits = None

        if self._log1p_target:
            self.labels_dict = {key: np.log1p(value) for key, value in self.labels_dict.items()}

        sorted(self.image_names)

    def split_train_test_valid(self, split_fractions=(0.7, 0.2, 0.1), seed=None):

        if abs(sum(split_fractions) - 1) > sys.float_info.epsilon:
            raise ValueError('The fractions of the parts must be equal to 1')

        random.seed(seed)
        random.shuffle(self.image_names)

        length = len(self.image_names)
        train_size = int(length * split_fractions[0])
        train_valid_size = int(length * (split_fractions[0] + split_fractions[1]))

        self.splits = dict()
        self.splits['train_image_names'] = self.image_names[:train_size]
        self.splits['valid_image_names'] = self.image_names[train_size: train_valid_size]
        self.splits['test_image_names'] = self.image_names[train_valid_size:]

        self.image_names.sort()

        return self.splits

    def __len__(self):
        return len(self.image_names)

    @abc.abstractmethod
    def get_absolute_path(self, image_name):
        return os.path.join(self.images_dir, image_name)
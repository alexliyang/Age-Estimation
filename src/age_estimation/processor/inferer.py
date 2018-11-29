import numpy as np
from scipy.misc import imresize
from utils import read_by_pyvips
from tensorflow.keras.utils import Progbar


class Inferer(object):

    def __init__(self, model_wrapper, dataset):
        self.model_wrapper = model_wrapper
        self.dataset = dataset

    def predict(self):

        prediction_dict = dict()
        progbar = Progbar(target=len(self.dataset.image_names))

        for image_name in self.dataset.image_names:
            progbar.add(1)
            image = read_by_pyvips(self.dataset.get_absolute_path(image_name))
            image = imresize(image, (self.model_wrapper.input_shape[0], self.model_wrapper.input_shape[1]))
            image = image / 255.

            prediction = self.model_wrapper.model.predict(np.expand_dims(image, axis=0))[0][0]
            prediction_dict[image_name] = prediction

        return prediction_dict
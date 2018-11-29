import os
import numpy as np
from processor.face import FaceDetector
from processor.face import FaceAligner
from scipy.io import loadmat
from scipy.misc import imsave
from skimage import exposure
from skimage import img_as_ubyte
from utils.helper import read_image_like_rgb
from utils.helper import matlab2datetime
from tensorflow.keras.utils import Progbar
from .base import BaseDatasetPreparator


class ImdbDatasetPreparator(BaseDatasetPreparator):

    def __init__(self, fname, origin, file_hash, dataset_name):
        super().__init__(fname, origin, file_hash, dataset_name)

    def prepare(self):
        raise NotImplementedError


class WikiDatasetPreparator(BaseDatasetPreparator):

    def __init__(self, fname, origin, file_hash, dataset_name):
        super().__init__(fname, origin, file_hash, dataset_name)


    def prepare(self):
        self._create_own_dir()
        unzip_dir = os.path.join(self.unzip_dir, 'wiki')
        mat_file = loadmat(os.path.join(unzip_dir, 'wiki.mat'))
        labels_dict = self._parse_mat(mat_file)

        face_detector = FaceDetector()
        face_aligner = FaceAligner(padding=0.1)
        progbar = Progbar(target=len(labels_dict))

        for image_subpath in list(labels_dict.keys()):
            progbar.add(1)
            image = read_image_like_rgb(os.path.join(unzip_dir, image_subpath))
            image = img_as_ubyte(exposure.equalize_adapthist(image))
            face_bboxes = face_detector.safe_detect_face_bboxes(image, include_cnn=False).clip(min=0)

            if face_bboxes.shape[0] == 0:
                continue

            else:
                cropped_image = face_aligner.align_and_crop(image, bboxes=face_bboxes, bbox_number=0)
                image_name = image_subpath.split('/')[1]
                image_path = os.path.join(self.data_dir, self.dataset_name, 'images', image_name)
                imsave(image_path, cropped_image)

        labels_path = os.path.join(self.data_dir, self.dataset_name, 'labels_dict.npy')
        labels_dict = {key.split('/')[1]: value for key, value in labels_dict.items()}
        np.save(labels_path, labels_dict)


    def _parse_mat(self, mat_file):
        labels_dict = dict()
        data = mat_file['wiki'][0, 0]
        data = data.astype(np.object)

        dob_column = data[0][0].tolist()
        photo_taken_column = data[1][0].tolist()
        full_path_column = [full_path[0] for full_path in data[2][0].tolist()]
        face_score_column = data[6][0].tolist()
        second_face_score_column = data[7][0].tolist()

        for index in range(len(full_path_column)):
            face_score = face_score_column[index]
            second_face_score = second_face_score_column[index]

            if face_score != -np.inf and face_score >= 1 and np.isnan(second_face_score):
                labels_dict[full_path_column[index]] = photo_taken_column[index] - matlab2datetime(dob_column[index]).year

        labels_dict = {key: value for key, value in labels_dict.items() if value > 0 and value <= 100}

        return labels_dict


class SoFDatasetPreparator(BaseDatasetPreparator):

    def __init__(self, fname, origin, file_hash, dataset_name):
        super().__init__(fname, origin, file_hash, dataset_name)

    def prepare(self):
        self._create_own_dir()
        unzip_dir = self.unzip_dir
        image_names = os.listdir(unzip_dir)
        labels_dict = dict()

        face_detector = FaceDetector()
        face_aligner = FaceAligner(padding=0.1)
        progbar = Progbar(target=len(image_names))

        for image_name in image_names:
            progbar.add(1)
            age = image_name.split('_')[3]
            labels_dict[image_name] = int(age)

            image = read_image_like_rgb(os.path.join(unzip_dir, image_name))
            image = img_as_ubyte(exposure.equalize_adapthist(image))
            face_bboxes = face_detector.safe_detect_face_bboxes(image, include_cnn=False).clip(min=0)

            if face_bboxes.shape[0] == 0:
                cropped_image = self._crop_center(image, image.shape[0]//2, image.shape[1]//2)

            else:
                cropped_image = face_aligner.align_and_crop(image, bboxes=face_bboxes, bbox_number=0)

            image_path = os.path.join(self.data_dir, self.dataset_name, 'images', image_name)
            imsave(image_path, cropped_image)

        labels_path = os.path.join(self.data_dir, self.dataset_name, 'labels_dict.npy')
        np.save(labels_path, labels_dict)

    def _crop_center(self, image, y_crop, x_cropx):
        x_start = image.shape[1] // 2 - (x_cropx // 2)
        y_start = image.shape[0] // 2 - (y_crop // 2)

        return image[y_start: y_start + y_crop, x_start: x_start + x_cropx]
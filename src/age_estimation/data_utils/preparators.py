import os
from face import crop_image
from face import FaceDetector
from helpers import read_by_pyvips
from .base import BaseDatasetPreparator

import matplotlib.pyplot as plt


class WikiDatasetPreparator(BaseDatasetPreparator):

    def __init__(self, fname, origin, file_hash, dataset_name):
        super().__init__(fname, origin, file_hash, dataset_name)

    def prepare(self):
        self._create_own_dir()
        unzip_dir = os.path.join('D:/Repositories/Age-Estimation/data/wiki', 'wiki')
        subdir_names = [subdir_name for subdir_name in os.listdir(unzip_dir) if not subdir_name.endswith('.mat')]
        face_detector = FaceDetector()

        for subdir_name in subdir_names:
            subdir = os.path.join(unzip_dir, subdir_name)

            for image_name in os.listdir(subdir):
                print(image_name)
                image = read_by_pyvips(os.path.join(subdir, image_name))
                face_bboxes = face_detector.detect_face_bboxes(image, detector_type='mtcnn')

                if face_bboxes.shape[0] == 0:
                    continue
                else:
                    cropped_image = crop_image(image, face_bboxes.clip(min=0), bbox_number=0)
                    # plt.imshow(image)
                    # plt.show()
                    # plt.imshow(cropped_image)
                    # plt.show()

                a = 5



        a = 45







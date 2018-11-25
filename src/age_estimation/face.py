import warnings
import cv2
import dlib
import numpy as np
from mtcnn.mtcnn import MTCNN
warnings.filterwarnings('ignore')


def crop_image(image, bboxes, bbox_number=0, padding=0):
    crop_image = image[
                 bboxes[bbox_number][1]-padding : bboxes[bbox_number][1]+bboxes[bbox_number][3]+padding,
                 bboxes[bbox_number][0]-padding : bboxes[bbox_number][0]+bboxes[bbox_number][2]+padding,
                 :]

    return crop_image


class FaceDetector(object):

    def __init__(self):
        self.cascade_scale_factor = 1.5
        self.cascade_min_neighbors = 5
        self.dlib_upsample = 1
        self.mtcnn_confidence_threshold = 0.95
        self.mtcnn_init = MTCNN()

        self.haarcascade_xml_path = '../../detectors/haarcascades/haarcascade_frontalface_alt2.xml'
        self.lbpcascade_xml_path = '../../detectors/lbpcascades/lbpcascade_frontalface_improved.xml'
        self.cnn_dat_path = '../../detectors/mmod_human_face_detector.dat'

    def detect_face_bboxes(self, image, detector_type='haarcascade'):

        if detector_type == 'haarcascade' or detector_type == 'lbpcascade':
            return self._detect_face_by_cascade(image, detector_type)

        elif detector_type == 'hogsvm' or detector_type == 'cnn':
            return self._detect_face_by_hogsvm_cnn(image, detector_type)

        elif detector_type == 'mtcnn':
            return self._detect_face_by_mtcnn(image)

        else:
            raise ValueError('There is no such detector, available: haarcascade, lbpcascade, hogsvm, cnn, mtcnn')

    def _detect_face_by_cascade(self, image, detector_type):
        face_detectors = {
            'haarcascade': cv2.CascadeClassifier(self.haarcascade_xml_path),
            'lbpcascade': cv2.CascadeClassifier(self.lbpcascade_xml_path)
        }
        face_detector = face_detectors[detector_type]
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bboxes = face_detector.detectMultiScale(
            gray_image,
            scaleFactor=self.cascade_scale_factor,
            minNeighbors=self.cascade_min_neighbors
        )

        if isinstance(bboxes, tuple):
            return np.asarray(list())
        else:
            return bboxes

    def _detect_face_by_hogsvm_cnn(self, image, detector_type):
        face_detectors = {
            'hogsvm': dlib.get_frontal_face_detector(),
            'cnn': dlib.cnn_face_detection_model_v1(self.cnn_dat_path)
        }
        face_detector = face_detectors[detector_type]
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rectangles = face_detector(gray_image, self.dlib_upsample)
        if isinstance(rectangles, dlib.mmod_rectangles):
            rectangles = [mmod_rectangle.rect for mmod_rectangle in rectangles]

        bboxes = list()
        for rectangle in rectangles:
            bboxes.append([
                rectangle.left(),
                rectangle.top(),
                rectangle.right() - rectangle.left(),
                rectangle.bottom() - rectangle.top()
            ])

        return np.asarray(bboxes)

    def _detect_face_by_mtcnn(self, image):
        face_objects = self.mtcnn_init.detect_faces(image)
        bboxes = [face_object['box'] for face_object in face_objects if
                  face_object['confidence'] > self.mtcnn_confidence_threshold]

        return np.asarray(bboxes)
import cv2
import dlib
import math
import warnings
import numpy as np
from utils.helper import crop_image
from mtcnn.mtcnn import MTCNN
warnings.filterwarnings('ignore')


class FaceDetector(object):

    def __init__(self, cascade_scale_factor=1.5, cascade_min_neighbors=5, dlib_upsample=1, mtcnn_confidence_threshold=0.95,
                 haarcascade_xml_path='../detectors/haarcascades/haarcascade_frontalface_alt2.xml',
                 lbpcascade_xml_path='../detectors/lbpcascades/lbpcascade_frontalface_improved.xml',
                 cnn_dat_path='../detectors/mmod_human_face_detector.dat'):

        self.cascade_scale_factor = cascade_scale_factor
        self.cascade_min_neighbors = cascade_min_neighbors
        self.dlib_upsample = dlib_upsample
        self.mtcnn_confidence_threshold = mtcnn_confidence_threshold
        self.mtcnn_init = MTCNN()

        self.haarcascade_xml_path = haarcascade_xml_path
        self.lbpcascade_xml_path = lbpcascade_xml_path
        self.cnn_dat_path = cnn_dat_path

    def safe_detect_face_bboxes(self, image, include_cnn=False):

        bboxes = self.detect_face_bboxes(image, detector_type='mtcnn')

        if bboxes.shape[0] == 0:
            bboxes = self.detect_face_bboxes(image, detector_type='hogsvm')

        if bboxes.shape[0] == 0:
            bboxes = self.detect_face_bboxes(image, detector_type='haarcascade')

        if bboxes.shape[0] == 0 and include_cnn:
            bboxes = self.detect_face_bboxes(image, detector_type='cnn')

        return bboxes

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


class FacemarkDetector(object):

    def __init__(self, facemarks_data_path='../detectors/shape_predictor_5_face_landmarks.dat'):
        self.facemarks_data_path = facemarks_data_path
        self.facemarks_predictor = dlib.shape_predictor(facemarks_data_path)

    def detect_facemarks_coords(self, image, bboxes):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rectangles = dlib.rectangles()
        rectangles.extend([dlib.rectangle(left=bbox[0],
                                          top=bbox[1],
                                          right=bbox[2] + bbox[0],
                                          bottom=bbox[3] + bbox[1]) for bbox in bboxes])
        facemarks_coords = list()
        for rectangle in rectangles:
            facemarks = self.facemarks_predictor(gray_image, rectangle)
            facemarks_coords.append(self._facemarks_to_coords(facemarks))

        return facemarks_coords

    def _facemarks_to_coords(self, facemarks, n_points=5, dtype=np.int):
        coords = np.zeros((n_points, 2), dtype=dtype)
        for i in range(0, n_points):
            coords[i] = (facemarks.part(i).x, facemarks.part(i).y)

        return coords


class FaceAligner(object):

    def __init__(self, padding=0.1):
        self.facemark_detector = FacemarkDetector()
        self.padding = padding

    def align_and_crop(self, image, bboxes, bbox_number=0):
        facemarks_coords = self.facemark_detector.detect_facemarks_coords(image, bboxes=bboxes)
        right_eye_center = facemarks_coords[bbox_number][0:2].mean(axis=0).astype(int)
        left_eye_center = facemarks_coords[bbox_number][2:4].mean(axis=0).astype(int)

        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX))

        face_center = facemarks_coords[bbox_number].mean(axis=0).astype(int)
        rotation_matrix = cv2.getRotationMatrix2D(center=(face_center[0], face_center[1]), angle=angle, scale=1)
        rotated_image = cv2.warpAffine(
            src=image, M=rotation_matrix,
            dsize=(image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC
        )

        rotated_bboxes = self._rotate_bboxes(bboxes, center=face_center, angle=angle)
        padding = int(max(rotated_bboxes[bbox_number][2:]) * self.padding)
        cropped_image = crop_image(rotated_image, rotated_bboxes, bbox_number=bbox_number, padding=padding)

        return cropped_image

    def _rotate_bboxes(self, bboxes, center, angle):
        rotated_bboxes = bboxes.copy()
        rad = math.radians(angle)
        ox, oy = center

        for index in range(len(bboxes)):
            px, py = bboxes[index][0], bboxes[index][1]
            rotated_bboxes[index][0] = ox + math.cos(rad) * (px - ox) - math.sin(rad) * (py - oy)
            rotated_bboxes[index][1] = oy + math.sin(rad) * (px - ox) + math.cos(rad) * (py - oy)

        return rotated_bboxes
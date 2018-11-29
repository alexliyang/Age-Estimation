import pyvips
import numpy as np
import datetime


def read_by_pyvips(path, grayscale=False):
    image = pyvips.Image.new_from_file(path, access='sequential')
    if grayscale:
        image = image.colourspace('b-w')

    memory_image = image.write_to_memory()
    numpy_image = np.ndarray(
        buffer=memory_image,
        dtype=np.uint8,
        shape=[image.height, image.width, image.bands]
    )

    return numpy_image

def read_image_like_rgb(path):
    image = read_by_pyvips(path)
    if image.shape[2] == 1:
        return np.squeeze(np.stack((image,) * 3, -1))

    elif image.shape[2] == 2:
        return np.squeeze(np.stack((image[..., 0],) * 3, -1))

    elif image.shape[2] == 4:
        return image[..., :3]

    else:
        return image

def crop_image(image, bboxes, bbox_number=0, padding=0):
    left = bboxes[bbox_number][1]
    top = bboxes[bbox_number][0]
    width = bboxes[bbox_number][3]
    height = bboxes[bbox_number][2]

    left_minus_pad = 0 if left-padding < 0 else left-padding
    top_minus_pad = 0 if top-padding < 0 else top-padding

    crop_image = image[left_minus_pad: left+width+padding, top_minus_pad: top+height+padding, :]

    return crop_image

def matlab2datetime(matlab_datenum):
    day = datetime.datetime.fromordinal(int(matlab_datenum))
    dayfrac = datetime.timedelta(days=matlab_datenum % 1) - datetime.timedelta(days=366)

    return day + dayfrac
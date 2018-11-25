import pyvips
import numpy as np

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
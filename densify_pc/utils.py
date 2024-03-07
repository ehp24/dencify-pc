"""Module utils: general functions """
import os
from PIL import Image
import numpy as np

def read_img_uint32(filename):
    """ Takes a single image path and returns np array representation of image in uint32."""
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    rgb_png = np.array(img_file, dtype='uint32')  # LAS stores its coordinates in uint32
    img_file.close()
    return rgb_png
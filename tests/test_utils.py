import os
import pytest
from PIL import Image
import numpy as np
from densify_pc.utils import read_img_uint32


@pytest.fixture(params=["png","jpg"])
def sample_img(tmpdir,request):
    # Create test image fixture both png and jpg and return temp img path
    img_data = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(img_data)
    img_path = os.path.join(tmpdir, f"test_image.{request.param}")
    img.save(img_path)
    return img_path

def test_read_img_uint32(sample_img):
    # Test reading img with size, type
    img_array = read_img_uint32(sample_img)
    assert isinstance(img_array, np.ndarray)
    assert img_array.dtype == np.uint32
    assert img_array.shape == (100, 100, 3) 
    
def test_read_img_uint32_with_nonexistent_file():
    # check function errors if given a non existent file path
    with pytest.raises(AssertionError):
        read_img_uint32("nonexistent_file.png")

def test_read_img_uint32_with_invalid_image_format(tmp_path):
    # Test whether passing a txt file produces an error
    invalid_image_path = tmp_path / "invalid_image.txt"
    with open(invalid_image_path, "w") as f:
        f.write("This is not an image file, it is a txt.")
    with pytest.raises(Exception):
        read_img_uint32(str(invalid_image_path))
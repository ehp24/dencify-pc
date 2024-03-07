import os
import pytest
import csv
import numpy as np
import laspy
from densify_pc.dataprocessor import read_csv, convertLAS2numpy

@pytest.fixture
def sample_csv(tmpdir):
    # Create a sample CSV file for testing
    csv_data = [{'file_name': 'image1', 'data': 'data1'},
                {'file_name': 'image2', 'data': 'data2'},
                {'file_name': 'image3', 'data': 'data3'}]
    
    csv_path = os.path.join(tmpdir, 'sample.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['file_name', 'data']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    return csv_path

def test_read_csv(sample_csv):
    # Test reading CSV data 
    img_paths_list = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
    img_csv_hmap = read_csv(sample_csv, img_paths_list)
    assert isinstance(img_csv_hmap, dict)
    assert 'image1' in img_csv_hmap
    assert 'image2' in img_csv_hmap
    assert 'image3' in img_csv_hmap
    assert img_csv_hmap['image1'] == {'file_name': 'image1', 'data': 'data1'}
    assert img_csv_hmap['image2'] == {'file_name': 'image2', 'data': 'data2'}
    assert img_csv_hmap['image3'] == {'file_name': 'image3', 'data': 'data3'}
    
    
    
@pytest.fixture
def fake_las(tmpdir):
    # Create a fake LAS file with specified scales and offsets
    header = laspy.LasHeader()
    header.offsets = [1000, 2000, 3000]
    header.scales = [0.01, 0.02, 0.03]
    
    las = laspy.LasData(header)
    num_points = 100
    las.X = np.random.randint(0, 1000, num_points)
    las.Y = np.random.randint(0, 1000, num_points)
    las.Z = np.random.randint(0, 1000, num_points)
    return las

def test_convertLAS2numpy(fake_las):
    # Test whether LAS converts correctly to numpy
    numpy_array = convertLAS2numpy(fake_las)
    
    assert isinstance(numpy_array, np.ndarray)
    assert np.allclose(numpy_array[0], (fake_las.X * fake_las.header.scales[0]) + fake_las.header.offsets[0])
    assert np.allclose(numpy_array[1], (fake_las.Y * fake_las.header.scales[1]) + fake_las.header.offsets[1])
    assert np.allclose(numpy_array[2], (fake_las.Z * fake_las.header.scales[2]) + fake_las.header.offsets[2])


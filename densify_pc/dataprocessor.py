"""process_data : functions to do with fetching/ processing/ converting data."""
import csv
import os
import numpy as np
import laspy

def read_csv(csvpath,imgs_paths_list):
    """Extracts csv data for corresponding images in list of image paths.

    Args:
        csvpath (str): The path to the csv file.
        imgs_paths_list (list): A list of paths (str) to image files.

    Returns:
        list: A list of dictionaries, each dict containing csv data for each correpsonding img in img_paths_list.
    """
    
    img_names = [os.path.splitext(imgpath.split('/')[-1])[0] for imgpath in imgs_paths_list ] # list of img names instead of img paths
    
    with open(csvpath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        imgdata = [row for row in reader if row['file_name'] in img_names] # each row is a dict and represents data for one unique image

    # first = imgdata[0]
    # print(first['file_name'])
    
    # each dict is a row in the csv file
    # need to write test to check there is a correponding csv row to image match - i.e. must have same name 
    return imgdata





def convertLAS2numpy(path2las):
    """Creates a Numpy representation of all points in LAS file.

    Args:
        path2las (str): The path to LAS point cloud file to process.

    Returns:
        numpy.ndarray: An m*n Numpy array of data points, m = 8 for each data type (xcoord, ycoord etc) and n = number of points in LAS file.
    """
    
    las = laspy.read(path2las) # creates LasData object instance which conatins all the points in the point cloud
   
    # each las point has a real coord that would need to eb stored as a float64 due to decimals
    # float64 requires much more memory and wuld be slower to process hence convert each coord to uint32 by applying a scale and offset
    # hence to obtain the original raw coordinates, need to apply the specified scale and offset defined in the las header, to all the points
    x_scale, y_scale, z_scale = las.header.scales
    x_offset, y_offset, z_offset = las.header.offsets
    x_coords = (las.X * x_scale) + x_offset
    y_coords = (las.Y * y_scale) + y_offset
    z_coords = (las.Z * z_scale) + z_offset
    ones = np.ones(len(x_coords))
    
    reds  = (las.red)/256 #LAS stores RGB as 16-bit, but usually we want 8-bit so convert to 8bit by dividing by 256
    greens = (las.green)/256
    blues = (las.blue)/256 
    greyscale = (np.add(reds,(np.add(blues,greens))))/3
    homogeneous_data_matrix = np.vstack((x_coords,y_coords,z_coords,ones,blues,greens,reds,greyscale)) # returns [x,y,z,1,B,G,R,greyscale]
    # first row is x, next row y, next row z
    # each column represents a differnt point
    return homogeneous_data_matrix
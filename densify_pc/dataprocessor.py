"""process_data : functions to do with fetching/ processing/ converting data."""
import csv
import os
import numpy as np
import laspy
from scipy import interpolate

def read_csv(csvpath,imgs_paths_list):
    """Extracts csv data for corresponding images in list of image paths.

    Args:
        csvpath (str): The path to the csv file.
        imgs_paths_list (list): A list of paths (str) to image files.

    Returns:
        dict: A nested dictionary, each dict key is imgfilename which has value containing the correponsing csv row in a dict if found. 
    """
    
    img_names = [os.path.splitext(imgpath.split('/')[-1])[0] for imgpath in imgs_paths_list ] # list of img names instead of img paths
    
    with open(csvpath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        # imgdata = [row for row in reader if row['file_name'] in img_names] # each row is a dict and represents data for one unique image
        img_csv_hmap = {row['file_name']:row for row in reader if row['file_name'] in img_names}
    
    # need to write test to check there is a correponding csv row to image match - i.e. must have same name 
   
    return img_csv_hmap





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


def interpolate_dmap(projected_dmap): 
    """Takes LAS projected depth map on image and returns interpolated depth map - [x,y,z] at every [u,v] pixel. 

    Args:
        projected_dmap (numpy.ndarray): A projected depth map of LAS points onto img plane (pix coord).
        
    Returns:
        numpy.ndarray: An interpolated depth map so that every pixel in image has a correponsing [x_wc,y_wc,z_wc]
    """
    
    # ALWAYS REMEBER - THE FIRST INDEX IS THE ROW I.E. Y COORD!
    x_cols = np.arange(0,projected_dmap.shape[1],1)
    y_rows = np.arange(0,projected_dmap.shape[0],1)
    x_grid, y_grid =np.meshgrid(x_cols,y_rows) # makes two grids, both of the shape of the img or x_cols x y_cols and fills the first with all the xvalues and the secodn one with all the y vals
    
    # the below argwhere usualy gives [ [x1,y1], [x3,y3], ... [xN,yN]] etc, i.e. a list of coords where we dont have [0,0,0](black pixel). the next two lines slplit this into the x values, then the y values
    y_las = np.argwhere((projected_dmap !=0).any(axis=2))[:,0] # x coords of img pix plane where we have data pts from projected las
    x_las = np.argwhere((projected_dmap !=0).any(axis=2))[:,1] # y coords of img pix plane where we have data pts from projected las
    data_las = projected_dmap[y_las,x_las] # gives all the array values [x_wc, y_wc, z_wc] that have non 0 value, i.e. the x_wc_16,y_wc_16,z_wc_16 values of the projected points, i.e. all the poitns stored at each xlas, ylas
    
    # create interpolated array with full XYZ_wc values at every img pixel
    # interLinear_map = interpolate.griddata((x_las,y_las),data_las,(x_grid,y_grid),method='linear') #for soem reason interlinear does not work
    raw_interp_dmap = interpolate.griddata((x_las,y_las),data_las,(x_grid,y_grid),method='linear') # raw interpolated depth map
    
    # need to process for rogue points
    interp_map_x = np.clip(raw_interp_dmap[:,:,0], np.nanmin(data_las[:,0]),np.nanmax(data_las[:,0]))
    interp_map_y = np.clip(raw_interp_dmap[:,:,1], np.nanmin(data_las[:,1]),np.nanmax(data_las[:,1]))
    # for clipping:
    # interNearest_map_z = np.clip(interNearest_map[:,:,2], np.nanmin(data_las[:,2]),np.nanmin(data_las[:,2]) + 0.1*(np.nanmax(data_las[:,2]) - np.nanmin(data_las[:,2])))
    interp_map_z = np.clip(raw_interp_dmap[:,:,2], np.nanmin(data_las[:,2]),np.nanmin(data_las[:,2]))

    interp_dmap = np.dstack((interp_map_x,interp_map_y,interp_map_z)) # interpolated depth map [x,y,z] at each [u,v] pixel
# interpolates the availaible projected points on the image array grid
    return interp_dmap

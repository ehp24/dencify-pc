import glob
import os
import laspy
import numpy as np
import math
import time
from densify_pc.utils import read_img_uint32
from densify_pc.dataprocessor import read_csv, convertLAS2numpy, interpolate_dmap, convertnumpy2LAS
from densify_pc.projection import rotMat, extrinsicsMat, intrinsicsMat, projection_WCS2PCS

def main():
    
    # initialise time 
    start_time = time.time()
    
    # fetch paths to data
    rootdir_path = os.path.dirname(__file__) 
    csv_path = (glob.glob(os.path.join(rootdir_path,"data/raw/csv/*.csv")))[0] 
    imgs_path_list = sorted(glob.glob(os.path.join(rootdir_path,'data/raw/rgb/*')),reverse=True) 
    LAS_path = (glob.glob(os.path.join(rootdir_path,'data/raw/las/*.LAS')))[0]
    
    # extracts csv data and las object
    csv_img_data = read_csv(csv_path,imgs_path_list)
    las = laspy.read(LAS_path) 
    
    # data about number of images for progress tracking
    no_imgs = len(imgs_path_list)
    no_rows_extracted = len(csv_img_data) # no. of imgs we are operating on
    im_count = 0
    not_present = []
    
    # convert LAS points to numpy format
    LAS_points = convertLAS2numpy(las)
    
    # A11 Red Lodge Lane 1 error correction
    error_correct = [-1,-0.5,1.5] 

    # Output settings
    skip_pts = True # skip some points for lower densification + less memory 
    n = 3 # skips every nth col and every nth row, hence nxn times less points

    for path2img in imgs_path_list:
        
        im_count+=1
        img_name = (os.path.splitext(path2img)[0]).split('/')[-1] # name of img e.g. A11redlodge0006200010Lane1
        filename = path2img.split('/')[-1]
        
        # np array of img
        img_array_uint32 = read_img_uint32(path2img) 
        
        # extract the csv row (dict) for the correponsing img
        if img_name in csv_img_data.keys():
            csv_img_data = csv_img_data[img_name]
        else: # img in data folder is not found in csv file, skip img
            not_present.append(filename)
            continue

        # project LAS points to img plane
        rgbd, projectedimg, projected_LAS_data_map = projection_WCS2PCS(csv_img_data, LAS_points, img_array_uint32, las, error_correct)
        
        # interpolate the projected dmap = depth map
        projected_LAS_dmap = projected_LAS_data_map[:,:,:3] # array with just [x_wc_uint32, r_wc_uin32, z_wc_uin32] projected depth map
        interpolated_dmap = interpolate_dmap(projected_LAS_dmap)
        
        # combine interpolated dmap [x,y,z] and orginal image [rgb]
        interpolated_im_depth_map = np.dstack((interpolated_dmap,img_array_uint32 *256 )) 
        
        if skip_pts == True:
            interpolated_im_depth_map = interpolated_im_depth_map[::n,::n,:]
        interpolated_im_depth_map = np.round(interpolated_im_depth_map)
        
        # data to create LAS 
        no_pts = interpolated_im_depth_map.shape[0]*interpolated_im_depth_map.shape[1] # tot no. las points = h x w of the image 
        new_LAS_points = interpolated_im_depth_map.reshape((no_pts,6)).T # convert to 2D array for processing, each row is [x1,x2 ...] then [y1,y2....] etc 
        nan_row_array = ~np.isnan(new_LAS_points).any(axis = 0) 
        new_LAS_points = new_LAS_points[:,nan_row_array] 
        
        # combine las points from all imgs together
        if im_count == 1:
            all_las_pts = new_LAS_points
        else:
            all_las_pts = np.concatenate((all_las_pts,new_LAS_points),axis=1)  
        
        # update variables for progress bar
        progress = int(im_count/no_rows_extracted*100)
        print(f"{progress}% done ------- {im_count}/{no_rows_extracted} images processed")
    
    
    print("Now creating the LAS file -------->")

    # convert interpolated np array points to LAS object
    densified_las = convertnumpy2LAS(las,all_las_pts,z_offset=200) 
    newlasname = os.path.join(rootdir_path,"result","result_pc"+'.las')
    densified_las.write(newlasname)
    
    if no_rows_extracted != no_imgs:
        print(f"WARNING: there are {no_imgs} images present but only {no_rows_extracted} images were found in the existing csv file." )
        print("The follwing images present in the data folder were not found in CSV file, please check images:")
        print(not_present)    
    
    end_time = time.time()
    print(f'Total runtime: {end_time-start_time}s')
    
if __name__ == '__main__':
    main()
    




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
    rootdir_path = os.path.dirname(__file__) # gives path to registration full code folder (i.e folder where this file is stored)
    csv_path = (glob.glob(os.path.join(rootdir_path,"data/raw/csv/*.csv")))[0] # should chekc we only hva eone file
    imgs_path_list = sorted(glob.glob(os.path.join(rootdir_path,'data/raw/rgb/*')),reverse=True) # returns lsit of images, dont constrain it to be a specific filetype incase png or jpg?
    LAS_path = (glob.glob(os.path.join(rootdir_path,'data/raw/las/*.LAS')))[0] #should allow LAS or las, and should check we only hva eone file
    
    # fetch corresponding csv data for the imgs present
    csv_img_data = read_csv(csv_path,imgs_path_list)
    
    # data about number of images for progress tracking
    no_imgs = len(imgs_path_list)
    no_rows_extracted = len(csv_img_data) #i.e. the number of imgs we are actually oeprating on
    im_count = 0
    not_present = []
    
    las = laspy.read(LAS_path) # extract LAS data from path
    # convert LAS points to numpy format
    LAS_points = convertLAS2numpy(las)
    
    # Manually found error corrections - will vary for different highways 
    error_correct = [-1,-0.5,1.5] # for first lane of A11 Red lodge, Lane 1 MAYBE PUT THIS IN DATA FILE SEPARATELY

    # Output settings
    increase_z = 0 # z offset for visual aid (as dense pc points coincide with normal pc points) LIMIT THIS
    skip_pts = True # if we want to skip some points for lower densification + less memory (still processed all of them tho so not faster..)
    n = 3 # skips every nth col and every nth row, hence nxn times less points

    for path2img in imgs_path_list:
        
        im_count+=1
        
        img_name = (os.path.splitext(path2img)[0]).split('/')[-1] # name of img e.g. A11redlodge0006200010Lane1
        img_type = os.path.splitext(path2img)[1] # jpg
        filename = path2img.split('/')[-1]

        # extract the csv row (dict) for the correponsing img
        if img_name in csv_img_data.keys():
            csv_img_data = csv_img_data[img_name]
        else: # img in data folder is not found in csv file, skip img
            not_present.append(filename)
            continue

        # # create extrinsics and intrinsics matrix COULD THIS LIVE INSIDE PROJECTION AND NOT NEED TO BE HERE?
        # f = 8.5 #mm
        # rph = [float(data['roll[deg]']), float(data['pitch[deg]']), float(data['heading[deg]'])] # [roll, pitch, heading]
        # xyz = [float(data['projectedX[m]']),float(data['projectedY[m]']),float(data['projectedZ[m]'])] # [x,y,z]
        # extr_mat = extrinsicsMat(rph,xyz,error_correct)
        # intr_mat = np.hstack((intrinsicsMat(f),np.array([[0.],[0.],[0.]])))
        
        # # image object in np
        img_array_uint32 = read_img_uint32(path2img)
         
        # im_height = img_array_uint32.shape[0]
        # im_width = img_array_uint32.shape[1]
        
        # project LAS points to img plane (pix coords)
        rgbd, projectedimg, LAS_data_array = projection_WCS2PCS(csv_img_data, LAS_points, img_array_uint32, las, error_correct)
        
        # interpolate the projected depth map
        # mapped_LAS_pts_rgb = LAS_data_array[:,:,3:] # creates array with just r_wc_uint16,g_wc_uint16,b_wc_uint16, though original rgb is redundsnt for now
        projected_LAS_dmap = LAS_data_array[:,:,:3] # creates array with just [x_wc_uint32, r_wc_uin32, z_wc_uin32] projected depth map
        interpolated_dmap = interpolate_dmap(projected_LAS_dmap)
        
        imArray_uint32 = img_array_uint32 *256 #NOT SURE WHAT DOIG N HERE, CONVERTING UINT32 TO ???
        
        interpolated_im_depth_map = np.dstack((interpolated_dmap,imArray_uint32)) # combining the orginal rgb vals from the 2D img with the XYZ coords extracted from LAS and interpolated that proiject to the image area 
        if skip_pts == True:
            interpolated_im_depth_map = interpolated_im_depth_map[::n,::n,:]
        interpolated_im_depth_map = np.round(interpolated_im_depth_map) # make sure theres no decimal places maybe?
        
        # data to create LAS 
        no_pts = interpolated_im_depth_map.shape[0]*interpolated_im_depth_map.shape[1] # number of new las points should just be h x w of the image depth map array
        new_LAS_points = interpolated_im_depth_map.reshape((no_pts,6)).T # convert 3D array into 2D array of all the points, as converting to LAS doesnt need it in the image structure so [x1,x2,x3...], [y1,y2,y3,...], [z1,z2,z3,....],[r1,r2,r3],....etc
        nan_row_array = ~np.isnan(new_LAS_points).any(axis = 0) # i think a boolean array where each element indicates whether the corresponding column in new_LAS_points contains at least one NaN value (True if it does, False otherwise)
        new_LAS_points = new_LAS_points[:,nan_row_array] 
        
        # maybejust initilise all las pints before for loop, and check whetehr its empty or not, rather than do im count
        # combining all of the images points otgether if there ar emultiple images in a super array. first row is x, next y, next z etc. Only 6, xyzrgb
        if im_count == 1:
            all_las_pts = new_LAS_points
        
        else:
            all_las_pts = np.concatenate((all_las_pts,new_LAS_points),axis=1)  
        
        # update variables for progress bar
        progress = int(im_count/no_rows_extracted*100)
        print(f"{progress}% done ------- {im_count}/{no_rows_extracted} images processed")
    
    
    print("Now creating the LAS file -------->")

    # convert interpolated np array points to LAS object
    densified_las = convertnumpy2LAS(las,all_las_pts,increase_z) 
    
    newlasname = os.path.join(rootdir_path,"result","result_pc"+'.las') # maybe keep name of original image in this 
    densified_las.write(newlasname)
    
    if no_rows_extracted != no_imgs:
        print(f"WARNING: there are {no_imgs} images present but only {no_rows_extracted} images were found in the existing csv file." )
        print("The follwing images present in the data folder were not found in CSV file, please check images:")
        print(not_present)    
    
    end_time = time.time()
    print(f'Total runtime: {end_time-start_time}s')
    
if __name__ == '__main__':
    main()
    




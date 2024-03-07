import glob
import os
import laspy
import numpy as np
import math
from scipy import interpolate
import time
from densify_pc.utils import read_img_uint32
from densify_pc.dataprocessor import read_csv, convertLAS2numpy
from densify_pc.projection import rotMat, extrinsicsMat, intrinsicsMat, projection_WCS2PCS

def interpolateImage(): 
    # interpolates the availaible projected points on the image array grid
    pass


def main():
    
    start_time = time.time()
    
    # fetch paths to data
    rootdir_path = os.path.dirname(__file__) # gives path to registration full code folder (i.e folder where this file is stored)
    csv_path = (glob.glob(os.path.join(rootdir_path,"data/raw/csv/*.csv")))[0] # should chekc we only hva eone file
    imgs_path_list = sorted(glob.glob(os.path.join(rootdir_path,'data/raw/rgb/*')),reverse=True) # returns lsit of images, dont constrain it to be a specific filetype incase png or jpg?
    LAS_path = (glob.glob(os.path.join(rootdir_path,'data/raw/las/*.LAS')))[0] #should allow LAS or las, and should check we only hva eone file
    
    # fetched all csv rows for the imgs present in data folder
    csvdata = read_csv(csv_path,imgs_path_list)
    csv_img_data = csvdata.copy() # REDUNDANT?

    
    im_count = 0
    
    # as shown in read_csv function, the values in the columns can be foudn by doing row['file_name']
    
    no_imgs = len(imgs_path_list)
    no_rows_extracted = len(csv_img_data) #i.e. the number of imgs we are actually oeprating on
    not_present = []
    LAS_points = convertLAS2numpy(LAS_path)
    
    # SETTING WHETHER WE WANT TO SKIP POINTS FOR LOWER DENSIFICATION
    skip_pts = True
    # every nth column taken, and every nth row, hence nxn times less points
    n = 3
    
    # ERROR CORRECTIONS - bestEC means best error correction (for first lane of A11 Red lodge, Lane 1)
    bestEC = [-1,-0.5,1.5]
    noEC = [0,0,0]
    error_correct = bestEC
    testname = f"finalLASresult"

    # for if you want to view the denser PC overlaid onto original - because of interpolation the points conflict as they have same depth so its sometimes hard to see the coinciding points 
    increase_z = 0

    
    

    for path2img in imgs_path_list:
        
        
        img_name = (os.path.splitext(path2img)[0]).split('/')[-1] # name of img e.g. A11redlodge0006200010Lane1
        img_type = os.path.splitext(path2img)[1] # jpg
        filename = path2img.split('/')[-1]
        data = None

        for row in csv_img_data: #extract the correct row for correspondign img
            if row['file_name'] == img_name:
                data = row
                break #there shouls be only one row that correponds to an image, so exit once we have foudn the row
                
        if data == None: # if images exist in data folder that are not present in csv file, dont let it go through rest of the code, skip to next iteration
            not_present.append(filename)
            continue

        im_count+=1
        progress = int(im_count/no_rows_extracted*100)
        
        roll = float(data['roll[deg]'])
        pitch = float(data['pitch[deg]'])
        heading = float(data['heading[deg]'])
        x = float(data['projectedX[m]'])
        y = float(data['projectedY[m]'])
        z = float(data['projectedZ[m]'])
        rph = [roll,pitch,heading]
        xyz = [x,y,z]
        
        f = 8.5 #mm
        exMat = extrinsicsMat(rph,xyz,error_correct)
        K = np.hstack((intrinsicsMat(f),np.array([[0.],[0.],[0.]])))
        
        imArray_uint32 = read_img_uint32(path2img) 
        im_height = imArray_uint32.shape[0]
        im_width = imArray_uint32.shape[1]
        
        rgbd, projectedimg, LAS_data_array = projection_WCS2PCS(K, exMat, LAS_points, im_width, im_height, LAS_path)
        
        
        mapped_LAS_pts_wc = LAS_data_array[:,:,:3] # creates array with just x_wc_uint32, r_wc_uin32, z_wc_uin32
        # orgihnal rgb of las is perhaps redundant
        mapped_LAS_pts_rgb = LAS_data_array[:,:,3:] # creates array with just r_wc_uint16,g_wc_uint16,b_wc_uint16

        # now interpolate    # ALWAYS REMEBER - THE FIRST INDEX IS THE ROW I.E. Y COORD!
        x_cols = np.arange(0,mapped_LAS_pts_wc.shape[1],1)
        y_rows = np.arange(0,mapped_LAS_pts_wc.shape[0],1)
        x_grid, y_grid =np.meshgrid(x_cols,y_rows) # makes two grids, both of the shape of the img or x_cols x y_cols and fills the first with all the xvalues and the secodn one with all the y vals
        
        # the below argwhere usualy gives [ [x1,y1], [x3,y3], ... [xN,yN]] etc, i.e. a list of coords where we dont have [0,0,0](black pixel). the next two lines slplit this into the x values, then the y values
        y_las = np.argwhere((mapped_LAS_pts_wc !=0).any(axis=2))[:,0] # x coords of img pix plane where we have data pts from projected las
        x_las = np.argwhere((mapped_LAS_pts_wc !=0).any(axis=2))[:,1] # y coords of img pix plane where we have data pts from projected las
        data_las = mapped_LAS_pts_wc[y_las,x_las] # gives all the array values [x_wc, y_wc, z_wc] that have non 0 value, i.e. the x_wc_16,y_wc_16,z_wc_16 values of the projected points, i.e. all the poitns stored at each xlas, ylas
        

       
        # create interpolated array with full XYZ_wc values at every img pixel
        # interLinear_map = interpolate.griddata((x_las,y_las),data_las,(x_grid,y_grid),method='linear') #for soem reason interlinear does not work
        interNearest_map = interpolate.griddata((x_las,y_las),data_las,(x_grid,y_grid),method='linear')
        
        # print(np.min(data_las[:,0]))
        # input()
        # print(interNearest_map)
        # input()
        
        
        interNearest_map_x = np.clip(interNearest_map[:,:,0], np.nanmin(data_las[:,0]),np.nanmax(data_las[:,0]))
        interNearest_map_y = np.clip(interNearest_map[:,:,1], np.nanmin(data_las[:,1]),np.nanmax(data_las[:,1]))
        # for clipping:
        # interNearest_map_z = np.clip(interNearest_map[:,:,2], np.nanmin(data_las[:,2]),np.nanmin(data_las[:,2]) + 0.1*(np.nanmax(data_las[:,2]) - np.nanmin(data_las[:,2])))
        interNearest_map_z = np.clip(interNearest_map[:,:,2], np.nanmin(data_las[:,2]),np.nanmin(data_las[:,2]))

        interNearest_map = np.dstack((interNearest_map_x,interNearest_map_y,interNearest_map_z))
        
   
    
        # print(f"internearest map:{interNearest_map.shape}")
        imArray_uint32 = read_img_uint32(path2img) *256
        interpolated_im_depth_map = np.dstack((interNearest_map,imArray_uint32)) # combining the orginal rgb vals from the 2D img with the XYZ coords extracted from LAS and interpolated that proiject to the image area 
        if skip_pts == True:
            interpolated_im_depth_map = interpolated_im_depth_map[::n,::n,:]

        
        
        interpolated_im_depth_map = np.round(interpolated_im_depth_map) # make sure theres no decimal places maybe?
        
        # np.savetxt("tmp_array_end.csv",interpolated_im_depth_map[5,:,:], delimiter = ",")
        # print(np.max(interpolated_im_depth_map[:,:,0]),np.min(interpolated_im_depth_map[:,:,0]))
        # print(np.max(interpolated_im_depth_map[:,:,1]),np.min(interpolated_im_depth_map[:,:,1]))
        # print(np.max(interpolated_im_depth_map[:,:,2]),np.min(interpolated_im_depth_map[:,:,2]))
        # interpolated_im_depth_map = interpolated_im_depth_map[0:interpolated_im_depth_map.shape[0]:10,0:interpolated_im_depth_map.shape[1]:10,:,:,:]
        
        
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
        
        print(f"{progress}% done ------- {im_count}/{no_rows_extracted} images completed")
        
    print("Now creating the LAS file -------->")
    # now we have las points of all images, create a LAS file
    # reading again! maybe just have a fucntion that returns the la sobject....
    lasfile = laspy.read(LAS_path)
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.add_extra_dim(laspy.ExtraBytesParams(name="random", type=np.int32))
    header.offsets = lasfile.header.offsets
    header.scales = lasfile.header.scales
    # print(all_las_pts[2,:][0])
    all_las_pts[2,:]=all_las_pts[2,:]+increase_z # maybe create another one for visual purposes?
    # print(all_las_pts[2,:][0])
    newlas = laspy.LasData(header)
    
    newlas.X = all_las_pts[0,:]
    newlas.Y = all_las_pts[1,:]
    newlas.Z = all_las_pts[2,:]
    newlas.red = all_las_pts[3,:]
    newlas.green = all_las_pts[4,:]
    newlas.blue = all_las_pts[5,:]

    # newlasname = rootdir_path +f"/Result/LAS result/projected_las"+ LAS_path.split('/')[-1]
    
    newlasname = os.path.join(rootdir_path,"result",testname+'.las')
    
    
    newlas.write(newlasname)

    
    
    if no_rows_extracted != no_imgs:
        print(f"WARNING: there are {no_imgs} images present but only {no_rows_extracted} images were found in the existing csv file." )
        print("The follwing images present in the data folder were not found in CSV file, please check images:")
        print(not_present)    
    
    end_time = time.time()
    print(f'Total runtime: {end_time-start_time}s')
    
if __name__ == '__main__':
    main()
    
    
    
# CHECKS
# need to make sure the images actually go into given point cloud
# can we automate this?
# e.g search a LAS and find the min and max gps time - if the image doesnt fall within this then we discard?
# for now assume a;l; images concide in the LAS



"""Projection Module: all functions related to projection."""
import math
import numpy as np
import laspy

def rotMat(roll, pitch, heading, mode='norm'):
    """Converts roll, pitch and yaw angles (degrees) into single rotation matrix.   COME BACK TO THIS =, NOT SURE WHETHER ANGLES ARE FLOATS OR INTS

    Args:
        roll (_type_): The roll angle in degrees.
        pitch (_type_): The pitch angle in degrees.
        heading (_type_): The heading or yaw angle in degrees.
        mode (str, optional): Mode to combine roll pitch and yaw matrices, norm= yaw_mat @ pitch_mat @ roll_mat. Defaults to 'norm'.

    Returns:
        numpy.ndarray: A single rotation matrix describing combined roll pitch and yaw in one rotation.
    """
    
    alpha = math.radians(heading)
    cosa = math.cos(alpha)
    sina = math.sin(alpha)
    
    beta  = math.radians(pitch)
    cosb = math.cos(beta)
    sinb = math.sin(beta)
    
    gamma = math.radians(roll)
    cosg = math.cos(gamma)
    sing = math.sin(gamma)
    
    yaw_mat = np.array([[cosa , -sina , 0],
                        [sina, cosa, 0],
                        [0, 0, 1]])
    
    pitch_mat = np.array([[cosb, 0, sinb],
                          [0, 1, 0],
                          [-sinb, 0, cosb]])
    
    roll_mat = np.array([[1, 0, 0],
                         [0, cosg, -sing],
                         [0, sing, cosg]])
    if mode =='norm':
        rotmat = yaw_mat @ pitch_mat @ roll_mat
    elif mode == 'rev':
        rotmat = roll_mat @ pitch_mat @ yaw_mat
    else:
        print("error in mode")
    
    return rotmat



def extrinsicsMat(rph_list, xyz_list, error_correction): # orientation and position matrix of CCS rel to WCS
    """Creates extrinsics matrix converting a point in the world coord system to camera coord system.

    Args:
        rph_list (list): A list of three angles in degrees (floats) in the form [roll, pitch, heading].
        xyz_list (_type_): A list of coordinates in form [x,y,z].
        error_correction (_type_): A list of error correction angles to add on in degrees (floats) in form [roll_ec, pitch_ec, heading_ec].

    Returns:
        numpy.ndarray: A 4x4 extrinsics matrix.
    """

    wcs2ccs = np.row_stack((np.column_stack((rotMat(180,0,0) ,np.array([[0],[0],[0]]))), np.array([0,0,0,1])))  # WCS 2 CCS
    convert2projangles = [90+rph_list[1],rph_list[0],-rph_list[2]] # camera orientation 
    t = np.swapaxes(np.array([xyz_list]),0,1) # in [[x],[y],[z]] rather than [x,y,z]
    A, B, C = np.add(convert2projangles,error_correction)
    R = rotMat(A,B,C) # create rot mat from euler angles of camera roatation wrt CCS
    extrinsics_mat = (np.row_stack((np.column_stack((R,t)), np.array([0,0,0,1])))) @ wcs2ccs # roatation of WCS wrt image CS
    return extrinsics_mat



def intrinsicsMat():
    """Creates camera intrinsics matrix from camera specs.

    Returns:
        numpy.ndarray: A 3x3 intrinsics matrix. 
    """
    focal_length_mm = 8.5 # mm
    CCDW = 2464 # pixels, CCD (image) width 
    CCDH = 2056 # pixels, CCD (image) height
    Xpp = -0.00558 # mm, principle point x coord
    Ypp = 0.14785  # mm, principle point y coord
    Psx = 0.00345 # mm, width of pixel
    Psy = 0.00345 # mm, height of pixel
    cx = Xpp/Psx # optical centre x coord in pixels
    cy = Ypp/Psy # optical centre y coord in pixels
    fx = focal_length_mm/Psx # focal length in pixels x dirn
    fy = focal_length_mm/Psy # focal length in pixels y dirn
    
    K = np.array([[fx, 0, cx+(CCDW/2)],
                [0, fy, cy+(CCDH/2)],
                [0, 0, 1]])    
    return K





def projection_WCS2PCS(csv_row, points, img_np, las, error_correct):
    
    """Creates rgbd depth map and LAS point map of LAS points that coincide with the images pixels.

    Returns:
        numpy.ndarray: rgbd_map, LAS_map and projected img.
    """
    
    # create extrinsics and intrinsics matrix
    rph = [float(csv_row['roll[deg]']), float(csv_row['pitch[deg]']), float(csv_row['heading[deg]'])] # [roll, pitch, heading]
    xyz = [float(csv_row['projectedX[m]']),float(csv_row['projectedY[m]']),float(csv_row['projectedZ[m]'])] # [x,y,z]
    extr_mat = extrinsicsMat(rph,xyz,error_correct)
    intr_mat = np.hstack((intrinsicsMat(),np.array([[0.],[0.],[0.]])))
    
    # convert image to numpy and get size
    map_height = img_np.shape[0]
    map_width = img_np.shape[1]
        
    x_offset, y_offset, z_offset = las.header.offsets
    x_scale, y_scale, z_scale = las.header.scales
        
    rgbd_map = np.zeros(shape=(map_height, map_width,4), dtype=np.uint8) # each element at coord x,y has array [r,g,b,depth from cam cneter] with rgb in uint8 (normal), is rgb image wtih depth value for each coloured pixel
    LAS_data_map = np.zeros(shape=(map_height, map_width,6), dtype='uint32') # stores [x_wc,y_wc,z_wc,r_wc_uint16,g_wc_uint16,b_wc_uint16] so all xyz and rgb of real world points but in LAS format that are captured in img
    LAS_new = []
    
    # convert points from WCS to pixel coord system, each column is a uniwque point
    points_ccs_hg = np.linalg.inv(extr_mat) @ points[0:4, :] # rows x,y,z,1 and then every col i.e. every point in las pc. We now have homogeneuos coords in camera coord system, not world coord system
    points_pix_cs_hg = intr_mat @  points_ccs_hg # converts homogeneous world points CCS into homogeneous pixel coord points, inv exmat because need wcs rel2 ccs, but exmat is ccs rel2 wcs
    uv_rows = points_pix_cs_hg[0:2, :].reshape((2, points.shape[1])) # just takes the first two rows of projected_pts_homo (x/s, y/s) and makes new matrix
    w_row = points_pix_cs_hg[2, :].reshape((1, points.shape[1])) # creates 1D row vector of the s values (homogeneous last row of pix cord sustem points)
    w_points = uv_rows / w_row # all LAS points in pixel coord system (non homogeneous) (2 x N array)

    for pt_pix, pt_cam, pt_wcs in zip(w_points.transpose(),
                                   points_ccs_hg.transpose(),
                                   points.transpose()):
        # this for loop goes row by row, i.e. each points data by point 
        
        # maybe here further disect u,v = ptpix etc
        if 0 < pt_pix[0] < map_width and 0 < pt_pix[1] < map_height and pt_cam[2]>=0 : # filter for projected points within the actual map boundary, and if Z_ccs >0 
            
            discretised_pt_pix = (int(pt_pix[0]), int(pt_pix[1])) # we cannot have decimals of pixels, so turn (x_proj_pixcs, y_proj_pixcs)  into ints 
            
            # need to convert the valid points back into LAS compatible format (uint32)
            x_wc_uint32 = (pt_wcs[0] - x_offset)/x_scale
            y_wc_uint32 = (pt_wcs[1] - y_offset)/y_scale
            z_wc_uint32 = (pt_wcs[2] - z_offset)/z_scale

            b_wc_uint8 = pt_wcs[4]
            g_wc_uint8 = pt_wcs[5]
            r_wc_uint8 = pt_wcs[6]
            greyscale = pt_wcs[7]

            # convert to uint16 instead of uint8 for LAS format
            b_wc_uint16 = b_wc_uint8*256
            g_wc_uint16 = g_wc_uint8*256
            r_wc_uint16 = r_wc_uint8*256

            # Colour depth map (rgbd) - projected points onto black plain image with the correct colours and distance from cam center 
            existing_depth = rgbd_map[discretised_pt_pix[1], discretised_pt_pix[0]][3]
            depth = np.sqrt(np.sum(np.square(pt_cam[:3]))) # depth from the camera centre, as these pts are defined in cam coord system. 
            
            # these two do the excat same thing if else so combine them!
            if existing_depth == 0:
                LAS_new.append([r_wc_uint8,g_wc_uint8,b_wc_uint8,depth]) # REDUNDANT?
                rgbd_map[discretised_pt_pix[1], discretised_pt_pix[0]] = [r_wc_uint8,g_wc_uint8,b_wc_uint8,depth] # if using pil - RGB in arrays
                LAS_data_map[discretised_pt_pix[1], discretised_pt_pix[0]] = [x_wc_uint32,y_wc_uint32,z_wc_uint32,r_wc_uint16,g_wc_uint16,b_wc_uint16]

            elif depth < existing_depth: # we want the pt closest to the cam i.e. highest up so (lowest depth) to be the visible one
                LAS_new.append([r_wc_uint8,g_wc_uint8,b_wc_uint8,depth])
                rgbd_map[discretised_pt_pix[1], discretised_pt_pix[0]] = [r_wc_uint8,g_wc_uint8,b_wc_uint8,depth]
                LAS_data_map[discretised_pt_pix[1], discretised_pt_pix[0]] = [x_wc_uint32,y_wc_uint32,z_wc_uint32,r_wc_uint16,g_wc_uint16,b_wc_uint16]
            
    projected_img = rgbd_map[:,:,:3] # same as rgbd_map, but removed depth value from all the arrays, so just an image
    # LAS_data_map : at every u,v of the u*v 2D map, there is a third array that conatins [x_wc,y_wc,z_wc,r_wc,g_wc,b_wc] so every correpsonging LAS world point that coincides at this pixel point
    # rgbd map : every las point's rgb + depth data in uint8 so visisble, located as a array at every [u,v] pixel coord on map 
    # projected_img : same as rgbd just rempve depth so we have a visible image
    return rgbd_map, projected_img, LAS_data_map
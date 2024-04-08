# densify-pc

The purpose of this project is to utilise the high resolution detail from 2D road images and the depth information from 3D LiDAR point cloud data of road networks to create a denser point cloud representation of the road.

- Currently, 3D point clouds of road networks provided by National Highways do not capture the presence road defects due to the low resolutions of the LiDAR scans obtained.
- This makes road maintenence very inefficient - engineers have to scroll through the sequence of images taken of road surfaces via surveying vehicles one by one and try to figure out where on the point cloud corresponds to the image location, to then be able to locate where a defect is with respect to the wider road network shown in the point cloud.
- My algorithm aims to solve this inefficiency - by using camera-lidar fusion to map the images onto the point cloud and register depth infomation, a densified point cloud of the road captured in the image can be created.
- This means engineers can have all the infomation from both data types in one data form of the 3D point cloud.

Running the File:
- Install the required dependencies in the pyproject.toml file and run main.py
- Output is a LAS point cloud file found in results folder.
- You will need a point cloud viewer to view the result - recommended is CloudCompare.
- Only one image is included for processing for demonstration purposes, but multiple images can be processed in one run.
- Idea is to process mutiple images taken in a sequence to produce a whole section of road in point cloud.
- Overlay the densified point cloud result and the orginal point cloud file to view comparison.
- Point cloud data is LARGE so it will take a minute or so to process each point cloud file (only one provided).
- Data is confidential so please do not share - belongs to National Highways and the University of Cambridge.


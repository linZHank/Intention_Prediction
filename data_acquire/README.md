**Make sure 'pitchdatafiles_2d9blk.txt' is in LOCALDIR**

# For pitch data recording:
    1. Make sure the system time is correct
    2. run "record_pitch_data.m" to obtain data using Kinect V2
    3. run "arrangeNewImages\(num_test = 1 or 0\)" to export newly recorded 
        images to Google Drive

# For data re-computing
    1. run "ExtractJointData_all.m" to extract joint data all over again, 
        and save to LOCALDIR
    2. run "ExportImageData_all.m" to extract croped 
        and resized color and depth image data all over again, and save to 
        LOCALDIR
    3. run function "prepareImagesAll.m" to copy color and depth images to
        Google Drive. Note: delete existed files in Google Drive.
    4. run function "prepareJointData.m" to copy correct joint data files 
        ro Google Drive. Note: delete existed files in Google Drive.
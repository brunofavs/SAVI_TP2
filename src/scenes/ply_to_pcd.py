#!/usr/bin/env python3
# Convert dataset ply to pcd files

import open3d as o3d
import glob
import os


def main():

    # --------------------------------------
    # Initialization
    # --------------------------------------

    
    # TODO fix naming
    dataset_path = os.getenv('SAVI_TP2') + '/dataset/scenes_dataset_v2/rgbd-scenes-v2_pc/rgbd-scenes-v2/pc'
    # dataset_path = os.getenv('SAVI_TP2') + '/dataset/rgbd-scenes-v2_pc/rgbd-scenes-v2/'
    
    ply_path = glob.glob(dataset_path + '/*.ply')
    print('Starting convertion on path: ' + dataset_path)
    print('Found '+ str(len(ply_path)) +' files.')

    # Check destination path if already exist
    if not os.path.exists(dataset_path + '/pcd'):
        os.mkdir(dataset_path + '/pcd')



    # --------------------------------------
    # Execution
    # --------------------------------------
        
    for file in ply_path:
        file_name = file.split('/')
        file_name = file_name[-1]
        file_name = file_name[:-4]
        
        #TODO add to readme Needs sudo apt install pcl-tools
        os.system('pcl_ply2pcd ' + file + ' ' + dataset_path + '/pcd/' + file_name + '.pcd')
   

if __name__ == "__main__":
    main()
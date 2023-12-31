#!/usr/bin/env python3
# Convert dataset ply files to pcd
# Mário Vasconcelos 2023

import open3d as o3d
import glob
import os


def main():

    # --------------------------------------
    # Initialization
    # --------------------------------------

    dataset_path = '/home/mario/Desktop/SAVI/Dataset/scenes'
    
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
        os.system('pcl_ply2pcd ' + file + ' ' + dataset_path + '/pcd/' + file_name + '.pcd')
   

if __name__ == "__main__":
    main()
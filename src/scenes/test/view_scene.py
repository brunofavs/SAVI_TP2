#!/usr/bin/env python3
# View original scene
# Mário Vasconcelos 2023

import os
import open3d as o3d
import numpy as np
import glob

view = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 2.7116048336029053, 1.2182252407073975, 3.8905272483825684 ],
			"boundingbox_min" : [ -2.4257750511169434, -1.6397310495376587, -1.3339539766311646 ],
			"field_of_view" : 60.0,
			"front" : [ -0.33682983603152233, -0.40052927348606165, -0.85212790274682682 ],
			"lookat" : [ 0.26085641706715251, 0.95360623106515774, 2.734619924963821 ],
			"up" : [ 0.4644300399545937, -0.85793168430988997, 0.21967695155607475 ],
			"zoom" : 0.68120000000000025
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}

def main():

    # --------------------------------------
    # Initialization
    # --------------------------------------

    dataset_path = os.getenv('SAVI_TP2')+ '/dataset'
    # scenes_paths = glob.glob(dataset_path + '/rgbd_scenes_v2/pcd/*.pcd')
    scenes_path = dataset_path +'/scenes_dataset_v2/rgbd-scenes-v2_pc/rgbd-scenes-v2/pc/pcd' 
    scenes_paths = glob.glob(scenes_path + '/*.pcd')

    
    # Print available scenes
    available_scenes = []
    for scenes in scenes_paths:
        file_name = scenes.split('/')
        file_name = file_name[-1]
        available_scenes.append(file_name)
    print(available_scenes)

    # User input scene select
    print("File format XX.pcd")
    scene_n = input("Scene number: ")


    filename = f'{scenes_path}/{scene_n}.pcd'
    print('Loading file '+ filename)
    ptCloud = o3d.io.read_point_cloud(filename)

    print(ptCloud)
    print('Points array')
    print(np.asarray(ptCloud.points))
    print('Color array')
    print(np.asarray(ptCloud.colors))


    # --------------------------------------
    # Execution
    # --------------------------------------

    # --------------------------------------
    # Visualizations
    # --------------------------------------
    entities = [ptCloud]
    o3d.visualization.draw_geometries(entities,
                                      zoom=view['trajectory'][0]['zoom'],
                                      front=view['trajectory'][0]['front'],
                                      lookat=view['trajectory'][0]['lookat'],
                                      up=view['trajectory'][0]['up'])
    
if __name__ == "__main__":
    main()
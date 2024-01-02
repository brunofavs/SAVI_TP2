#!/usr/bin/env python3
# View labeled scene
# Mário Vasconcelos 2023

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

    dataset_path = '/home/mario/Desktop/SAVI/Dataset/'
    scenes_path = glob.glob(dataset_path + 'scenes/pcd/*.pcd')
    
    # Print available scenes
    available_scenes = []
    for scenes in scenes_path:
        file_name = scenes.split('/')
        file_name = file_name[-1]
        available_scenes.append(file_name)
    print(available_scenes)

    # User input scene select
    # scene_n = input("Scene number: ")
    scene_n = "01"

    print('--------- Scene Properties --------- ')
    filename = dataset_path + 'scenes/pcd/'+ scene_n + ".pcd"
    print('Loading file '+ filename)
    ptCloud = o3d.io.read_point_cloud(filename)
    print(ptCloud)

    # --------------------------------------
    # Labeling
    # --------------------------------------
    # bowl=1, cap=2, cereal_box=3, coffee_mug=4, coffee_table=5 
    # office_chair=6, soda_can=7, sofa=8, table=9, background=10
    object_n = "4"

    # Open and read labels file
    f = open(dataset_path + 'scenes/'+ scene_n + ".label",'r')
    labels = f.read().splitlines()
    labels.pop(0) # Remove first item of label list
    
    # Calculate index of labeled object
    ptcloud_ori = np.asarray(ptCloud.points)
    inliers_idx = []
    outliers_idx = []
    for idx in range(len(ptcloud_ori)):
        if labels[idx] == object_n:
            inliers_idx.append(idx)
        else:
            outliers_idx.append(idx)

    print('--------- Object Properties --------- ')
    print("Object points: " + str(len(inliers_idx)))   
    object_cloud = ptCloud.select_by_index(inliers_idx)

    # --------------------------------------
    # Calculate Object Cloud properties
    # --------------------------------------
    
    # Average Color
    cloud_colors = np.asarray(object_cloud.colors)
    average_color  = np.mean(cloud_colors, axis=0)
    print('Average Color: ' + str(average_color))

    # Dimensions
    min_bound = object_cloud.get_min_bound()
    max_bound = object_cloud.get_max_bound()
    dimensions = max_bound - min_bound
    print('Dimentions:'+ str(dimensions))

    # Bounding Box
    axis_aligned_bounding_box = object_cloud.get_axis_aligned_bounding_box()
    axis_aligned_bounding_box.color = (1.0,0,0)

    oriented_bounding_box = object_cloud.get_oriented_bounding_box()
    oriented_bounding_box.color = (0,1,0)

    # --------------------------------------
    # Visualizations
    # --------------------------------------
    entities = [ptCloud, axis_aligned_bounding_box]
    # entities = [object_cloud]
    o3d.visualization.draw_geometries(entities, 
                                      zoom=view['trajectory'][0]['zoom'],
                                      front=view['trajectory'][0]['front'],
                                      lookat=view['trajectory'][0]['lookat'],
                                      up=view['trajectory'][0]['up'])
    f.close()
if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# View labeled scene

import open3d as o3d
import numpy as np
import glob
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import cm
from more_itertools import locate

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

class PlaneDetection():
    def __init__(self, point_cloud):

        self.point_cloud = point_cloud

    def colorizeInliers(self, r,g,b):
        self.inlier_cloud.paint_uniform_color([r,g,b]) # paints the plane in red

    def segment(self, distance_threshold=0.04, ransac_n=3, num_iterations=100):

        print('Starting plane detection')
        plane_model, inlier_idxs = self.point_cloud.segment_plane(distance_threshold=distance_threshold, 
                                                    ransac_n=ransac_n,
                                                    num_iterations=num_iterations)
        [self.a, self.b, self.c, self.d] = plane_model

        self.inlier_cloud = self.point_cloud.select_by_index(inlier_idxs)

        outlier_cloud = self.point_cloud.select_by_index(inlier_idxs, invert=True)

        return outlier_cloud

    def __str__(self):
        text = 'Segmented plane from pc with ' + str(len(self.point_cloud.points)) + ' with ' + str(len(self.inlier_cloud.points)) + ' inliers. '
        text += '\nPlane: ' + str(self.a) +  ' x + ' + str(self.b) + ' y + ' + str(self.c) + ' z + ' + str(self.d) + ' = 0' 
        return text
    
def main():

    # --------------------------------------
    # Initialization
    # --------------------------------------

    dataset_path = '../../dataset'
    scenes_path = glob.glob(dataset_path + '/rgbd_scenes_v2/pcd/*.pcd')
    
    # Print available scenes
    available_scenes = []
    for scenes in scenes_path:
        file_name = scenes.split('/')
        file_name = file_name[-1]
        available_scenes.append(file_name)
    print('Available scenes: ' + str(available_scenes))

    # User input scene select
    # scene_n = input("Scene number: ")
    scene_n = "07"

    print('--------- Scene Properties --------- ')
    filename = dataset_path + '/rgbd_scenes_v2/pcd/'+ scene_n + ".pcd"
    print('Loading file '+ filename)
    ptCloud_ori = o3d.io.read_point_cloud(filename)
    print(ptCloud_ori)

    # Downsample scene
    ptCloud_downsampled = ptCloud_ori.voxel_down_sample(voxel_size=0.01) 
    print('After downsampling: ' + str(len(ptCloud_downsampled.points)) + ' points')


    # ------------------------------------------
    # Execution
    # ------------------------------------------

    # Plane detection parameters
    number_of_planes = 2 # Number of planes to detect
    minimum_number_points = 25
    colormap = cm.Pastel1(list(range(0,number_of_planes)))

    ptcloud = deepcopy(ptCloud_downsampled)
    planes = []
    while True: 

        # Create point cloud with plane outliers for next itteration
        plane = PlaneDetection(ptcloud) # New plane instance
        ptcloud = plane.segment()       # New point cloud with outliers
        print(plane)

        # Pick Plane color
        idx_color = len(planes)
        color = colormap[idx_color, 0:3]
        plane.colorizeInliers(r=color[0], g=color[1], b=color[2])
        planes.append(plane)

        # Stop while loop when:
        if len(planes) >= number_of_planes: # stop detection planes
            print('Detected planes >= ' + str(number_of_planes))
            break
        elif len(ptcloud.points) < minimum_number_points:
            print('Number of remaining points < ' + str(minimum_number_points))
            break

    # Table plane detector
    # (This method uses the average height of the plane)
    table_plane = None
    table_plane_mean_y = 1000
    for plane_idx, plane in enumerate(planes):
        center = plane.inlier_cloud.get_center()
        print('Cloud ' + str(plane_idx) + ' has center ' + str(center))
        mean_y = center[1]

        if mean_y < table_plane_mean_y:
            table_plane = plane
            table_plane_mean_y = mean_y

    table_plane.colorizeInliers(r=1, g=0, b=0) # Force plane table to be red

    # # Cluster extraction
    # cluster_idxs = list(table_plane.inlier_cloud.cluster_dbscan(eps=0.15, min_points=25, print_progress=True))

    # # print(cluster_idxs)
    # # print(type(cluster_idxs))

    # # -1 means noise
    # possible_values = list(set(cluster_idxs))
    # if -1 in possible_values:
    #     possible_values.remove(-1)
    # print(possible_values)

    # largest_cluster_num_points = 0
    # largest_cluster_idx = None
    # for value in possible_values:
    #     num_points = cluster_idxs.count(value)
    #     if num_points > largest_cluster_num_points:
    #         largest_cluster_idx = value
    #         largest_cluster_num_points = num_points

    # largest_idxs = list(locate(cluster_idxs, lambda x: x == largest_cluster_idx))
    # table_cloud = table_plane.inlier_cloud.select_by_index(largest_idxs)
    # table_cloud.paint_uniform_color([0,1,0]) # paints the table green


    # --------------------------------------
    # Visualizations
    # --------------------------------------
    entities = [ptCloud_downsampled]
    entities.append(table_plane.inlier_cloud)
    # entities.append(table_cloud)
    # entities.append(planes[0].inlier_cloud)
    # entities.append(planes[1].inlier_cloud)

    # entities = [object_cloud]
    o3d.visualization.draw_geometries(entities, 
                                      zoom   =view['trajectory'][0]['zoom'],
                                      front  =view['trajectory'][0]['front'],
                                      lookat =view['trajectory'][0]['lookat'],
                                      up     =view['trajectory'][0]['up'])

if __name__ == "__main__":
    main()
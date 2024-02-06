#!/usr/bin/env python3
# View labeled scene

import open3d as o3d
import numpy as np
import glob
from copy import deepcopy
import math
from more_itertools import locate
from matplotlib import cm
import os
import cv2
import json

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
			"front" : [ -0.86797159921609002, -0.27893672293557875, -0.41087663300828348 ],
			"lookat" : [ -0.13711904974393682, -0.307851942294602, 1.4581324743200366 ],
			"up" : [ 0.39786518017848138, -0.88571554082741299, -0.23918879392301828 ],
			"zoom" : 0.29999999999999999
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}

class PointCloudOperations():


    def __init__(self,pcd):
        self.original_pcd = pcd
        self.gui_pcd = deepcopy(self.original_pcd)
        self.verticality_threshold = 0.05
        
        print('Original PCD has: ' + str(len(self.gui_pcd.points)) + ' points')

    def pre_process(self,voxelsize):
        self.voxelize(voxelsize)
        self.estimateNormals()
    
    def removeStatisticalNoise(self,neighbours_factor = 0.01,agressiveness = 2):

        print("Removing noise")
        total_num_points = len(self.gui_pcd.points)
        _,idxs = self.gui_pcd.remove_statistical_outlier(nb_neighbors = int(0.01*total_num_points), std_ratio = agressiveness)
        # print(idxs)
        self.gui_pcd.select_by_index(idxs)

    def removeRadiallNoise(self,neighbours_factor = 0.10,radius = 0.5):

        print("Removing noise")
        total_num_points = len(self.gui_pcd.points)

        _,idxs = self.gui_pcd.remove_radius_outlier(nb_points = 5, radius = radius)
        # print(idxs)
        self.gui_pcd.select_by_index(idxs)

    def voxelize(self,voxel_size):
        # Downsample scene
        self.gui_pcd = self.gui_pcd.voxel_down_sample(voxel_size= voxel_size)  
        print('After downsampling: ' + str(len(self.gui_pcd.points)) + ' points')

    def estimateNormals(self):
        
        # Estimate normals
        self.gui_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        self.gui_pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, 1]))
    
    def computeVerticalNormals(self):

        # Select voxels idx that are 90º degrees with X Camera Axis
        self.vertical_normals_idxs = []
        for idx,normal_vector in enumerate(self.gui_pcd.normals):

            # Compute angle between two 3d vectors
            norm_normal = np.linalg.norm(normal_vector)

            x_axis = [1, 0, 0] # Compare with X axis
            norm_comp_axis = np.linalg.norm(x_axis)

            theta = math.acos(np.dot(normal_vector, x_axis) / (norm_normal * norm_comp_axis))

            # Keep points where angle to z_axis is small enough
            if abs(theta - math.pi/2) < self.verticality_threshold:  # we have a point that belongs to an horizontal surface
                self.vertical_normals_idxs.append(idx)

    def computeAngle(self,reference_vector = [0,0,1]):

        thetas = []
        for normal_vector in self.gui_pcd.normals:
            # Compute angle between two 3d vectors
            norm_normal = np.linalg.norm(normal_vector)

            norm_reference_vector = np.linalg.norm(reference_vector)

            theta = -(math.pi/2 - math.acos(np.dot(normal_vector, reference_vector) / (norm_normal * norm_reference_vector)))
            thetas.append(theta)
        
        avg_theta = sum(thetas)/len(thetas)
        print(f'Computed angle is : {avg_theta * 180/math.pi:.2f}º')

        return avg_theta



    def transgeom(self,rotation = np.array([0,0,0]),translation = np.array([0,0,0])):
        
        # Rotate 
        rot = self.gui_pcd.get_rotation_matrix_from_xyz(rotation)
        self.gui_pcd.rotate(rot, center=(0, 0, 0,))

        # Translate
        self.gui_pcd.translate(translation)

    def cropPcd(self,min_bound,max_bound):

        crop_box = [min_bound,max_bound]
    
        # Create a bounding box using the provided crop_box
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=crop_box[0], max_bound=crop_box[1])

        # Crop point cloud
        self.gui_pcd = self.gui_pcd.crop(bounding_box)

    def filterByIdx(self,idxs = None):
        if idxs is None:
            self.gui_pcd = self.gui_pcd.select_by_index(self.vertical_normals_idxs)
        else:
            self.gui_pcd = self.gui_pcd.select_by_index(idxs)
        
    def filterBiggestCluster(self,eps = 0.045,min_points = 50):

        print("Finding biggest cluster")
        idxs = self.gui_pcd.cluster_dbscan(eps=eps, min_points=50, print_progress=True)
        dominant_idx = mostCommon(idxs)

        dominant_idxs = [index for index,value in enumerate(idxs) if value == dominant_idx]

        self.filterByIdx(dominant_idxs) 

    def getDiameter(self):

        points = np.asarray(self.gui_pcd.points)

        # * Here I'm not even going to lie, it's some ChatGPT Wizary LMAO
        # Compute pairwise distances between all points
        distances = np.linalg.norm(points[:, None] - points, axis=-1)

        # Find the maximum distance
        diameter = np.max(distances)
        print(f"Diameter: {diameter:.2f}")

        return diameter

    def computeAvgNearestNeighborDistance(self):

        print(f'Computing Average Nearest Neighbour Distance')

        # Compute nearest neighbor for each point
        distances = self.gui_pcd.compute_nearest_neighbor_distance()

        avg_distance = np.mean(distances)

        return avg_distance

    def segment(self, distance_threshold=0.03, ransac_n=3, num_iterations=200 ,outliers = True):

        print('Starting plane detection')
        _, inlier_idxs = self.gui_pcd.segment_plane(distance_threshold=distance_threshold, 
                                                    ransac_n=ransac_n,
                                                    num_iterations=num_iterations)


        self.gui_pcd = self.gui_pcd.select_by_index(inlier_idxs, invert=outliers)

    def view(self,entities = None,seixos_on = True):
        entities_to_draw = [self.gui_pcd] if entities is None else [entities]
        if seixos_on:
            seixos = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))
            entities_to_draw.append(seixos)

        o3d.visualization.draw_geometries(entities_to_draw, 
                                          zoom   =view['trajectory'][0]['zoom'],
                                          front  =view['trajectory'][0]['front'],
                                          lookat =view['trajectory'][0]['lookat'],
                                          up     =view['trajectory'][0]['up'])
   
def mostCommon(lst):
    # Converts to set and then tests for each unique element the return value of lst.count(element)
    # Without converting to set would still work but would be a lot less efficient
    return max(set(lst), key=lst.count)


def main():

    # Script parameters
    scene_n = "01"
    dataset_path = f'{os.getenv("SAVI_TP2")}/dataset'
    
    #Load Camera Intrinsics
    with open("./lib/jsons/intrinsic.json",'r') as f:
        intrinsics_matrix = np.asarray(json.load(f))

    #Load scene pointcloud
    # TODO maybe it's not always the best to choose the image 0000
    img_path   = dataset_path + f'/scenes_dataset_v2/rgbd-scenes-v2_pc/rgbd-scenes-v2/imgs/scene_{scene_n}/00000-color.png'
    scene_path = dataset_path + f'/scenes_dataset_v2/rgbd-scenes-v2_pc/rgbd-scenes-v2/pc/pcd/{scene_n}.pcd' 
    label_path = dataset_path + f'/scenes_dataset_v2/rgbd-scenes-v2_pc/rgbd-scenes-v2/pc//{scene_n}.label'


    original_scene_pcd = o3d.io.read_point_cloud(scene_path)
    scene_pre_processed_checkpoints = []

    # ? Segmenting other scenes doesn't work as well
    #* 1º Step - Find table center

    scene_operations = PointCloudOperations(original_scene_pcd)
    scene_operations.pre_process(voxelsize=0.004)
    
    # Filter horizontal planes
    scene_operations.estimateNormals()
    scene_operations.computeVerticalNormals()
    scene_operations.filterByIdx()

    scene_pre_processed_checkpoints.append(scene_operations.gui_pcd)

    scene_operations.segment(outliers=False)

    
    # Voxelize further because only need to extract the rough diameter and rough center
    # And before clustering because that operation is very heavy
    scene_operations.voxelize(voxel_size=0.04)
    
    # This guarantees the center is still the table if the table isnt centered and there are a lot more blobs
    scene_operations.filterBiggestCluster(eps = 0.3)

    camera2table_center = scene_operations.gui_pcd.get_center()
    table_center2camera = camera2table_center * -1

    scene_operations.transgeom(translation=table_center2camera)
    scene_operations.view()

    #* 2º Step - Find table size to crop and angle to 
    
    # Eps relies heavily on the density and gives bad results if its too small
    table_scene_operations = scene_operations
    table_scene_operations.filterBiggestCluster(eps = 0.3)    
    table_scene_operations.removeStatisticalNoise()
    # Find diameter of table

    diameter = table_scene_operations.getDiameter()
    radius = diameter/2

    # Need to recompute the normals because now they are fewer, it's not the most elegant solution though
    table_scene_operations.estimateNormals()
    z_offset_about_x = table_scene_operations.computeAngle(reference_vector=[0,0,1])


    #* 3º Step - Find object clusters and their centroids

    # Reseting scene operations to the original scene pre-processed (without over-voxelization / cropping)
    # scene_operations.gui_pcd = scene_pre_processed_checkpoints[-1]
    scene_operations.gui_pcd = original_scene_pcd
    
    # ! Cannot do translation and rotation at the same time, this is a bug
    # scene_operations.transgeom(rotation=np.array([z_offset_about_x,0,0]),translation = table_center2camera)
    scene_operations.transgeom(translation = table_center2camera)
    scene_operations.transgeom(rotation= np.array([z_offset_about_x,0,0]))

    scene_operations.cropPcd(np.array([-radius,-0.5,-radius]),np.array([radius,-0.025,radius*0.7]))
    
    # avg_neighbour_distance = scene_operations.computeAvgNearestNeighborDistance()







if __name__ == "__main__":
    main()
#!/usr/bin/env python3


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

def mostCommon(lst):
    # Converts to set and then tests for each unique element the return value of lst.count(element)
    # Without converting to set would still work but would be a lot less efficient
    return max(set(lst), key=lst.count)


class PointCloudOperations:

    def __init__(self, pcd,perspective = None):
        self.original_pcd = pcd
        self.gui_pcd = deepcopy(self.original_pcd)
        self.verticality_threshold = 0.05
        self.perspective = perspective

        print("Original PCD has: " + str(len(self.gui_pcd.points)) + " points")

    def pre_process(self, voxelsize):
        self.voxelize(voxelsize)
        self.estimateNormals()

    def removeStatisticalNoise(self, neighbours_factor=0.01, agressiveness=2):

        print("Removing noise")
        total_num_points = len(self.gui_pcd.points)
        _, idxs = self.gui_pcd.remove_statistical_outlier(
            nb_neighbors=int(0.01 * total_num_points), std_ratio=agressiveness
        )
        # print(idxs)
        self.gui_pcd.select_by_index(idxs)

    def removeRadiallNoise(self, neighbours_factor=0.10, radius=0.5):

        print("Removing noise")
        total_num_points = len(self.gui_pcd.points)

        _, idxs = self.gui_pcd.remove_radius_outlier(nb_points=5, radius=radius)
        # print(idxs)
        self.gui_pcd.select_by_index(idxs)

    def voxelize(self, voxel_size):
        # Downsample scene
        self.gui_pcd = self.gui_pcd.voxel_down_sample(voxel_size=voxel_size)
        print("After downsampling: " + str(len(self.gui_pcd.points)) + " points")

    def estimateNormals(self):

        # Estimate normals
        self.gui_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        self.gui_pcd.orient_normals_to_align_with_direction(
            orientation_reference=np.array([0, 0, 1])
        )

    def computeVerticalNormals(self):

        # Select voxels idx that are 90º degrees with X Camera Axis
        self.vertical_normals_idxs = []
        for idx, normal_vector in enumerate(self.gui_pcd.normals):

            # Compute angle between two 3d vectors
            norm_normal = np.linalg.norm(normal_vector)

            x_axis = [1, 0, 0]  # Compare with X axis
            norm_comp_axis = np.linalg.norm(x_axis)

            theta = math.acos(
                np.dot(normal_vector, x_axis) / (norm_normal * norm_comp_axis)
            )

            # Keep points where angle to z_axis is small enough
            if (
                abs(theta - math.pi / 2) < self.verticality_threshold
            ):  # we have a point that belongs to an horizontal surface
                self.vertical_normals_idxs.append(idx)

    def computeAngle(self, reference_vector=[0, 0, 1]):

        thetas = []
        for normal_vector in self.gui_pcd.normals:
            # Compute angle between two 3d vectors
            norm_normal = np.linalg.norm(normal_vector)

            norm_reference_vector = np.linalg.norm(reference_vector)

            theta = -(
                math.pi / 2
                - math.acos(
                    np.dot(normal_vector, reference_vector)
                    / (norm_normal * norm_reference_vector)
                )
            )
            thetas.append(theta)

        avg_theta = sum(thetas) / len(thetas)
        print(f"Computed angle is : {avg_theta * 180/math.pi:.2f}º")

        return avg_theta

    def transGeom(self, rotation=np.array([0, 0, 0]), translation=np.array([0, 0, 0])):

        # Rotate
        rot = self.gui_pcd.get_rotation_matrix_from_xyz(rotation)
        self.gui_pcd.rotate(
            rot,
            center=(
                0,
                0,
                0,
            ),
        )

        # Translate
        self.gui_pcd.translate(translation)

    def cropPcd(self, min_bound, max_bound):

        crop_box = [min_bound, max_bound]

        # Create a bounding box using the provided crop_box
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=crop_box[0], max_bound=crop_box[1]
        )

        # Crop point cloud
        self.gui_pcd = self.gui_pcd.crop(bounding_box)

    def filterByIdx(self, idxs=None, inPlace=True):
        if inPlace:
            if idxs is None:
                self.gui_pcd = self.gui_pcd.select_by_index(self.vertical_normals_idxs)
            else:
                self.gui_pcd = self.gui_pcd.select_by_index(idxs)
        else:
            filtered_pcd = self.gui_pcd.select_by_index(idxs)
            return filtered_pcd

    def filterBiggestCluster(self, eps=0.045, min_points=50):

        print("Finding biggest cluster")
        idxs = self.gui_pcd.cluster_dbscan(eps=eps, min_points=50, print_progress=True)
        dominant_idx = mostCommon(idxs)

        dominant_idxs = [
            index for index, value in enumerate(idxs) if value == dominant_idx
        ]

        self.filterByIdx(dominant_idxs)

    def computeClusters(self, eps=0.045, min_points=50):

        print("Computing clusters")
        self.cluster_idxs = list(
            self.gui_pcd.cluster_dbscan(eps=eps, min_points=50, print_progress=True)
        )
        cluster_idx_set = set(self.cluster_idxs)
        self.cluster_idxs = np.array(self.cluster_idxs)
        cluster_idx_set.discard(-1)  # Doesn't raise keyword if there is no noise

        object_point_clouds = []

        for cluster in cluster_idx_set:
            indexes_matching_cluster = list(np.where(self.cluster_idxs == cluster)[0])
            object_point_clouds.append(
                self.filterByIdx(idxs=indexes_matching_cluster, inPlace=False)
            )

        # print(object_point_clouds)
        return object_point_clouds

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

        print(f"Computing Average Nearest Neighbour Distance")

        # Compute nearest neighbor for each point
        distances = self.gui_pcd.compute_nearest_neighbor_distance()

        avg_distance = np.mean(distances)

        return avg_distance

    def segment(
        self, distance_threshold=0.03, ransac_n=3, num_iterations=200, outliers=True
    ):

        print("Starting plane detection")
        _, inlier_idxs = self.gui_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )

        self.gui_pcd = self.gui_pcd.select_by_index(inlier_idxs, invert=outliers)

    def associateGT(self, scene_labels,label_dict, original_pcd):
        
        distances = original_pcd.compute_point_cloud_distance(self.gui_pcd)  
        distances = np.asarray(distances)
        
        mean_distance = np.mean(distances)  # 0.24
        min_distance = np.min(distances)    # 0.16
        max_distance = np.max(distances)    # 0.34

        # print(mean_distance)
        # print(max_distance)
        # print(min_distance)
        # print(np.percentile(distances,1))   # 0.21

        object_scene_idxs = np.where(distances < min_distance*1.1)[0]


        obj_label = []
        for idx in object_scene_idxs:
            label = scene_labels[idx]
            obj_label.append(label)

        self.typeGT = label_dict[mostCommon(obj_label)]


    def view(self, entities=None, seixos_on=True,use_view = False):
        entities_to_draw = [self.gui_pcd] if entities is None else [entities]
        if seixos_on:
            seixos = o3d.geometry.TriangleMesh().create_coordinate_frame(
                size=0.5, origin=np.array([0.0, 0.0, 0.0])
            )
            entities_to_draw.append(seixos)

        if use_view and self.perspective is not None:
            o3d.visualization.draw_geometries(
                entities_to_draw,
                zoom=self.perspective["trajectory"][0]["zoom"],
                front=self.perspective["trajectory"][0]["front"],
                lookat=self.perspective["trajectory"][0]["lookat"],
                up=self.perspective["trajectory"][0]["up"],
            )
        else:
            o3d.visualization.draw_geometries(entities_to_draw)



    def savePCD(self, filename, path):
        cur_dir = os.getcwd()
        os.chdir(path)

        o3d.io.write_point_cloud(filename, self.gui_pcd)
    
    def computePcdBBox(self):

        bbox = self.gui_pcd.get_axis_aligned_bounding_box()

        # o3d.visualization.draw_geometries([self.gui_pcd,bbox])

        min_bound =   list(bbox.get_min_bound())
        max_bound  =  list(bbox.get_max_bound())

        self.axisAlignedBBox = {"min_bound":min_bound,"max_bound":max_bound}

    def computeRGB_projection(self,point,intrinsics_matrix,extrinsics_matrix = 0):
            # Check if variable is a list
        assert isinstance(point, list), "Variable is not a list"
        
        # Check if the list has exactly 3 elements
        assert len(point) == 3, "List does not contain exactly 3 elements"
        
        # Check if all elements in the list are numbers
        for element in point:
            assert isinstance(element, (int, float)), "Element {} is not a number".format(element)

        point = np.array(point)

        rgb_image_point,_ = cv2.projectPoints(point,np.zeros((3,1)),np.zeros((3,1)),intrinsics_matrix,np.zeros((5,1)))

        rgb_image_point = rgb_image_point.flatten()
        
        return rgb_image_point.astype(int)

    def computeRgbBbox(self,intrinsics_matrix):
        min_bound = self.computeRGB_projection(self.axisAlignedBBox["min_bound"],intrinsics_matrix)
        max_bound = self.computeRGB_projection(self.axisAlignedBBox["max_bound"],intrinsics_matrix)

        self.RgbBBox = {"min_bound":min_bound,"max_bound":max_bound}

    def transformVoxel2Frame(self,point,extrinsic_matrix,camera_extrinsics):
        pass

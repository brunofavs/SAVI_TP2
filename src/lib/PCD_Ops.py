#!/usr/bin/env python3


import webcolors
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

from lib.googleTtS import text2Speech

def mostCommon(lst):
    # Converts to set and then tests for each unique element the return value of lst.count(element)
    # Without converting to set would still work but would be a lot less efficient
    return max(set(lst), key=lst.count)

def quaternion_to_euler_matrix(q):
    rotation_matrix = np.zeros((3, 3))
    q0, q1, q2, q3 = q

    rotation_matrix[0, 0] = 1 - 2*(q2**2 + q3**2)
    rotation_matrix[0, 1] = 2*(q1*q2 - q0*q3)
    rotation_matrix[0, 2] = 2*(q1*q3 + q0*q2)

    rotation_matrix[1, 0] = 2*(q1*q2 + q0*q3)
    rotation_matrix[1, 1] = 1 - 2*(q1**2 + q3**2)
    rotation_matrix[1, 2] = 2*(q2*q3 - q0*q1)

    rotation_matrix[2, 0] = 2*(q1*q3 - q0*q2)
    rotation_matrix[2, 1] = 2*(q2*q3 + q0*q1)
    rotation_matrix[2, 2] = 1 - 2*(q1**2 + q2**2)

    return rotation_matrix

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name

class PointCloudOperations:

    def __init__(self, pcd,perspective = None):
        self.original_pcd = pcd
        self.gui_pcd = deepcopy(self.original_pcd)
        self.verticality_threshold = 0.05
        self.perspective = perspective
        self.type = None

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
        # cv2.rectangle(scene_gui_rgb,top_left_rgb_bbox, bottom_right_rgb_bbox,(0,0,255),3)
        # # cv2.imshow('Scene', scene_gui_rgb)
        # cv2.imshow("ROI",object_operations[number].Rgb_ROI)
        # cv2.waitKey(0)

    def describe(self):
        text = f'''This object is a {self.type}, and its dimensions are {self.dimensions}, and its color is {get_colour_name(self.average_color)} '''
        print(text)
        text2Speech(text)



        

    def computeProperties(self):

        cloud_colors = np.asarray(self.gui_pcd.colors)
        self.average_color  = np.round(np.mean(cloud_colors, axis=0),decimals=2)

        self.average_color = (self.average_color * 255)
        self.average_color.astype(int)

        print('Average Color: ' + str(self.average_color))

        # Dimensions
        min_bound = self.gui_pcd.get_min_bound()
        max_bound = self.gui_pcd.get_max_bound()
        self.dimensions = np.round((max_bound - min_bound),decimals=2)
        print('Dimentions:'+ str(self.dimensions))

    
    def computePcdBBox(self):

        bbox = self.gui_pcd.get_axis_aligned_bounding_box()

        # o3d.visualization.draw_geometries([self.gui_pcd,bbox])

        min_bound =   list(bbox.get_min_bound())
        max_bound  =  list(bbox.get_max_bound())

        self.axisAlignedBBox = {"min_bound":min_bound,"max_bound":max_bound}

    def computePcdCentroid(self):
        self.centroid_pcd = self.gui_pcd.get_center()

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
    
    def computeImages(self,img_paths,n_divs = 10):
        self.n_divs = n_divs
        
        n_images = len(img_paths)
        PointCloudOperations.rgb_images = []
        for img_path in img_paths[::round(n_images/n_divs)]:
            img = cv2.imread(img_path)
            PointCloudOperations.rgb_images.append(img)

    def computeRGBCentroid(self,img_paths, poses,intrinsics_matrix):

        n_images = len(img_paths)
        
        self.RGBCentroids = []
        for img_path in img_paths[::round(n_images/self.n_divs)]:
            img_numb =  int(img_path[-15:-10])
            pose = poses[img_numb][:]

            # Split line by " "
            pose = pose.split()
            # Convert to int
            pose = [float(i) for i in pose]

            # Pose matrix    
            WTP          = np.eye(4, dtype = float)
            WTP[3,3]     = 1
            WTP[0:3,0:3] = quaternion_to_euler_matrix(pose[0:4])
            WTP[0:3,3]   = pose[4:7]
            PTW = np.linalg.inv(WTP)

            centroid_w = self.centroid_pcd
            # Shape to homogenic coordinate
            centroid_pcd_p = np.append(centroid_w,1)
            centroid_pcd_p = centroid_pcd_p.reshape(-1,1)

            # Apply transformation to point
            centroid_pcd_p = np.dot(PTW,centroid_pcd_p)
            centroid_pcd_p = np.reshape(centroid_pcd_p, (1,4))
            centroid_pcd_p = centroid_pcd_p[0][:-1]   

            rgbcentroid = self.computeRGB_projection(list(centroid_pcd_p),intrinsics_matrix)    
            self.RGBCentroids.append(rgbcentroid)


    def computeRgbBboxs(self,img_paths, poses,intrinsics_matrix):

        n_images = len(img_paths)
        
        self.RgbBBoxs = []
        for img_path in img_paths[::round(n_images/self.n_divs)]:
            
            img_numb =  int(img_path[-15:-10])
            pose = poses[img_numb][:]
            

            # Split line by " "
            pose = pose.split()
            # Convert to int
            pose = [float(i) for i in pose]

            # Pose matrix    
            WTP          = np.eye(4, dtype = float)
            WTP[3,3]     = 1
            WTP[0:3,0:3] = quaternion_to_euler_matrix(pose[0:4])
            WTP[0:3,3]   = pose[4:7]
            PTW = np.linalg.inv(WTP)

            min_bound_pcd_W = self.axisAlignedBBox["min_bound"]
            # Shape to homogenic coordinate
            min_bound_pcd_P = np.append(min_bound_pcd_W,1)
            min_bound_pcd_P = min_bound_pcd_P.reshape(-1,1)

            # Apply transformation to point
            min_bound_pcd_P = np.dot(PTW,min_bound_pcd_P)
            min_bound_pcd_P = np.reshape(min_bound_pcd_P, (1,4))
            min_bound_pcd_P = min_bound_pcd_P[0][:-1]   
           
            # -------------------------------
            max_bound_pcd_W = self.axisAlignedBBox["max_bound"]
            # Shape to homogenic coordinate
            max_bound_pcd_P = np.append(max_bound_pcd_W,1)
            max_bound_pcd_P = max_bound_pcd_P.reshape(-1,1)

            # Apply transformation to point
            max_bound_pcd_P = np.dot(PTW,max_bound_pcd_P)
            max_bound_pcd_P = np.reshape(max_bound_pcd_P, (1,4))
            max_bound_pcd_P = max_bound_pcd_P[0][:-1] 


            min_bound = self.computeRGB_projection(list(min_bound_pcd_P),intrinsics_matrix)    
            max_bound = self.computeRGB_projection(list(max_bound_pcd_P),intrinsics_matrix)
            
            RgbBBox = {"min_bound":min_bound,"max_bound":max_bound}
            self.RgbBBoxs.append(RgbBBox)
            break
   
    def computeROIs(self):

        # Compute weight and height
        width = self.RgbBBoxs[0]["max_bound"][0] - self.RgbBBoxs[0]["min_bound"][0]
        height = self.RgbBBoxs[0]["max_bound"][1] - self.RgbBBoxs[0]["min_bound"][1]
        
        width  = width  * 1.2
        height = height * 1.2

        width = max(width, height)
        height = width

        # print(width)
        # print(height)

        # print("width: " + str(width))
        # print("height: " + str(height))

        

        self.rgb_ROIs = []
        for idx, img in enumerate(PointCloudOperations.rgb_images):

            centroid = self.RGBCentroids[idx]
            # print("Centroid: " + str(centroid))
            # cv2.circle(img,centroid,20,(255,0,0),2)

            top_corn = [max(round((centroid[0]-width/2)),0),max(round((centroid[1]-height/2)),0)]
            bot_corn = [max(round((centroid[0]+width/2)),0),max(round((centroid[1]+height/2)),0)]
            
            # print("Top corner: " + str(top_corn))
            # print("Bot corner: " + str(bot_corn))
            cropped_img = img[top_corn[1]:bot_corn[1],top_corn[0]:bot_corn[0]]
            self.rgb_ROIs.append(cropped_img)

            # cv2.imshow("obj",cropped_img) 
            # cv2.waitKey(0)
            # self.rgb_ROIs.append(cropped_img)

            
            



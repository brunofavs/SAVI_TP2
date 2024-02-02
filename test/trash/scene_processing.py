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

view = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 0.5, 0.5, 0.5 ],
			"boundingbox_min" : [ -0.34867820089130808, -0.37054497667191461, -0.029999999999999999 ],
			"field_of_view" : 60.0,
			"front" : [ 0.45673827815753421, -0.68198350241761374, 0.57121681321184492 ],
			"lookat" : [ -0.01896661811825958, 0.046179291564060657, 0.049008207407589467 ],
			"up" : [ -0.43580189559863336, 0.38824860398903976, 0.81199737025017504 ],
			"zoom" : 0.86120000000000041
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

    def segment(self, distance_threshold=0.03, ransac_n=4, num_iterations=200):

        print('Starting plane detection')
        plane_model, inlier_idxs = self.point_cloud.segment_plane(distance_threshold=distance_threshold, 
                                                    ransac_n=ransac_n,
                                                    num_iterations=num_iterations)
        [self.a, self.b, self.c, self.d] = plane_model

        self.inlier_cloud = self.point_cloud.select_by_index(inlier_idxs)

        outlier_cloud = self.point_cloud.select_by_index(inlier_idxs, invert=False)

        return outlier_cloud

    def __str__(self):
        text = 'Segmented plane from pc with ' + str(len(self.point_cloud.points)) + ' with ' + str(len(self.inlier_cloud.points)) + ' inliers. '
        text += '\nPlane: ' + str(self.a) +  ' x + ' + str(self.b) + ' y + ' + str(self.c) + ' z + ' + str(self.d) + ' = 0' 
        return text

def crop_point_cloud_with_bbox(point_cloud, crop_box):
    # Create a bounding box using the provided crop_box
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=crop_box[0], max_bound=crop_box[1])

    # Crop the point cloud
    cropped_point_cloud = point_cloud.crop(bounding_box)

    return cropped_point_cloud


def main():

    # --------------------------------------
    # Initialization
    # --------------------------------------

    # Camera Intrinsics
    fx = 570
    fy = 570
    cx = 320
    cy = 240
    width = 640
    height = 480
    intrinsic_matrix = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    # Point Cloud Path
    dataset_path = f'{os.getenv("SAVI_TP2")}/dataset'
    scenes_path = dataset_path +'/scenes_dataset_v2/rgbd-scenes-v2_pc/rgbd-scenes-v2/pc/pcd' 
    scenes_paths = glob.glob(scenes_path + '/*.pcd')
    
    # Print available scenes pcd
    available_scenes = []
    for scenes in scenes_paths:
        file_name = scenes.split('/')
        file_name = file_name[-1]
        available_scenes.append(file_name)
    print('Available scenes: ' + str(available_scenes))

    # User input scene select
    # scene_n = input("Scene number: ")
    scene_n = "04"

    print('--------- Scene Properties --------- ')
    # filename = dataset_path + '/rgbd_scenes_v2/pcd/'+ scene_n + ".pcd"
    filename = f'{scenes_path}/{scene_n}.pcd'
    print('Loading file '+ filename)
    ptCloud_ori = o3d.io.read_point_cloud(filename)
    print(ptCloud_ori)

    # Downsample scene
    ptCloud_ori_downsampled = ptCloud_ori.voxel_down_sample(voxel_size=0.05) 
    print('After downsampling: ' + str(len(ptCloud_ori_downsampled.points)) + ' points')

    # Generate carteesian frame object
    seixos = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))

    # Load scene the image
    img_path = dataset_path + f'/scenes_dataset_v2/rgbd-scenes-v2_pc/rgbd-scenes-v2/imgs/scene_{scene_n}/00000-color.png'
    scene_img = cv2.imread(img_path) # relative path
    
    exit(0)
    # ------------------------------------------
    # Estimte normals and remove non horizontal planes
    # ------------------------------------------

    ptCloud_ori.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    ptCloud_ori.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, 1]))


    # Select voxels idx that are 90º degrees with X Camera Axis
    hori_idxs = []
    for idx,normal in enumerate(ptCloud_ori.normals):

        # Compute angle between two 3d vectors
        norm_normal = math.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)

        comp_axis = [1, 0, 0] # Compare with X axis
        norm_comp_axis = math.sqrt(comp_axis[0]**2 + comp_axis[1]**2 + comp_axis[2]**2)

        theta = math.acos(np.dot(normal, comp_axis) / (norm_normal * norm_comp_axis)) * 180/math.pi

        # Keep points where angle to z_axis is small enough
        if abs(theta - 90) < 0.05:  # we have a point that belongs to an horizontal surface
            hori_idxs.append(idx)

    # Create new point cloud
    ptCloud_hori = ptCloud_ori.select_by_index(hori_idxs)
    print("Horizontal: " + str(len(hori_idxs)))


    # ------------------------------------------
    # Remove Outliers
    # ------------------------------------------
    (ptCloud_hori_clean, ind) = ptCloud_hori.remove_radius_outlier(nb_points=300, radius=0.3)
    
    
    # ------------------------------------------
    # Find table plane
    # ------------------------------------------
    table_plane = PlaneDetection(ptCloud_hori_clean)
    ptCloud_table = table_plane.segment()
    ptCloud_table_center = ptCloud_table.get_center()



    # ------------------------------------------
    # Translate and Rotate original point cloud
    # ------------------------------------------
    ptCloud_GUI = deepcopy(ptCloud_ori)

    # Translate
    ptCloud_GUI.translate((-ptCloud_table_center[0],-ptCloud_table_center[1],-ptCloud_table_center[2]))
    
    # Rotate on X axis
    rot = ptCloud_GUI.get_rotation_matrix_from_xyz((-2.1,0,0))
    ptCloud_GUI.rotate(rot, center=(0, 0, 0,))

    # Rotate on Z axis
    rot = ptCloud_GUI.get_rotation_matrix_from_xyz((0,0,-2.1))
    ptCloud_GUI.rotate(rot, center=(0, 0, 0,))

    # Rotate on Y axis
    rot = ptCloud_GUI.get_rotation_matrix_from_xyz((0,-0.15,0))
    ptCloud_GUI.rotate(rot, center=(0, 0, 0,))

    # ------------------------------------------
    # Crop table from original point cloud
    # ------------------------------------------

    xmin = -0.5
    ymin = -0.5
    zmin = 0.05

    xmax = 0.5
    ymax = 0.5
    zmax = 0.4

    crop_box = np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])


    # Call the function to crop the point cloud
    ptCloud_GUI_croped = crop_point_cloud_with_bbox(ptCloud_GUI, crop_box)
    print("After croping: " + str(len(ptCloud_GUI_croped.points)))

    # ------------------------------------------
    # Cluster objects and save them
    # ------------------------------------------
    group_idxs = list(ptCloud_GUI_croped.cluster_dbscan(eps=0.045, min_points=50, print_progress=True))

    
    # Filter clusters (-1 means noise)
    obj_idxs = list(set(group_idxs))
    colormap = cm.Pastel1(range(0, len(obj_idxs)))
    if -1 in obj_idxs:
        obj_idxs.remove(-1)
    
    print("#Objects:  "+ str(len(obj_idxs)))
    
    obj_centers = []
    for obj_idx in obj_idxs:
        group_points_idxs = list(locate(group_idxs, lambda x: x == obj_idx))

        ptcloud_group = ptCloud_GUI_croped.select_by_index(group_points_idxs)

        # Save object
        filename = "../bin/objs/pcd/obj"+str(obj_idx)+".pcd"
        o3d.io.write_point_cloud(filename,ptcloud_group)

        # Rotate on Y axis
        rot = ptcloud_group.get_rotation_matrix_from_xyz((0,+0.15,0))
        ptcloud_group.rotate(rot, center=(0, 0, 0,))

        # Rotate on Z axis
        rot = ptcloud_group.get_rotation_matrix_from_xyz((0,0,+2.1))
        ptcloud_group.rotate(rot, center=(0, 0, 0,))

        # Roll back reference
        rot = ptcloud_group.get_rotation_matrix_from_xyz((+2.1,0,0))
        ptcloud_group.rotate(rot, center=(0, 0, 0,))

        # Translate
        ptcloud_group.translate((ptCloud_table_center[0],ptCloud_table_center[1],ptCloud_table_center[2]))

        # Get object center
        center = ptcloud_group.get_center()
        center_homo = np.asarray([center[0],center[1],center[2], 1])

        # dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
        img_point, _  = cv2.projectPoints(center,np.zeros((3,1)),np.zeros((3,1)),intrinsic_matrix,np.zeros((5,1)))
        print(img_point)
        img_point_B = img_point[0][0][:]
        print(img_point_B[0])
        cv2.circle(scene_img,(round(img_point_B[0]),round(img_point_B[1])),50,(0,0,255),3)
        
     
    cv2.imshow('scene', scene_img)
    cv2.waitKey(0)

    # group_point_clouds = []
    # for group_idx in group_idxs:
        # # Add color to object
        # color = colormap[group_idx, 0:3]
        # ptcloud_group.paint_uniform_color(color)
        # group_point_clouds.append(ptcloud_group)

    # --------------------------------------
    # Visualizations
    # --------------------------------------
    # entities = [ptcloud_group]
    # entities.append(seixos)
    # # entities.append(plane_ori_bounding_box)

    # # entities = [object_cloud]
    # o3d.visualization.draw_geometries(entities, 
    #                                   zoom   =view['trajectory'][0]['zoom'],
    #                                   front  =view['trajectory'][0]['front'],
    #                                   lookat =view['trajectory'][0]['lookat'],
    #                                   up     =view['trajectory'][0]['up'])

if __name__ == "__main__":
    
    main()
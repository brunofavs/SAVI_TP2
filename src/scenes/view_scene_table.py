#!/usr/bin/env python3
# View labeled scene

import open3d as o3d
import numpy as np
import glob
from copy import deepcopy
import math

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
    scene_n = "04"

    print('--------- Scene Properties --------- ')
    filename = dataset_path + '/rgbd_scenes_v2/pcd/'+ scene_n + ".pcd"
    print('Loading file '+ filename)
    ptCloud_ori = o3d.io.read_point_cloud(filename)
    print(ptCloud_ori)

    # Downsample scene
    ptCloud_ori_downsampled = ptCloud_ori.voxel_down_sample(voxel_size=0.05) 
    print('After downsampling: ' + str(len(ptCloud_ori_downsampled.points)) + ' points')

    # Generate carteesian frame object
    frame_plane = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))


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
    ptCloud_ori_horizontal = ptCloud_ori.select_by_index(hori_idxs)
    print("Horizontal: " + str(len(hori_idxs)))


    # ------------------------------------------
    # Remove Outliers
    # ------------------------------------------
    (ptCloud_ori_horizontal_clean, ind) = ptCloud_ori_horizontal.remove_radius_outlier(nb_points=300, radius=0.3)

    # ------------------------------------------
    # Find table plane
    # ------------------------------------------
    table_plane = PlaneDetection(ptCloud_ori_horizontal_clean)
    ptCloud_table = table_plane.segment()
    ptCloud_table_center = ptCloud_table.get_center()



    # ------------------------------------------
    # Create transformation matrix
    # ------------------------------------------

    # Translate
    ptCloud_ori_downsampled.translate((-ptCloud_table_center[0],-ptCloud_table_center[1],-ptCloud_table_center[2]))
    
    # Rotate on X axis
    rot = ptCloud_ori_downsampled.get_rotation_matrix_from_xyz((-2.1,0,0))
    ptCloud_ori_downsampled.rotate(rot, center=(0, 0, 0,))

    # Rotate on Z axis
    rot = ptCloud_ori_downsampled.get_rotation_matrix_from_xyz((0,0,-2.1))
    ptCloud_ori_downsampled.rotate(rot, center=(0, 0, 0,))

    # Rotate on Y axis
    rot = ptCloud_ori_downsampled.get_rotation_matrix_from_xyz((0,-0.15,0))
    ptCloud_ori_downsampled.rotate(rot, center=(0, 0, 0,))

    # ------------------------------------------
    # Crop table from original point cloud
    # ------------------------------------------
    



    # --------------------------------------
    # Visualizations
    # --------------------------------------
    entities = [ptCloud_ori_downsampled]
    # entities.append(table_plane.inlier_cloud)
    # entities.append(table_cloud)
    # entities.append(planes[0].inlier_cloud)
    # entities.append(planes[1].inlier_cloud)
    entities.append(frame_plane)
    # entities.append(plane_ori_bounding_box)

    # entities = [object_cloud]
    o3d.visualization.draw_geometries(entities, 
                                      zoom   =view['trajectory'][0]['zoom'],
                                      front  =view['trajectory'][0]['front'],
                                      lookat =view['trajectory'][0]['lookat'],
                                      up     =view['trajectory'][0]['up'])

if __name__ == "__main__":
    main()
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

class PointCloudOperations():
    def __init__(self,):
        pass

    def load(self, datapath ):
        
        # Load original pointcloud
        print('Loading file '+ datapath)
        self.ori = o3d.io.read_point_cloud(datapath)
        print(self.ori)

        # Copy original pcd
        self.gui = deepcopy(self.ori)

    def pre_process(self,voxelsize):

        # Downsample scene
        self.gui = self.gui.voxel_down_sample(voxel_size=voxelsize) 
        print('After downsampling: ' + str(len(self.gui.points)) + ' points')

        # Estimate normals
        self.gui.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        self.gui.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, 1]))


    def transgeom(self,rx,ry,rz,tx,ty,tz):
        
        # Rotate 
        rot = self.gui.get_rotation_matrix_from_xyz((math.radians(rx),math.radians(ry),math.radians(rz)))
        self.gui.rotate(rot, center=(0, 0, 0,))

        # Translate
        self.gui.translate((tx,ty,tz))

    def crop(self,xmin,ymin,zmin,xmax,ymax,zmax):

        crop_box = np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
    
        # Create a bounding box using the provided crop_box
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=crop_box[0], max_bound=crop_box[1])

        # Crop point cloud
        self.gui = self.gui.crop(bounding_box)

    def segment(self, distance_threshold=0.03, ransac_n=3, num_iterations=200 ,outliers = True,):

        print('Starting plane detection')
        _, inlier_idxs = self.gui.segment_plane(distance_threshold=distance_threshold, 
                                                    ransac_n=ransac_n,
                                                    num_iterations=num_iterations)


        self.gui = self.gui.select_by_index(inlier_idxs, invert=outliers)

    def view(self,seixos_on = True):
        entities = [self.gui]
         
        if seixos_on:
            seixos = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))
            entities.append(seixos)
        
        # entities.append(plane_ori_bounding_box)

        # entities = [object_cloud]
        o3d.visualization.draw_geometries(entities, 
                                          zoom   =view['trajectory'][0]['zoom'],
                                          front  =view['trajectory'][0]['front'],
                                          lookat =view['trajectory'][0]['lookat'],
                                          up     =view['trajectory'][0]['up'])

class PlaneDetection():
    def __init__(self, point_cloud):

        self.point_cloud = point_cloud

    def colorizeInliers(self, r,g,b):
        self.inlier_cloud.paint_uniform_color([r,g,b]) # paints the plane in red

    def segment(self, distance_threshold=0.03, ransac_n=4, num_iterations=200 ,outliers = True,):

        print('Starting plane detection')
        plane_model, inlier_idxs = self.point_cloud.segment_plane(distance_threshold=distance_threshold, 
                                                    ransac_n=ransac_n,
                                                    num_iterations=num_iterations)
        [self.a, self.b, self.c, self.d] = plane_model

        self.inlier_cloud = self.point_cloud.select_by_index(inlier_idxs)

        cloud = self.point_cloud.select_by_index(inlier_idxs, invert=outliers)

        return cloud

    def __str__(self):
        text = 'Segmented plane from pc with ' + str(len(self.point_cloud.points)) + ' with ' + str(len(self.inlier_cloud.points)) + ' inliers. '
        text += '\nPlane: ' + str(self.a) +  ' x + ' + str(self.b) + ' y + ' + str(self.c) + ' z + ' + str(self.d) + ' = 0' 
        return text

def get_scene_objs_centroids(datapath):

    # ------------------------------------------
    # Load Pointcloud
    # ------------------------------------------
    print('--------------------- PointCloud A --------------------- ')
    ptCloudA = PointCloudOperations()
    ptCloudA.load(datapath)
    ptCloudA.pre_process(voxelsize = 0.004)

    # ------------------------------------------
    # Estimte normals and remove non horizontal planes
    # ------------------------------------------

    # Select voxels idx that are 90º degrees with X Camera Axis
    hori_idxs = []
    for idx,normal in enumerate(ptCloudA.gui.normals):

        # Compute angle between two 3d vectors
        norm_normal = math.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)

        comp_axis = [1, 0, 0] # Compare with X axis
        norm_comp_axis = math.sqrt(comp_axis[0]**2 + comp_axis[1]**2 + comp_axis[2]**2)

        theta = math.acos(np.dot(normal, comp_axis) / (norm_normal * norm_comp_axis)) * 180/math.pi

        # Keep points where angle to z_axis is small enough
        if abs(theta - 90) < 0.05:  # we have a point that belongs to an horizontal surface
            hori_idxs.append(idx)

    # Create new point cloud with only horizontal point cloud
    ptCloudA.gui = ptCloudA.gui.select_by_index(hori_idxs)
    print("Horizontal points: " + str(len(hori_idxs)))

    
    # ------------------------------------------
    # Remove Outliers
    # ------------------------------------------

    (ptCloudA.gui, _) = ptCloudA.gui.remove_radius_outlier(nb_points=300, radius=0.3)
    print("With no outliers: " + str(len(ptCloudA.gui.points)))
    
    # ------------------------------------------
    # Find table center
    # ------------------------------------------
    ptCloudA.segment(distance_threshold=0.03, ransac_n=3, num_iterations=200, outliers = False)
    table_center = ptCloudA.gui.get_center()
    print("Table center at: " + str(table_center))


    # ------------------------------------------
    # Load new Pointcloud
    # ------------------------------------------
    print('--------------------- PointCloud B --------------------- ')
    ptCloudB = PointCloudOperations()
    ptCloudB.gui = ptCloudA.ori
    ptCloudB.pre_process(voxelsize = 0.001)

    # ------------------------------------------
    # Crop table
    # ------------------------------------------
    
    # Translate to the center of the table
    ptCloudB.transgeom(0,0,0,-table_center[0],-table_center[1],-table_center[2]) 

    # Rotate to align references 
    ptCloudB.transgeom(-120,0,0,0,0,0)
    ptCloudB.transgeom(0,0,-120,0,0,0)
    ptCloudB.transgeom(0,-10,0,0,0,0)
    
    # Crop point cloud
    ptCloudB.crop(-0.5,-0.5,-0.05,0.5,0.5,0.4)
    print("After croping: " + str(len(ptCloudB.gui.points)))
    
    # Remove talbe
    ptCloudB.segment(distance_threshold=0.03, ransac_n=3, num_iterations= 200, outliers = True)
    # ptCloudB.view()

    # ------------------------------------------
    # Cluster objects, get center and save them
    # ------------------------------------------
    group_idxs = list(ptCloudB.gui.cluster_dbscan(eps=0.045, min_points=50, print_progress=True))

    # Filter clusters (-1 means noise)
    obj_idxs = list(set(group_idxs))
    colormap = cm.Pastel1(range(0, len(obj_idxs)))
    if -1 in obj_idxs:
        obj_idxs.remove(-1)
    
    # Delete existing file
    for file in glob.glob('../bin/objs/pcd/*.pcd'):
        os.remove(file)
        print('Temporary .pcd files removed')


    print("#Objects:  "+ str(len(obj_idxs)))

    obj_centers = np.zeros((len(obj_idxs),3))
    for obj_idx in obj_idxs:
        group_points_idxs = list(locate(group_idxs, lambda x: x == obj_idx))

        ptcloud_group = ptCloudB.gui.select_by_index(group_points_idxs)

        # Save object
        filename = "../bin/objs/pcd/obj"+str(obj_idx)+".pcd"
        o3d.io.write_point_cloud(filename,ptcloud_group)

        #  Reverte translations and rotation
    
        ptCloudB.transgeom(0,10,0,0,0,0)
        ptCloudB.transgeom(0,0,120,0,0,0)
        ptCloudB.transgeom(120,0,0,0,0,0)
        ptCloudB.transgeom(0,0,0,table_center[0],table_center[1],table_center[2]) 


        # Get object center
        obj_centers[obj_idx,:] = np.asarray(ptcloud_group.get_center())

    return obj_centers

def objs_images(img_path,centroids, intrinsics):

    #Load image
    scene_img = cv2.imread(img_path)

    
    for centroid in centroids:  
        #Convert world centroid to camera center  
        img_point, _  = cv2.projectPoints(centroid,np.zeros((3,1)),np.zeros((3,1)),intrinsics,np.zeros((5,1)))
        img_point[0][0][:] = img_point


        seg_size = (224,224)
        start_point =  (img_point[0]-seg_size[0]/2, img_point[1]-seg_size[1]/2)
        end_point   =  (img_point[0]+seg_size[0]/2, img_point[1]+seg_size[1]/2)
        #Draw rectangle on main image
        cv2.rectangle(scene_img,start_point, end_point,(0,0,255),3)
    cv2.imshow('scene', scene_img)
    cv2.waitKey(0)
def main():

    # --------------------------------------
    # Initialization
    # --------------------------------------

    # Scritp parameters
    scene_n = "01"
    dataset_path = f'{os.getenv("SAVI_TP2")}/dataset'


    #Load Camera Intrinsics
    with open("./lib/jsons/intrinsic.json",'r') as f:
        extrinsics_matrix = np.asarray(json.load(f))
    
    #Load scene Image
    img_path = dataset_path + f'/scenes_dataset_v2/rgbd-scenes-v2_pc/rgbd-scenes-v2/imgs/scene_{scene_n}/00000-color.png'

    #Load scene pointcloud
    scenes_path = dataset_path + f'/scenes_dataset_v2/rgbd-scenes-v2_pc/rgbd-scenes-v2/pc/pcd/{scene_n}.pcd' 

    # --------------------------------------
    # Execution
    # --------------------------------------

    # Get scene objects centroid
    # centroids = get_scene_objs_centroids(scenes_path)


    centroids =np.asarray([[ 0.35646105 ,-0.22050809  ,0.06902391],                     
                [ 0.16774505 ,-1.3631073   ,0.68673332],
                [ 0.93245048 ,-1.44329362  ,1.33248909],
                [ 0.42233888 ,-1.840807    ,1.88490167],
                [ 2.00624375 ,-2.83768111  ,1.79761973]])

    # Segment scene image based on centroid locaition
    image_out = objs_images(img_path,centroids,extrinsics_matrix)
    
    exit(0)



if __name__ == "__main__":
    
    main()
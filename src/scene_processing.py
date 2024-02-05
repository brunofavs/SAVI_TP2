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
from scipy.spatial.transform import Rotation

# TODO list:
# .Groundtruth imagens;
# .Bounding box mask;
# .Scenes com labeling;

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
    def __init__(self):
        pass

    def load(self, datapath):
        
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
   
def mostCommon(lst):
    # Converts to set and then tests for each unique element the return value of lst.count(element)
    # Without converting to set would still work but would be a lot less efficient
    return max(set(lst), key=lst.count)

def objsPtcloudSegmentation(scenes_path,dump_path):
    

    # ------------------------------------------
    # Load Pointcloud
    # ------------------------------------------
    print('--------------------- PointCloud A --------------------- ')
    ptCloudA = PointCloudOperations()
    ptCloudA.load(scenes_path)
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
    
    '''TODO Maybe instead search for the most quadrangular/circular segment?
            What happens if the table isn't in the center?
    '''


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
    # * This is hardcoded, what would be a better way?
    ptCloudB.transgeom(-120,0,0,0,0,0)
    ptCloudB.transgeom(0,0,-120,0,0,0)
    ptCloudB.transgeom(0,-10,0,0,0,0)
    
    # Crop point cloud
    ptCloudB.crop(-0.5,-0.5,-0.05,0.5,0.5,0.4)
    print("After croping: " + str(len(ptCloudB.gui.points)))
    
    # Remove talbe
    ptCloudB.segment(distance_threshold=0.03, ransac_n=3, num_iterations= 200, outliers = True)
    

    # ------------------------------------------
    # Cluster objects, get center and save them
    # ------------------------------------------

    #  Revert translations and rotation
    # *Again, this will break upon switching scene
    
    ptCloudB.transgeom(0,10,0,0,0,0)
    ptCloudB.transgeom(0,0,120,0,0,0)
    ptCloudB.transgeom(120,0,0,0,0,0)
    ptCloudB.transgeom(0,0,0,table_center[0],table_center[1],table_center[2]) 

    group_idxs = list(ptCloudB.gui.cluster_dbscan(eps=0.045, min_points=50, print_progress=True))

    # Filter clusters (-1 means noise)
    obj_idxs = list(set(group_idxs))
    colormap = cm.Pastel1(range(0, len(obj_idxs)))
    if -1 in obj_idxs:
        obj_idxs.remove(-1)
    
    # Delete existing files
    for file in glob.glob(dump_path+ 'pcd/*.pcd'):
        os.remove(file)
    print('Temporary .pcd files removed')


    print("#Objects:  "+ str(len(obj_idxs)))

    for obj_idx in obj_idxs:
        group_points_idxs = list(locate(group_idxs, lambda x: x == obj_idx))

        ptcloud_group = ptCloudB.gui.select_by_index(group_points_idxs)

        # Save object pcd
        filename = dump_path + "pcd/obj_"+str(obj_idx)+".pcd"
        o3d.io.write_point_cloud(filename,ptcloud_group)

    print("Objects pcd saved at " + dump_path + "pcd/")

def objsImages(img_path,objs_props,intrinsics_matrix,objs_path):
    
    quat_df = [0.451538, -0.0219017, 0.812727, 0.367557]
    rot = Rotation.from_quat(quat_df)
    rot_euler = rot.as_euler('xyz', degrees=False)
    trans =  np.asarray([-0.674774, -0.776193, 1.75478])

    print(rot_euler)

    print("")
    print('--------------------- Obj Image Croppping --------------------- ')

    #Load image
    scene_ori = cv2.imread(img_path)
    scene_gui = deepcopy(scene_ori)

    # Delete existing files
    for file in glob.glob(objs_path + 'rgb/*.png'):
        os.remove(file)
    print('Temporary .png files removed')

    count = 0
    for obj in objs_props:  

        obj_gui = scene_gui 

        label      = obj[0]
        centroid_W = obj[1]
        min_bond_W = obj[2]
        max_bond_W = obj[3]

        # # Mask background
        # min_bond_I,_ = cv2.projectPoints(min_bond_W,np.zeros((3,1)),np.zeros((3,1)),intrinsics_matrix,np.zeros((5,1)))
        # max_bond_I,_ = cv2.projectPoints(max_bond_W,np.zeros((3,1)),np.zeros((3,1)),intrinsics_matrix,np.zeros((5,1)))
        # min_bond_I = [round(num) for num in min_bond_I[0][0][:]]  
        # max_bond_I = [round(num) for num in max_bond_I[0][0][:]]  

        # masked = np.zeros_like(obj_gui)
        # masked[min_bond_I[1]:max_bond_I[1],min_bond_I[0]:max_bond_I[0]] = obj_gui[min_bond_I[1]:max_bond_I[1],min_bond_I[0]:max_bond_I[0]]


        #Convert world centroid to camera center  
        # centr_I, _  = cv2.projectPoints(centroid_W,np.zeros((3,1)),np.zeros((3,1)),intrinsics_matrix,np.zeros((5,1)))
        centr_I, _  = cv2.projectPoints(centroid_W,rot_euler,trans,intrinsics_matrix,np.zeros((5,1)))
        centr_I     = centr_I[0][0][:]

        print(centr_I)
        cv2.circle(scene_gui, (round(centr_I[0]),abs(round(centr_I[1]))), 20, (255,0,0), 2)
        cv2.imshow('scene', scene_gui)
        cv2.waitKey(0)
        continue

        seg_size    = (224,224)
        start_point =  (round(centr_I[0]-seg_size[0]/2), round(centr_I[1]-seg_size[1]/2))
        end_point   =  (round(centr_I[0]+seg_size[0]/2), round(centr_I[1]+seg_size[1]/2))
        
        # Crop and save image
        cropped_image = obj_gui[start_point[1]: start_point[1]+seg_size[1], start_point[0]: start_point[0]+seg_size[0]] 

        # Save image
        cv2.imwrite(objs_path + "rgb/"+label+".png", cropped_image)
        count = count + 1

        #Draw rectangle on gui image
        cv2.rectangle(scene_ori,start_point, end_point,(0,0,255),3)

        cv2.imshow('scene', cropped_image)
        cv2.waitKey(0)

    print("Objects images saved at " + objs_path + "objs/rgb/")

    return scene_ori

def objsPtcloudLabeling(scene_path,objs_path,labels_path):

    print("")
    print('--------------------- Objs setting Ground Truth --------------------- ')
    # ------------------------------------------
    # Load scene labeling
    # ------------------------------------------
    # bowl=1, cap=2, cereal_box=3, coffee_mug=4, coffee_table=5 
    # office_chair=6, soda_can=7, sofa=8, table=9, background=10
    
    f = open(labels_path,'r')
    labels = f.read().splitlines()
    labels.pop(0) # Remove first item of label list

    label_dict = {
        '1': "bowl"         ,    
        '2': "cap"          ,
        '3': "cereal_box"   ,
        '4': "coffee_mug"   ,
        '5': "coffee_table" ,
        '6': "office_chair" ,
        '7': "soda_can"     ,
        '8': "sofa"         ,
        '9': "table"        ,
        '10': "background"   
    }

    # ------------------------------------------
    # Process objs pointcloud
    # ------------------------------------------
    for obj_pcd in os.listdir(objs_path + "pcd/"):

        # Load obj pointcloud
        ptCloud_obj = o3d.io.read_point_cloud(objs_path + "pcd/" + obj_pcd)
        # Load scene pointcloud (for groundtruth)
        ptCloud_ori = o3d.io.read_point_cloud(scene_path)
 
        # Find obj index from croped obj point cloud
        dists = ptCloud_ori.compute_point_cloud_distance(ptCloud_obj)
        dists = np.asarray(dists)
        scene_ind = np.where(dists < 0.0003)[0]

        obj_label = []
        for idx in scene_ind:
            label = labels[idx]
            obj_label.append(label)

        label_name = label_dict[mostCommon(obj_label)]

        # # Get geometric centroid
        # centroid = ptCloud_obj.get_center()

        # Rename object
        src = objs_path + "pcd/" + obj_pcd
        dst = objs_path + "pcd/obj_"+ label_name + ".pcd"
        os.rename(src, dst)

        print("Renamed " + obj_pcd + " to obj_" + label_name + ".pcd")

def objsPtcloudProperties(objs_path):

    print("")
    print('--------------------- Objs Properties --------------------- ')
   
    objs_props = []
    for obj_pcd in os.listdir(objs_path + "pcd/"):

        # Load obj pointcloud
        ptCloud_obj = o3d.io.read_point_cloud(objs_path + "pcd/" + obj_pcd)

        # Data from pointcloud
        label = obj_pcd[:-4]
        centroid = ptCloud_obj.get_center()
        bbox     = ptCloud_obj.get_axis_aligned_bounding_box()

        min_bound =  bbox.get_min_bound()
        max_bond  =  bbox.get_max_bound()

        # Save data
        data = [label, centroid, min_bound, max_bond]
        objs_props.append(data)
        
        # # Print data
        # print("------ "+ label + " ------")
        # print("Centroid: "+ str(centroid))
        # print(bbox)

        # axis_aligned_bounding_box = ptCloud_obj.get_axis_aligned_bounding_box()
        # axis_aligned_bounding_box.color = (1.0,0,0)

        # entities = [ptCloud_obj, axis_aligned_bounding_box]
        # o3d.visualization.draw_geometries(entities)



    return objs_props
def main():

    # --------------------------------------
    # Initialization
    # --------------------------------------

    # Scritp parameters
    scene_n = "01"
    dataset_path = f'{os.getenv("SAVI_TP2")}/dataset'


    #Load Camera Intrinsics
    with open("./lib/jsons/intrinsic.json",'r') as f:
        intrinsics_matrix = np.asarray(json.load(f))
    

    #Load scene pointcloud
    img_path   = dataset_path + f'/scenes_dataset_v2/rgbd-scenes-v2_pc/rgbd-scenes-v2/imgs/scene_{scene_n}/00462-color.png'
    scene_path = dataset_path + f'/scenes_dataset_v2/rgbd-scenes-v2_pc/rgbd-scenes-v2/pc/pcd/{scene_n}.pcd' 
    label_path = dataset_path + f'/scenes_dataset_v2/rgbd-scenes-v2_pc/rgbd-scenes-v2/pc//{scene_n}.label'

    # Path to Dump objects pcd and images
    # objs_path = dataset_path + '../bin/objs/'
    objs_path = '../bin/objs/'

    # --------------------------------------
    # Execution
    # --------------------------------------

    # Segment objects from scene
    # objsPtcloudSegmentation(scene_path,objs_path)

    # objsPtcloudLabeling(scene_path,objs_path,label_path)

    objs_props =  objsPtcloudProperties(objs_path)

    # # Segment scene image based on centroid locaition
    image_out = objsImages(img_path,objs_props,intrinsics_matrix,objs_path)
    
    cv2.imshow('scene', image_out)
    cv2.waitKey(0)
    exit(0)



if __name__ == "__main__":
    main()
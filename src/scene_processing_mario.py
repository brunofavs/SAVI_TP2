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

# TODO: 
# Selecionar X em x imagens
# Buscar as poses
# Buscar as imagens dos objetos
# Gravar numa pasta

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

def objsPtcloudProperties(objs_path):

    print("")
    print('--------------------- Objs Properties --------------------- ')
   
    objs_props = []
    for obj_pcd in os.listdir(objs_path + "pcd/"):

        # Load obj pointcloud
        ptCloud_obj = o3d.io.read_point_cloud(objs_path + "pcd/" + obj_pcd)

        # Data from pointcloud
        label = obj_pcd[:-4]
        centroid_W = ptCloud_obj.get_center()
        bbox     = ptCloud_obj.get_axis_aligned_bounding_box()

        min_bound =  bbox.get_min_bound()
        max_bond  =  bbox.get_max_bound()

        print("-------- " + str(label)+" -------- ")

        # Generate seixos
        seixos_ori = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))
        seixos_gui = deepcopy(seixos_ori)
        

        # 0.449246 -0.0213099 0.813616 0.368433 -0.66902 -0.779701 1.75719  Line 463
        quat_df = np.asarray([0.449246, -0.0213099, 0.813616, 0.368433])
        trans =  np.asarray([-0.66902, -0.779701, 1.75719])
        # Rotation
        rot = seixos_gui.get_rotation_matrix_from_quaternion(quat_df)
        seixos_gui.rotate(rot, center=(0, 0, 0,))
        # Translate
        seixos_gui.translate(trans)

        # Pose matrix    
        WTP = np.zeros((4,4), dtype = float)
        WTP[3,3] = 1
        WTP[0:3,0:3] = rot
        WTP[0:3,3]   = trans

        PTW = np.linalg.inv(WTP)
    
        print("Wrld centroid: " + str(centroid_W))
        
        # Shape to homogenic coordinate
        centroid_P = np.append(centroid_W,1)
        centroid_P = centroid_P.reshape(-1,1)

        # Apply transformation to point
        centroid_P = np.dot(PTW,centroid_P)
        centroid_P = np.reshape(centroid_P, (1,4))
        centroid_P = centroid_P[0][:-1]        
        print("Pose Centroid: "+str(centroid_P))

        # Visualization
        entities = [ptCloud_obj, seixos_ori,seixos_gui]
        # o3d.visualization.draw_geometries(entities)

   
        # Save data
        data = [label, centroid_W,centroid_P, min_bound, max_bond, rot, trans]
        objs_props.append(data)

    return objs_props

def objsImages(img_path,pose,objs_props,intrinsics_matrix,objs_path):

    img_name =  img_path[-15:]
    print("-------------------------------------------------------")
    print('Loading ' + img_name)
    # print('Pose: ' + str(pose)) 
    print()
    # Pose matrix    
    WTP          = np.zeros((4,4), dtype = float)
    WTP[3,3]     = 1
    WTP[0:3,0:3] = quaternion_to_euler_matrix(pose[0:4])
    WTP[0:3,3]   = pose[4:7]
    PTW = np.linalg.inv(WTP)

    #Load image
    scene_ori = cv2.imread(img_path)
    scene_gui = deepcopy(scene_ori)

    # Delete existing files
    for file in glob.glob(objs_path + 'rgb/cropped/*.png'):
        os.remove(file)
    print('Temporary .png files removed')


    for obj in objs_props:  

        label       = obj[0]
        centroid_w  = obj[1]
        
        # Shape to homogenic coordinate
        centroid_P = np.append(centroid_w,1)
        centroid_P = centroid_P.reshape(-1,1)

        # Apply transformation to point
        centroid_P = np.dot(PTW,centroid_P)
        centroid_P = np.reshape(centroid_P, (1,4))
        centroid_P = centroid_P[0][:-1]        


        print("-------- " + str(label)+" -------- ")
        #Convert world centroid to camera center  
        centr_I, _  = cv2.projectPoints(centroid_P,np.zeros((3,1)),np.zeros((3,1)),intrinsics_matrix,np.zeros((5,1)))
        centr_I     = centr_I[0][0][:]

        print("Img centroid: " + str(centr_I))
        cv2.circle(scene_gui, (round(centr_I[0]),abs(round(centr_I[1]))), 20, (255,0,0), 2)  
        cv2.imshow('scene', scene_gui)
        
        # continue

    cv2.waitKey(0)
    print("Objects images saved at " + objs_path + "objs/rgb/")

    return scene_ori

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
    img_path   = dataset_path + f'/scenes_dataset_v2/rgbd-scenes-v2_pc/rgbd-scenes-v2/imgs/scene_{scene_n}/'
    # img_path   = dataset_path + f'/scenes_dataset_v2/rgbd-scenes-v2_pc/rgbd-scenes-v2/imgs/scene_{scene_n}/00000-color.png'
    
    scene_path = dataset_path + f'/scenes_dataset_v2/rgbd-scenes-v2_pc/rgbd-scenes-v2/pc/pcd/{scene_n}.pcd' 
    label_path = dataset_path + f'/scenes_dataset_v2/rgbd-scenes-v2_pc/rgbd-scenes-v2/pc/{scene_n}.label'
    pose_path  = dataset_path + f'/scenes_dataset_v2/rgbd-scenes-v2_pc/rgbd-scenes-v2/pc/{scene_n}.pose'


    # Path to Dump objects pcd and images
    objs_path = '../bin/objs/'


    # --------------------------------------
    # Execution
    # --------------------------------------


    objs_props =  objsPtcloudProperties(objs_path)

    # Get pose from scene
    f = open(pose_path,'r')
    poses = f.read().splitlines()

    # Get available scene images
    img_paths = glob.glob(img_path + '/*-color.png')
    n_images = len(img_paths)
    print(str(n_images) + " images found")

    
    n_divs = 10


    for img_path in img_paths[::round(n_images/n_divs)]:
           
        img_numb =  int(img_path[-15:-10])
        pose = poses[img_numb][:]

        # Split line by " "
        pose = pose.split()
        # Convert to int
        pose = [float(i) for i in pose]

        objsImages(img_path,pose,objs_props,intrinsics_matrix,objs_path)



    
    # # Segment scene image based on centroid locaition
    # objsImages(img_path,objs_props,intrinsics_matrix,objs_path)
    



if __name__ == "__main__":
    main()
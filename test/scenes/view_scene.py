#!/usr/bin/env python3
# View original scene
# Mário Vasconcelos 2023

import os
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
from copy import deepcopy

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

    dataset_path = os.getenv('SAVI_TP2')+ '/dataset'
    scene_path = dataset_path + f'/scenes_dataset_v2/rgbd-scenes-v2_pc/rgbd-scenes-v2/pc/pcd/01.pcd' 
    
    
    # --------------------------------------
    # Execution
    # --------------------------------------
    ptCloud = o3d.io.read_point_cloud(scene_path)

    seixos_ori = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))
    seixos_gui = deepcopy(seixos_ori)


    # 0.451538 -0.0219017 0.812727 0.367557 -0.674774 -0.776193 1.75478 
    quat_df = [0.451538, -0.0219017, 0.812727, 0.367557]
    trans =  np.asarray([-0.674774, -0.776193, 1.75478])

    # Rotation
    rot = seixos_gui.get_rotation_matrix_from_quaternion(quat_df)
    seixos_gui.rotate(rot, center=(0, 0, 0,))
    # Translate
    seixos_gui.translate(trans)
    
 

    # --------------------------------------
    # Visualizations
    # --------------------------------------
    entities = [ptCloud,seixos_ori,seixos_gui]
    o3d.visualization.draw_geometries(entities,
                                      zoom=view['trajectory'][0]['zoom'],
                                      front=view['trajectory'][0]['front'],
                                      lookat=view['trajectory'][0]['lookat'],
                                      up=view['trajectory'][0]['up'])
    
if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# View labeled scene

import open3d as o3d
from copy import deepcopy
import math
import numpy as np
view = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 0.36552801728248596, 0.4259859025478363, 0.14010117948055267 ],
			"boundingbox_min" : [ 0.20026302337646484, 0.28656396269798279, 0.050015594810247421 ],
			"field_of_view" : 60.0,
			"front" : [ 0.30156568175015352, -0.71206056049748678, 0.63405669917963181 ],
			"lookat" : [ -0.01896661811825958, 0.046179291564060657, 0.049008207407589467 ],
			"up" : [ -0.37292618198900029, 0.5239458279036624, 0.7657720497703302 ],
			"zoom" : 2.0
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}

def save_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()

def save_rgb_depth(pcd,rx,ry,rz):

    # Rotate point cloud
    rot = pcd.get_rotation_matrix_from_xyz((math.degrees(rx),math.degrees(ry),math.degrees(rz)))
    pcd.rotate(rot, center=(0, 0, 0,))


    # Create new visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    # Render options
    vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.Color
    # vis.get_render_option().point_size = 3.0

    # Save rbg and depth images
    vis.capture_depth_image("objs/rgbd/obj0_depth.png", do_render=True)
    vis.capture_screen_image("objs/rgbd/obj0_rgb.png", do_render=True)
    # vis.run()
    vis.destroy_window() 

def main():
    pass

if __name__ == "__main__":
    
    #Open point cloud
    filename = "./objs/pcd/obj1.pcd"
    ptCloud_obj = o3d.io.read_point_cloud(filename)
    frame_plane = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))
    # save_view_point(ptCloud_obj, "./objs/viewpoint.json")
    save_rgb_depth(ptCloud_obj,20,0,0)
    
    exit(0)
    entities = []
    entities.append(ptCloud_obj)
    entities.append(frame_plane)

    o3d.visualization.draw_geometries(entities)
                                      
    # main()
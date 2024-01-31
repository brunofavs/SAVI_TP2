#!/usr/bin/env python3
# View labeled scene

import open3d as o3d
from copy import deepcopy


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


def main():

    # --------------------------------------
    # Initialization
    # --------------------------------------

    #Open point cloud
    filename = "./objs/pcd/obj0.pcd"
    ptCloud_obj = o3d.io.read_point_cloud(filename)

    # --------------------------------------
    # Create Image from point cloud
    # --------------------------------------
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.Color
    # vis.get_render_option().point_size = 3.0

    vis.add_geometry(ptCloud_obj)
    vis.capture_screen_image("objs/rgb/obj0.jpg", do_render=True)
    vis.destroy_window()

    exit(0)
    # --------------------------------------
    # Visualizations
    # --------------------------------------
    entities = []
    entities.append(ptCloud_obj)

    # entities = [object_cloud]
    o3d.visualization.draw_geometries(entities, 
                                      zoom   =view['trajectory'][0]['zoom'],
                                      front  =view['trajectory'][0]['front'],
                                      lookat =view['trajectory'][0]['lookat'],
                                      up     =view['trajectory'][0]['up'])

if __name__ == "__main__":
    main()
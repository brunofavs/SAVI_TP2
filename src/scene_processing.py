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
import argparse

import matplotlib.pyplot as plt

from lib.PCD_Ops import *
from lib.googleTtS import text2Speech
from rgb_model_wrapper import modelWrapper

from lib.ESRGAN_upscaler.upscaler_adapted import imageUpscaler
from lib.ESRGAN_upscaler.test import upscaler2

"""'
TODO

# ///add suplots with all imgs to plot shower in model done

# ///use cmd line arguments to turn on upscale - done

beautify?

colorama nos coments

plot loss/epoch

"""


view = {
    "class_name": "ViewTrajectory",
    "interval": 29,
    "is_loop": False,
    "trajectory": [
        {
            "boundingbox_max": [
                2.7116048336029053,
                1.2182252407073975,
                3.8905272483825684,
            ],
            "boundingbox_min": [
                -2.4257750511169434,
                -1.6397310495376587,
                -1.3339539766311646,
            ],
            "field_of_view": 60.0,
            "front": [-0.86797159921609002, -0.27893672293557875, -0.41087663300828348],
            "lookat": [-0.13711904974393682, -0.307851942294602, 1.4581324743200366],
            "up": [0.39786518017848138, -0.88571554082741299, -0.23918879392301828],
            "zoom": 0.29999999999999999,
        }
    ],
    "version_major": 1,
    "version_minor": 0,
}


def flatten_list(nested_list):
    flattened_list = []
    for item in nested_list:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list


def main():

    # * ---Configuration of argparse----
    parser = argparse.ArgumentParser(description="Scene Describer")
    parser.add_argument(
        "-pNN", "--plotNN", action="store_true", help="Plots NN Classification"
    )
    parser.add_argument(
        "-us", "--upscale", action="store_true", help="Use Upscale for ROIs"
    )
    parser.add_argument(
        "-s", "--scene", type=str, required=False, default="01", help="Chooses scene"
    )
    args = vars(parser.parse_args())
    # * Initialize model wrapper

    # predicter = modelWrapper(model_name = "densenet121_full_4ep.pkl")
    predicter = modelWrapper(
        model_name="densenet121_mini_10_ep.pkl",
        matchings_name="rgb_images_matchings_mini.json",
    )

    # * Initialize image upscaler

    upscaler = imageUpscaler()

    # Script parameters
    dataset_path = f'{os.getenv("SAVI_TP2")}/dataset'
    # scene_n = "02"
    # use_upscale = False
    # plot_NN = False

    scene_n = args["scene"]
    use_upscale = args["upscale"]
    plot_NN = args["plotNN"]

    # Load Camera Intrinsics
    with open(f'{os.getenv("SAVI_TP2")}/src/jsons/intrinsic.json', "r") as f:
        intrinsics_matrix = np.asarray(json.load(f))

    # Load scene pointcloud
    # img_path   = f"{dataset_path}/scenes_dataset_v2/rgbd-scenes-v2_pc/rgbd-scenes-v2/imgs/scene_{scene_n}/"
    pose_path = f"{dataset_path}/scenes_dataset_v2/rgbd-scenes-v2_pc/rgbd-scenes-v2/pc/{scene_n}.pose"

    # Bruno
    img_path = f"{dataset_path}/scenes_dataset_v2/rgbd-scenes-v2_imgs/rgbd-scenes-v2/imgs/scene_{scene_n}/"
    # img_path   = f"{dataset_path}/scenes_dataset_v2/rgbd-scenes-v2_imgs/rgbd-scenes-v2/imgs/scene_{scene_n}/00000-color.png"
    scene_path = f"{dataset_path}/scenes_dataset_v2/rgbd-scenes-v2_pc/rgbd-scenes-v2/pc/pcd/{scene_n}.pcd"
    label_path = f"{dataset_path}/scenes_dataset_v2/rgbd-scenes-v2_pc/rgbd-scenes-v2/pc//{scene_n}.label"
    jsons_path = f"{dataset_path}/jsons"

    clustered_pcds_path = f'{os.getenv("SAVI_TP2")}/bin/objs/pcd'

    for file in glob.glob(f"{clustered_pcds_path}/*.pcd"):
        os.remove(file)
    print("Temporary .pcd files removed")

    with open(label_path, "r") as f:
        scene_labels = f.read().splitlines()
        scene_labels.pop(0)  # Remove first item of label list

    try:
        with open(f"{jsons_path}/scene_labeling_matchings.json", "r") as f:
            scene_label_dict = json.load(f)
    except:
        print(f"No scene labeling dictionary found, using defaults")
        scene_label_dict = {
            "1": "bowl",
            "2": "cap",
            "3": "cereal_box",
            "4": "coffee_mug",
            "5": "coffee_table",
            "6": "office_chair",
            "7": "soda_can",
            "8": "sofa",
            "9": "table",
            "10": "background",
        }

    # Load poses file
    f = open(pose_path, "r")
    poses = f.read().splitlines()

    # Get available scene images
    img_paths = glob.glob(img_path + "/*-color.png")
    n_images = len(img_paths)
    print(str(n_images) + " images found")

    # --------------------------------------------------
    # Execution
    # --------------------------------------------------
    original_scene_pcd = o3d.io.read_point_cloud(scene_path)
    scene_pre_processed_checkpoints = []

    # ? Segmenting other scenes doesn't work as well
    # * 1º Step - Find table center

    scene_operations = PointCloudOperations(original_scene_pcd, perspective=view)
    scene_operations.pre_process(voxelsize=0.004)

    # Filter horizontal planes
    scene_operations.estimateNormals()
    scene_operations.computeVerticalNormals()
    scene_operations.filterByIdx()

    scene_pre_processed_checkpoints.append(scene_operations.gui_pcd)

    scene_operations.segment(outliers=False)

    # Voxelize further because only need to extract the rough diameter and rough center
    # And before clustering because that operation is very heavy
    scene_operations.voxelize(voxel_size=0.04)

    # This guarantees the center is still the table if the table isnt centered and there are a lot more blobs
    scene_operations.filterBiggestCluster(eps=0.3)

    camera2table_center = scene_operations.gui_pcd.get_center()
    table_center2camera = camera2table_center * -1

    scene_operations.transGeom(translation=table_center2camera)

    # * 2º Step - Find table size to crop and angle to

    # Eps relies heavily on the density and gives bad results if its too small
    table_scene_operations = scene_operations
    table_scene_operations.filterBiggestCluster(eps=0.3)
    table_scene_operations.removeStatisticalNoise()
    # Find diameter of table

    diameter = table_scene_operations.getDiameter()
    radius = diameter / 2

    # Need to recompute the normals because now they are fewer, it's not the most elegant solution though
    table_scene_operations.estimateNormals()
    z_offset_about_x = table_scene_operations.computeAngle(reference_vector=[0, 0, 1])

    # * 3º Step - Find object clusters and their centroids

    # Reseting scene operations to the original scene pre-processed (without over-voxelization / cropping)
    # scene_operations.gui_pcd = scene_pre_processed_checkpoints[-1]
    scene_operations.gui_pcd = original_scene_pcd

    # ! Cannot do translation and rotation at the same time, this is a bug
    # scene_operations.transgeom(rotation=np.array([z_offset_about_x,0,0]),translation = table_center2camera)
    scene_operations.transGeom(translation=table_center2camera)
    scene_operations.transGeom(rotation=np.array([z_offset_about_x, 0, 0]))

    # * Both this
    scene_operations.cropPcd(
        np.array([-radius, -0.5, -radius]), np.array([radius, -0.025, radius * 0.7])
    )

    ##* And this... works, the first works betters, the latter seems more universal
    # scene_operations.cropPcd(np.array([-radius,-0.5,-radius]),np.array([radius,0,radius]))
    # scene_operations.segment(outliers=True)

    # avg_neighbour_distance = scene_operations.computeAvgNearestNeighborDistance()
    object_pcds = scene_operations.computeClusters(eps=0.020, min_points=1000)
    object_operations = dict()
    # scene_operations.view()

    """ TODO:
        Leitura das poses
        listagem das imagens;
        Transformações;

    """
    for number, object in enumerate(object_pcds):
        object_operations[number] = PointCloudOperations(object)

        # Going back to initial pose for backprojection
        object_operations[number].transGeom(
            rotation=np.array([-z_offset_about_x, 0, 0])
        )
        object_operations[number].transGeom(translation=camera2table_center)
        # ! Not working
        # object_operations[number].associateGT(scene_lobels,scene_label_dict,scene_operations.original_pcd)
        # object_type = object_operations[number].typeGT
        # object_operations[number].savePCD(f"pcd_{object_type}_{number}.pcd", clustered_pcds_path)
        object_operations[number].savePCD(f"pcd_{number}.pcd", clustered_pcds_path)

        # * Compute Axis Aligned BBox's on the camera referential
        object_operations[number].computeImages(img_paths, n_divs=10)

        object_operations[number].computePcdBBox()
        object_operations[number].computeRgbBboxs(img_paths, poses, intrinsics_matrix)

        object_operations[number].computePcdCentroid()
        object_operations[number].computeRGBCentroid(
            img_paths, poses, intrinsics_matrix
        )

        object_operations[number].computeProperties()

        object_operations[number].computeROIs()

        if use_upscale:

            upscaled_images = []
            for img_roi in object_operations[number].rgb_ROIs:
                cv2.imwrite(
                    f'{os.getenv("SAVI_TP2")}/src/lib/ESRGAN_upscaler/LR/1.png', img_roi
                )
                upscaler2()
                roi_image_upscaled = cv2.imread(
                    f'{os.getenv("SAVI_TP2")}/src/lib/ESRGAN_upscaler/results/1_rlt.png'
                )
                upscaled_images.append(roi_image_upscaled)

            object_operations[number].type = predicter.averageGuess(
                upscaled_images, plot=plot_NN
            )
        else:
            object_operations[number].type = predicter.averageGuess(
                object_operations[number].rgb_ROIs, plot=plot_NN
            )

        # object_operations[number].describe()
        # object_operations[number].view()

    # * FINAL DESCRIPTION SHOWCASE HEHEHEHE

    # scene_gui_rgb = cv2.imread(f'{img_path}/00000-color.png')
    scene_gui_rgb = PointCloudOperations.rgb_images[0]

    # Create a figure and subplots

    plt.figure(1)
    # Plot the big image in the center with a title
    plt.imshow(cv2.cvtColor(scene_gui_rgb, cv2.COLOR_BGR2RGB))
    plt.title(f"Scene nº{scene_n}")
    plt.axis("off")

    plt.figure(2)

    objects_in_scene = []
    # Plot the small images on the bottom row
    for i, _ in enumerate(object_pcds):

        plt.subplot(1, 5, i + 1)

        plt.imshow(cv2.cvtColor(object_operations[i].rgb_ROIs[0], cv2.COLOR_BGR2RGB))
        objects_in_scene.append(object_operations[i].type)
        plt.title(f"{objects_in_scene[-1]}")
        plt.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )
        plt.xlabel(
            f"""Color is {''.join(filter(str.isalpha,get_colour_name(object_operations[i].average_color)[1]))
}\n
Dimensions are \n{object_operations[i].dimensions}"""
        )

    speech = f"Here we have scene {scene_n}, and it has the following objects:"

    for object_str in objects_in_scene:
        object_str = "".join(filter(str.isalpha, object_str))
        speech += f"\n {object_str}"

    text2Speech(speech)

    plt.show()


if __name__ == "__main__":
    main()

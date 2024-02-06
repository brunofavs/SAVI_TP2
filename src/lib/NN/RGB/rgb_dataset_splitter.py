#!/usr/bin/env python3


import glob
import json
import os
from sklearn.model_selection import train_test_split


def main():

    # -----------------------------------------------------------------
    # Prepare Datasets
    # -----------------------------------------------------------------
    rgb_data_path = f'{os.getenv("SAVI_TP2")}/dataset/object_dataset/rgbd-dataset'
    image_filenames = glob.glob(rgb_data_path + '/**/*.png',recursive=True)

    image_filenames = [file for file in image_filenames if "crop" in file and "depth" not in file and "mask" not in file]

    items = ["bowl","cap","cereal_box","coffee_mug","soda_can"]
    image_filenames = [file for file in image_filenames if any(item in file for item in items)]
    

    # Use a rule of 70% train, 20% validation, 10% test

    train_filenames, remaining_filenames = train_test_split(image_filenames, test_size=0.3)
    validation_filenames, test_filenames = train_test_split(remaining_filenames, test_size=0.33)

    print('We have a total of ' + str(len(image_filenames)) + ' images.')
    print('Used ' + str(len(train_filenames)) + ' train images')
    print('Used ' + str(len(validation_filenames)) + ' validation images')
    print('Used ' + str(len(test_filenames)) + ' test images')

    d = {'train_filenames': train_filenames,
         'validation_filenames': validation_filenames,
         'test_filenames': test_filenames}

    json_object = json.dumps(d, indent=2)

    # Writing to sample.json

    os.chdir(f'{os.getenv("SAVI_TP2")}/dataset/jsons')
    with open("rgb_images_filenames_mini.json", "w") as outfile:
        outfile.write(json_object)


if __name__ == "__main__":
    main()

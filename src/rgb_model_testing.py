#!/usr/bin/env python3

import json
import os
from numpy import mean
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchmetrics

from torchvision.models import densenet121,DenseNet121_Weights

from lib.NN.RGB.rgb_dataset import DatasetRGB
from lib.NN.RGB.model_architectures.classes_model import Model
from lib.NN.RGB.rgb_trainer import Trainer


def main():

    # -----------------------------------------------------------------
    # Prepare Datasets
    # -----------------------------------------------------------------

    script_dir = os.getcwd()
    os.chdir(f'{os.getenv("SAVI_TP2")}/dataset/jsons')

    with open("rgb_images_filenames_mini.json", "r") as f:
        dataset_filenames = json.load(f)

    with open("rgb_images_matchings_mini.json", "r") as f:
        label_names2idx = json.load(f)

    os.chdir(script_dir)

    test_filenames = dataset_filenames["test_filenames"]
    # test_filenames = test_filenames[0:10]

    print("Used " + str(len(test_filenames)) + " for testing")

    test_dataset = DatasetRGB(test_filenames, label_names2idx)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=1, shuffle=True
    )

    # -----------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------

    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    num_classes = len(label_names2idx)
    print(num_classes)
    # Modify the last fully connected layer (classifier)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    model_path = f'{os.getenv("SAVI_TP2")}/src/lib/NN/RGB/model_architectures/trained_models/densenet121_mini_10_ep.pkl'
    loss = nn.CrossEntropyLoss()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    model.eval()

    # Metric Initialization
    f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes).to(device)
    precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes).to(
        device
    )
    recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes).to(device)
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(
        device
    )
    confusion_matrix = torchmetrics.ConfusionMatrix(
        task="multiclass", num_classes=num_classes
    ).to(device)

    #-------------------------------------
    # Testing sparse images from folder
    #-------------------------------------
    # bp_imgs_path = f'{os.getenv("SAVI_TP2")}/bin/objs/rgb/cropped/'

    # img_bp_filenames = os.listdir(bp_imgs_path)

    # imgs_bp_cropped = []

    # img_transforms = transforms.Compose([
    #         transforms.Resize((224, 224)),
    #         transforms.ToTensor()
    #     ])


    # for image_filename in img_bp_filenames:

    #     img = Image.open(bp_imgs_path + image_filename)
    #     tensor_image = img_transforms(img)

    #     tensor_image = tensor_image.to(device)

    #     labels_predicted = model.forward(tensor_image.unsqueeze(0))

    #     # Apply softmax to convert logits into probabilities
    #     probabilities = F.softmax(labels_predicted, dim=1)

    #     # Get the predicted class label (index of the maximum probability)
    #     predicted_label_index = torch.argmax(probabilities, dim=1)

    #     key_list = list(label_names2idx.keys())
    #     val_list = list(label_names2idx.values())

    #     label_idx2names = dict(zip(val_list,key_list))

    #     predicted_label_name = label_idx2names[predicted_label_index.data.item()]


    #     plt.imshow(img)
    #     plt.title(predicted_label_name)
    #     plt.show()






    # return
    #-------------------------------------
    # Testing test images from dataset
    #-------------------------------------
    batch_losses = []
    for batch_idx, (inputs, labels_gt) in tqdm(
        enumerate(test_loader), total=len(test_loader)
    ):

        # move tensors to device
        # this converts it from GPU to CPU and selects first image
        # convert image back to Height,Width,Channels
        # show the image
        img = inputs.cpu().numpy()[0]
        img = np.transpose(img, (1, 2, 0))

        inputs = inputs.to(device)
        labels_gt = labels_gt.to(device)

        # Get predicted labels
        labels_predicted = model.forward(inputs)

        # Apply softmax to convert logits into probabilities
        probabilities = F.softmax(labels_predicted, dim=1)

        # Get the predicted class label (index of the maximum probability)
        predicted_label_index = torch.argmax(probabilities, dim=1)

        key_list = list(label_names2idx.keys())
        val_list = list(label_names2idx.values())

        label_idx2names = dict(zip(val_list,key_list))

        predicted_label_name = label_idx2names[predicted_label_index.data.item()]

        # Metrics

        # ? Not sure if these are well computed, if .update() or .compute() should be used, since its giving always 1
        f1_score = f1(predicted_label_index, labels_gt)
        precision_score = precision(predicted_label_index, labels_gt)
        recall_score = recall(predicted_label_index, labels_gt)
        accuracy_score = accuracy(predicted_label_index, labels_gt)
        confusion_matrix_score = accuracy.update(predicted_label_index, labels_gt)

        # Compute loss comparing labels_predicted labels
        batch_loss = loss(labels_predicted, labels_gt)
        batch_losses.append(batch_loss.data.item())

        # plt.imshow(img)
        # plt.title(predicted_label_name)
        # plt.show()

    # Compute test validation loss
    test_loss = mean(batch_losses)
    print(f"Test loss is {test_loss}")
    print(f"f1 score is {f1_score}")
    print(f"precision score is {precision_score}")
    print(f"recall score is {recall_score}")
    print(f"accuracy score is {accuracy_score}")

    plt.show()


if __name__ == "__main__":
    main()

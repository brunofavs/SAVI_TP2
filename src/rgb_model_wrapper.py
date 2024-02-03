#!/usr/bin/env python3

import json
import os
from typing import Any
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

import cv2

from lib.NN.RGB.rgb_dataset import DatasetRGB
from lib.NN.RGB.model_architectures.classes_model import Model
from lib.NN.RGB.rgb_trainer import Trainer


class modelWrapper:

    def __init__(
        self,
        model_name="densenet121_full_28ep.pkl",
        matchings_name="rgb_images_matchings.json",
    ):
        # -----------------------------------------------------------------
        # Load matchings dictionary
        # -----------------------------------------------------------------

        script_dir = os.getcwd()
        os.chdir(f'{os.getenv("SAVI_TP2")}/dataset/jsons')

        with open(matchings_name, "r") as f:
            self.label_names2idx = json.load(f)

        key_list = list(self.label_names2idx.keys())
        val_list = list(self.label_names2idx.values())

        self.label_idx2names = dict(zip(val_list, key_list))
        os.chdir(script_dir)

        # -----------------------------------------------------------------
        # Load model
        # -----------------------------------------------------------------

        # self.model = torchvision.models.densenet121(pretrained=True)
        self.model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.num_classes = len(self.label_names2idx)
        # Modify the last fully connected layer (classifier)
        self.model.classifier = nn.Linear(
            self.model.classifier.in_features, self.num_classes
        )

        model_path = f'{os.getenv("SAVI_TP2")}/src/lib/NN/RGB/model_architectures/trained_models/{model_name}'

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)

        self.model.eval()

        self.img_transforms = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )

    def __call__(self, cv_img, plot=False):

        # Converting cv2 image to PIL
        color_coverted = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_coverted)

        # Applying transforms
        tensor_image = self.img_transforms(pil_image)

        # Putting image on GPU
        tensor_image = tensor_image.to(self.device)

        # Predict label into cross entropy frequencies
        labels_predicted = self.model.forward(tensor_image.unsqueeze(0))

        # Apply softmax to convert logits into probabilities
        probabilities = F.softmax(labels_predicted, dim=1)

        # Get the predicted class label (index of the maximum probability)
        predicted_label_index = torch.argmax(probabilities, dim=1)

        # Get the corresponding label name
        predicted_label_name = self.label_idx2names[predicted_label_index.data.item()]

        if plot:
            plt.imshow(pil_image)
            plt.title(predicted_label_name)
            plt.show()

        return predicted_label_name


def main():

    os.chdir(f'{os.getenv("SAVI_TP2")}/bin/objs/rgb/cropped')

    img = cv2.imread("blackmugweb.png")

    predicter = modelWrapper()

    predicter(img, plot=True)


if __name__ == "__main__":
    main()

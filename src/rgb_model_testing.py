#!/usr/bin/env python3

import json
import os
from numpy import mean
import torch
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torchvision
import torch.nn.functional as F

from lib.NN.RGB.rgb_dataset                             import DatasetRGB
from lib.NN.RGB.model_architectures.classes_model       import Model
from lib.NN.RGB.rgb_trainer                             import Trainer


def main():

    # -----------------------------------------------------------------
    # Prepare Datasets
    # -----------------------------------------------------------------

    script_dir = os.getcwd()
    os.chdir(f'{os.getenv("SAVI_TP2")}/dataset/jsons')

    with open('rgb_images_filenames.json', 'r') as f:
        dataset_filenames = json.load(f)

    with open('rgb_images_matchings.json', 'r') as f:
        matching_dict = json.load(f)
    
    os.chdir(script_dir)

    test_filenames = dataset_filenames['test_filenames']
    # test_filenames = test_filenames[0:1000]

    print('Used ' + str(len(test_filenames)) + ' for testing')


    test_dataset = DatasetRGB(test_filenames,matching_dict)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
    # -----------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------
    # model = Model()
    model = torchvision.models.densenet121(pretrained=True)

    # Modify the last fully connected layer (classifier)
    num_classes = len(matching_dict)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    model_path = f'{os.getenv("SAVI_TP2")}/src/lib/NN/RGB/model_architectures/trained_models/densenet121_full_28ep.pkl'
    loss = nn.CrossEntropyLoss()

    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    model.eval()



    # Testing --------------------------------------------
    batch_losses = []
    for batch_idx, (inputs, labels_gt) in tqdm(enumerate(test_loader),
                                            total=len(test_loader)):

        # move tensors to device
        # this converts it from GPU to CPU and selects first image
        #convert image back to Height,Width,Channels
        #show the image
        img = inputs.cpu().numpy()[0]
        img = np.transpose(img, (1,2,0))

        inputs = inputs.to(device)
        labels_gt = labels_gt.to(device)
        

        # Get predicted labels
        labels_predicted = model.forward(inputs)
        
        # Apply softmax to convert logits into probabilities
        probabilities = F.softmax(labels_predicted, dim=1) 

        # Get the predicted class label (index of the maximum probability)
        predicted_label_index = torch.argmax(probabilities, dim=1).data.item()

        key_list = list(matching_dict.keys())
        val_list = list(matching_dict.values())
        
        # print key with val 100
        position = val_list.index(predicted_label_index)
        print(key_list[position])

        predicted_label_name = key_list[position]

        # Compute loss comparing labels_predicted labels
        batch_loss = loss(labels_predicted, labels_gt)

        batch_losses.append(batch_loss.data.item())
        
        # plt.imshow(img)
        # plt.title(predicted_label_name)
        # plt.show()  

    # Compute test validation loss
    test_loss = mean(batch_losses)
    print(test_loss)



    plt.show()


if __name__ == "__main__":
    main()

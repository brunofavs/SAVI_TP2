#!/usr/bin/env python3

import json
import os
import torch
from torchvision import transforms
import torchvision
import torch.nn as nn
from torchvision.models import densenet121,DenseNet121_Weights

import matplotlib.pyplot as plt

from lib.NN.RGB.rgb_dataset                             import DatasetRGB
from lib.NN.RGB.model_architectures.classes_model       import Model
from lib.NN.RGB.rgb_trainer                             import Trainer


def main():

    # -----------------------------------------------------------------
    # Hyperparameters initialization
    # -----------------------------------------------------------------

    hyperparams = {'batch_size':100,
                   'lr' : 0.001,
                   'num_epochs' : 30}

    # -----------------------------------------------------------------
    # Prepare Datasets
    # -----------------------------------------------------------------

    script_dir = os.getcwd()
    os.chdir(f'{os.getenv("SAVI_TP2")}/dataset/jsons')

    # with open('rgb_images_filenames.json', 'r') as f:
    #     dataset_filenames = json.load(f)

    with open('rgb_images_filenames_mini.json', 'r') as f:
        dataset_filenames = json.load(f)

    # try:
    #     with open('rgb_images_matchings.json', 'r') as f:
    #         label_names2idx = json.load(f)
    # except:
    #     label_names2idx = None

    try:
        with open('rgb_images_matchings.json', 'r') as f:
            label_names2idx = json.load(f)
    except:
        label_names2idx = None
    
    os.chdir(script_dir)

    train_filenames = dataset_filenames['train_filenames']
    validation_filenames = dataset_filenames['validation_filenames']

    # train_filenames = train_filenames[0:1000]
    # validation_filenames = validation_filenames[0:200]

    print('Used ' + str(len(train_filenames)) + ' for training and ' + str(len(validation_filenames)) +
          ' for validation.')


    train_dataset = DatasetRGB(train_filenames,label_names2idx)
    validation_dataset = DatasetRGB(validation_filenames,label_names2idx)

    # Try the train_dataset
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=hyperparams['batch_size'], shuffle=True)

    # -----------------------------------------------------------------
    # Create model
    # -----------------------------------------------------------------
    # model = Model()


    # * DensetNet121
    # Load the pre-trained DenseNet121 model
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

    # Modify the last fully connected layer (classifier)
    num_classes = len(label_names2idx)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    # model.classifier = nn.Sequential(nn.Linear(model.classifier.in_features, num_classes),nn.Softmax(dim=1))
    # Freeze pre-trained layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the parameters of the final fully connected layer
    for param in model.classifier.parameters():
        param.requires_grad = True

    # -----------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------
    trainer = Trainer(model=model,
                      train_loader=train_loader,
                      validation_loader=validation_loader,
                      learning_rate=hyperparams['lr'],
                      num_epochs=hyperparams['num_epochs'],
                      model_path=f'{os.getenv("SAVI_TP2")}/src/lib/NN/RGB/model_architectures/trained_models/densenet121_full_30ep.pkl',
                      load_model=True)
    trainer.train()

    plt.show()


if __name__ == "__main__":
    main()

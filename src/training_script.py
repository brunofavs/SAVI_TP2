#!/usr/bin/env python3

import json
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from lib.NN.dataset                     import Dataset
from lib.NN.model_architectures.model   import Model
from lib.NN.trainer                     import Trainer


def main():

    # -----------------------------------------------------------------
    # Hyperparameters initialization
    # -----------------------------------------------------------------
    batch_size = 100
    learning_rate = 0.001
    num_epochs = 50

    hyperparams = {'batch_size':100,
                   'lr' : 0.001,
                   'num_epochs' : 50}

    # -----------------------------------------------------------------
    # Create model
    # -----------------------------------------------------------------
    model = Model()

    # -----------------------------------------------------------------
    # Prepare Datasets
    # -----------------------------------------------------------------

    with open('dataset_filenames.json', 'r') as f:
        dataset_filenames = json.load(f)

    train_filenames = dataset_filenames['train_filenames']
    validation_filenames = dataset_filenames['validation_filenames']

    # train_filenames = train_filenames[0:1000]
    # validation_filenames = validation_filenames[0:200]

    print('Used ' + str(len(train_filenames)) + ' for training and ' + str(len(validation_filenames)) +
          ' for validation.')

    train_dataset = Dataset(train_filenames)
    validation_dataset = Dataset(validation_filenames)

    # Try the train_dataset
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=hyperparams['batch_size'], shuffle=True)

    # -----------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------
    trainer = Trainer(model=model,
                      train_loader=train_loader,
                      validation_loader=validation_loader,
                      learning_rate=hyperparams['lr'],
                      num_epochs=hyperparams['num_epochs'],
                      model_path='./lib/NN/trained_models/checkpoint.pkl',
                      load_model=True)
    trainer.train()

    plt.show()


if __name__ == "__main__":
    main()

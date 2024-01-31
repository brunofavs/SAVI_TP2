#!/usr/bin/env python3


import os
import torch
from torchvision import transforms
from PIL import Image


class Dataset(torch.utils.data.Dataset):

    def __init__(self, filenames):
        self.filenames = filenames
        self.number_of_images = len(self.filenames)

        # Compute the corresponding labels
        self.label_types = set()
        self.image_idx_labels = []
        self.label_matchings = dict()

        for filename in self.filenames:
            basename = os.path.basename(filename)
            blocks = basename.split('_')
            first_numerical_block = next(idx for idx,item in enumerate(blocks) if item.isnumeric())

            label = ''
            for i in range(first_numerical_block):
                label = blocks[i] if not label else f'{label}_{blocks[i]}' # because basename can have 2 words "bell_pepper_3_2_205_crop.png"


            if label in self.label_types:

                corresponding_label_idx = self.label_matchings[label]
                self.image_idx_labels.append(corresponding_label_idx)
                
            else:
                self.label_types.add(label)
                self.label_matchings[label] = len(self.label_types)
                self.image_idx_labels.append(len(self.label_types))
            
        
        print(self.label_matchings)
        print(self.label_types)
        print(self.image_idx_labels)

        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        # must return the size of the data
        return self.number_of_images

    def __getitem__(self, index):
        # Must return the data of the corresponding index

        # Load the image in pil format
        filename = self.filenames[index]
        pil_image = Image.open(filename)

        # Convert to tensor
        tensor_image = self.transforms(pil_image)

        # Get corresponding label
        label = self.image_idx_labels[index]

        return tensor_image, label

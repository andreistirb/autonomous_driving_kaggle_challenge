from collections import defaultdict
import cv2
import json
import logging
import numpy as np
import os
import pandas as pd
import pathlib

import torch
from torch.utils.data import Dataset
from utils import str2coords

# Car dataset class for pytorch models
# instantiate it with a dataframe and a root_path for images
# has a training flag
class CarDataset(Dataset):
    def __init__(self, dataframe, root_path, training=True, transform=None):
        # instantiate pandas dataframe from csv file
        self.dataframe = dataframe

        # get image list from directory
        self.imgs_path = root_path
        self.image_list = os.listdir(self.imgs_path)

        self.training = training
        self.transform = transform
        
    def __getitem__(self, index):

        # get entry from dataframe
        image_id, labels_string = self.dataframe.values[index]
        coords = str2coords(labels_string)

        # print(coords)
        # print(image_id) 

        # open image
        img = cv2.imread(self.imgs_path + image_id + ".jpg")

        # display image - just for testing
        # w, h, _ = img.shape
        # img_resize = cv2.resize(img, (int(0.2 * h), int(0.2 * w)))
        # cv2.imshow("asdas", img_resize)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # process entry from dataframe in order to extract annotations
        return 0


    def __len__(self):
        return len(self.dataframe)
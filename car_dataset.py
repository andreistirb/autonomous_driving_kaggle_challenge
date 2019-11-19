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

class CarDataset(Dataset):
    def __init__(self, csv_file, imgs_path):
        # instantiate pandas dataframe from csv file
        self.dataframe = pd.read_csv(csv_file)

        # get image list from directory
        self.imgs_path = imgs_path
        self.image_list = os.listdir(imgs_path)
        
    def __getitem__(self, index):
        # open image
        # get entry from dataframe
        # process entry from dataframe in order to extract annotations
        return 0


    def __len__(self):
        return len(self.image_list)
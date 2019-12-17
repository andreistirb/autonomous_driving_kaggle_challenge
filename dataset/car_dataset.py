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
from utils.utils import str2coords, get_img_coords, visualize, get_mask_and_regr, preprocess_image

# Car dataset class for pytorch models
# instantiate it with a dataframe and a root_path for images
# has a training flag
class CarDataset(Dataset):
    def __init__(self, dataframe, root_path, camera_matrix, training=True, transform=None):
        # instantiate pandas dataframe from csv file
        self.dataframe = dataframe

        # get image list from directory
        self.imgs_path = root_path
        self.image_list = os.listdir(self.imgs_path)

        self.camera_matrix = camera_matrix

        self.training = training
        self.transform = transform


    # TO-DO: 
    #     - put all the preprocessing on image and labels in transform classes so we can 
    # better control how we create the heatmaps
    def __getitem__(self, index):

        # get entry from dataframe
        image_id, labels_string = self.dataframe.values[index]
        #print("In dataset {}".format(labels_string))

        # process entry from dataframe in order to extract annotations
        # coords = str2coords(labels_string)

        # print(coords)
        # print(image_id) 

        # get image coordinates
        # x_coords, y_coords = get_img_coords(labels_string, self.camera_matrix)
        # print(x_coords)
        # print(y_coords)

        # open image
        img0 = cv2.imread(self.imgs_path + image_id + ".jpg")
        img = preprocess_image(img0)

        # do we need to convert from HWC to CHW because of opencv??? MAJOR ISSUE!!!
        img = np.rollaxis(img, 2, 0)

        mask, regr = get_mask_and_regr(img0, labels_string, self.camera_matrix)
        #print("Regression shape {}".format(regr.shape))

        # convert from HWC to CHW
        regr = np.rollaxis(regr, 2, 0)

        #print("Regression shape after rollaxis {}".format(regr.shape))

        # print(mask.shape)
        # print(regr.shape)

        # display image - just for testing
        # w, h, _ = img.shape

        # for x, y in zip(x_coords, y_coords):
        #    cv2.circle(img0, (int(x), int(y)), 10, (0, 0, 255), -1)

        #img0 = visualize(img0, coords, self.camera_matrix)
        #img0 = cv2.resize(img0, (int(0.2 * h), int(0.2 * w)))
    
        # cv2.imshow("asdas", img)
        # cv2.imwrite("de_test.jpg", img)
        # cv2.imshow("mask", mask)
        # cv2.imshow("regr", regr[:,:,0])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # we need to return x, y, z, yaw, pitch, roll values/heatmap and regression heatmap
        
        
        return img, mask, regr
        #return img, mask, regr, img0

    def __len__(self):
        return len(self.dataframe)

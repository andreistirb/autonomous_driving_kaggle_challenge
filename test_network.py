import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from car_dataset import CarDataset

camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)

if __name__ == "__main__":
    ROOT_PATH = "D:/Datasets/pku-autonomous-driving/"
    df = pd.read_csv(ROOT_PATH + "train.csv")

    train_images_dir = ROOT_PATH + "train_images/"
    test_images_dir = ROOT_PATH + "test_images/"

    df_train, df_val = train_test_split(df, test_size=0.01, random_state=72)

    # create dataset objects
    train_dataset = CarDataset(df_train, train_images_dir, camera_matrix)
    val_dataset = CarDataset(df_val, train_images_dir, camera_matrix)

    print(train_dataset[1])
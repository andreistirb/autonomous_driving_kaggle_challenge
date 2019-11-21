import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from car_dataset import CarDataset


if __name__ == "__main__":
    ROOT_PATH = "D:/Datasets/pku-autonomous-driving/"
    df = pd.read_csv(ROOT_PATH + "train.csv")

    train_images_dir = ROOT_PATH + "train_images/"
    test_images_dir = ROOT_PATH + "test_images/"

    df_train, df_val = train_test_split(df, test_size=0.01, random_state=72)

    # create dataset objects
    train_dataset = CarDataset(df_train, train_images_dir)
    val_dataset = CarDataset(df_val, train_images_dir)

    print(train_dataset[2])
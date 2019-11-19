import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from utils import str2coords

DATASET_PATH = "D:/Datasets/pku-autonomous-driving/"

train_df = pd.read_csv(DATASET_PATH + "train.csv")
#print(train_df["ImageId"])

camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)
camera_matrix_inv = np.linalg.inv(camera_matrix)

# image example from dataset
img = cv2.imread(DATASET_PATH + "train_images/ID_8a6e65317" + ".jpg", cv2.IMREAD_COLOR)
h, w, _ = img.shape
new_h, new_w = int(0.3 * h), int(0.3 * w)
img = cv2.resize(img, (new_w, new_h))

cv2.imshow("img_window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

inp = train_df['PredictionString'][0]
print('Example input:\n', inp)
print()
print('Output:\n', str2coords(inp))
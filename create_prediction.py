import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2 

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim import lr_scheduler

from dataset.car_dataset import CarDataset
from nn.network import MyUNet
from utils.utils import coords2str, extract_coords
from utils.evaluate_map import compute_map
from utils.utils import visualize

from efficientnet_pytorch import EfficientNet

camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)

device = torch.device("cuda")

IMG_WIDTH = 1024
IMG_HEIGHT = IMG_WIDTH // 16 * 5
MODEL_SCALE = 8


if __name__ == "__main__":

    ROOT_PATH = "/media/andreis/storage/datasets/pku-autonomous-driving/"
    df_prediction = pd.read_csv(ROOT_PATH + "sample_submission.csv")
    test_images_dir = ROOT_PATH + "test_images/"
    BATCH_SIZE = 4

    prediction_dataset = CarDataset(df_prediction, test_images_dir, camera_matrix)
    prediction_loader = DataLoader(dataset=prediction_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = MyUNet(10).to(device)

    model.load_state_dict(torch.load("model.pth"))
    predictions = []
    model.eval()
    for img, _, _ in tqdm(prediction_loader):
        with torch.no_grad():
            output = model(img.to(device))
        output = output.data.cpu().numpy()
        for out in output:
            coords = extract_coords(out)
            s = coords2str(coords)
            predictions.append(s)

    prediction_df = pd.read_csv(ROOT_PATH + 'sample_submission.csv')
    prediction_df['PredictionString'] = predictions
    prediction_df.to_csv('predictions.csv', index=False)
    print(prediction_df.head())

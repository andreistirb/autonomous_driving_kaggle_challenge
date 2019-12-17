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
    df = pd.read_csv(ROOT_PATH + "train.csv")
    df_test = pd.read_csv(ROOT_PATH + "sample_submission.csv")

    train_images_dir = ROOT_PATH + "train_images/"
    test_images_dir = ROOT_PATH + "test_images/"

    df_train, df_val = train_test_split(df, test_size=0.01, random_state=72)
    df_val_gt = df_val.copy()

    # create dataset objects
    train_dataset = CarDataset(df_train, train_images_dir, camera_matrix)
    val_dataset = CarDataset(df_val, train_images_dir, camera_matrix)
    test_dataset = CarDataset(df_test, test_images_dir, camera_matrix)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyUNet(10).to(device)

    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4)

    #img, mask, regr = val_dataset[0]

    #output = model(torch.tensor(img[None]).to(device))

    #output = output.data.cpu().numpy()
    
    predictions = []
    for img, _, _, img0 in tqdm(val_loader):
        img_np = np.moveaxis(torch.squeeze(img).numpy(), 0, 2)
        img0 = torch.squeeze(img0).numpy()#p.moveaxis(torch.squeeze(img).numpy(), 0, 2)
        #print(img_np.shape)
        with torch.no_grad():
            #output = model(torch.tensor(img[None]).to(device))
            output = model(img.to(device))
            output = output.data.cpu().numpy()
            # looping over batch items
            for out in output:
                coords = extract_coords(out)
                print(coords)
                # s = coords2str(coords)
                
                #predictions.append(s)
                q_img = visualize(img0, coords, camera_matrix)
                print(q_img.shape)
                q_img = cv2.resize(q_img, (int(q_img.shape[1]*0.25), int(q_img.shape[0]*0.25) ))
                # show predictions on image
                cv2.imshow("Prediction", q_img)
                cv2.waitKey()
                # cv2.imshow("Predictions", visualize(img_np, coords, camera_matrix))
                # cv2.waitKey()


    #df_val['PredictionString'] = predictions
    #df_test.to_csv('predictions.csv', index=False)
    #print(df_val.head())

    #def sigmoid(x):
    #    return 1 / (1 + np.exp(-x))

    #map = compute_map(df_val_gt, df_val)
    #print(map)

    #logits = output[0,0].data.cpu().numpy()
    #sigmoids = np.apply_along_axis(sigmoid, -1, logits)
    #print(output.shape)
    #print(logits.shape)
    #print(sigmoids.shape)
    #print(sigmoids)
    #print(np.max(sigmoids))

    #points = np.argwhere(logits > 0)
    #print(points)
    #preds = extract_coords(output)


    #img = np.rollaxis(img, 0, 3)
    #print(type(img))

    #cv2.imshow("imagine", img)
    #cv2.imshow("mask", mask)
    #cv2.imshow("regr", regr[:,:,-1])
    #cv2.imshow("predictions", sigmoids)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()




    



    
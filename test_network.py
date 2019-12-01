import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim import lr_scheduler

from dataset.car_dataset import CarDataset
from nn.network import MyUNet
from utils.utils import coords2str, extract_coords

from efficientnet_pytorch import EfficientNet

camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)

device = torch.device("cuda")

IMG_WIDTH = 1024
IMG_HEIGHT = IMG_WIDTH // 16 * 5
MODEL_SCALE = 8

def criterion(prediction, mask, regr, size_average=True):
    # Binary mask loss
    pred_mask = torch.sigmoid(prediction[:, 0])
#     mask_loss = mask * (1 - pred_mask)**2 * torch.log(pred_mask + 1e-12) + (1 - mask) * pred_mask**2 * torch.log(1 - pred_mask + 1e-12)
    mask_loss = mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)
    mask_loss = -mask_loss.mean(0).sum()
    
    # Regression L1 loss
    pred_regr = prediction[:, 1:]
    regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
    regr_loss = regr_loss.mean(0)
    
    # Sum
    loss = mask_loss + regr_loss
    if not size_average:
        loss *= prediction.shape[0]
    return loss

def train_model(model, epoch, train_loader, history=None):
    model.train()

    for batch_idx, (img_batch, mask_batch, regr_batch) in enumerate(tqdm(train_loader)):
        img_batch = img_batch.to(device)
        mask_batch = mask_batch.to(device)
        regr_batch = regr_batch.to(device)
        
        optimizer.zero_grad()
        output = model(img_batch)
        loss = criterion(output, mask_batch, regr_batch)
        if history is not None:
            history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()
        
        loss.backward()
        
        optimizer.step()
        exp_lr_scheduler.step()
    
    print('Train Epoch: {} \tLR: {:.6f}\tLoss: {:.6f}'.format(
        epoch,
        optimizer.state_dict()['param_groups'][0]['lr'],
        loss.data))

def evaluate_model(model, epoch, dev_loader, history=None):
    model.eval()
    loss = 0
    
    with torch.no_grad():
        for img_batch, mask_batch, regr_batch in dev_loader:
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)
            regr_batch = regr_batch.to(device)

            output = model(img_batch)

            loss += criterion(output, mask_batch, regr_batch, size_average=False).data
    
    loss /= len(dev_loader.dataset)
    
    if history is not None:
        history.loc[epoch, 'dev_loss'] = loss.cpu().numpy()
    
    print('Dev loss: {:.4f}'.format(loss))

if __name__ == "__main__":

    ROOT_PATH = "/media/andreis/storage/datasets/pku-autonomous-driving/"
    df = pd.read_csv(ROOT_PATH + "train.csv")
    df_test = pd.read_csv(ROOT_PATH + "sample_submission.csv")

    train_images_dir = ROOT_PATH + "train_images/"
    test_images_dir = ROOT_PATH + "test_images/"

    df_train, df_val = train_test_split(df, test_size=0.01, random_state=72)

    # create dataset objects
    train_dataset = CarDataset(df_train, train_images_dir, camera_matrix)
    val_dataset = CarDataset(df_val, train_images_dir, camera_matrix)
    test_dataset = CarDataset(df_test, test_images_dir, camera_matrix)

    img, mask, regr = train_dataset[0]
    # print(img.shape)

    BATCH_SIZE = 4
    train_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    dev_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    #for img, mask, regr in train_loader:
    #    print(img.shape)
    #    exit(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)

    n_epochs = 300

    model = MyUNet(8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=max(n_epochs, 10) * len(train_loader) // 3, gamma=0.1)

    history = pd.DataFrame()

    for epoch in range(n_epochs):
        torch.cuda.empty_cache()
        gc.collect()
        train_model(model, epoch, train_loader, history)
        evaluate_model(model, epoch, dev_loader, history)

    torch.save(model.state_dict(), './model.pth')

    # prepare a submission
    '''
    predictions = []

    test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False, num_workers=4)
    model.eval()
    for img, _, _ in tqdm(test_loader):
        with torch.no_grad():
            output = model(img.to(device))
        output = output.data.cpu().numpy()
        for out in output:
            coords = extract_coords(out)
            s = coords2str(coords)
            predictions.append(s)

    test = pd.read_csv(ROOT_PATH + 'sample_submission.csv')
    test['PredictionString'] = predictions
    test.to_csv('predictions.csv', index=False)
    print(test.head())'''

    # check mAP
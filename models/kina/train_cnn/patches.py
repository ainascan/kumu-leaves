import pandas as pd
import numpy as np
import warnings
import deltalake as dl
import boto3
import os
import time
from sklearn.utils import resample
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch as t
import torchvision as tv
import torch.nn as nn
import mlflow
import multiprocessing
from torchmetrics import AUROC
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import util
import data
import model

warnings.filterwarnings('ignore')


def get_dataset(dir):
    classes = ['healthy', 'defective']
    
    df = []
    for i, cls in enumerate(classes):     
        for file in os.listdir(os.path.join(dir, cls)):
            df.append((os.path.join(dir, cls, file), i))
            
    df = pd.DataFrame(df, columns=['image_path', 'defective'])

    return df


def balance(df):

    df_majority = df[df['defective'] == 1]
    df_minority = df[df['defective'] == 0]

    print(f'Total Defective Annotations: {len(df_majority)}')
    print(f'Total Healthy Annotations: {len(df_minority)}')
    
    if len(df_majority) < len(df_minority):
        df_majority, df_minority = df_minority, df_majority
        
    
    if len(df_majority) < len(df_minority):
        df_majority, df_minority = df_minority, df_majority
        
    df_majority_downsampled = resample(df_majority,
                                        replace=False,
                                        n_samples=len(df_minority),
                                        random_state=seed)

    # combine majority class with upsampled minority class
    df = pd.concat([df_majority_downsampled, df_minority])

    # set column 'sett' to be training, validation, or test
    df['sett'] = np.random.choice(['train', 'val', 'test'], size=len(df), p=[0.7, 0.1, 0.2])
    
    return df


if __name__ == "__main__":
    
    multiprocessing.set_start_method('spawn', force=True)

    images_dir = '/home/jack/Documents/Workspace/kumu/patches'
    mlflow_uri = 'http://127.0.0.1:8081'
    seed = np.random.randint(0, 1000000)
    epochs = 60
    val_epochs = 5
    learning_rate = 0.0008
    batch_size = 32
    
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("Kina Leaf Defect Detection")

    np.random.seed(seed)
    t.manual_seed(seed)
        
    df = get_dataset(images_dir)
    df = balance(df)
    
    print(df.groupby(['sett', 'defective']).size())
    print()

    train_dataset = DataLoader(data.DefectivePatches(df, sett='train'), batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataset = DataLoader(data.DefectivePatches(df, sett='val'), batch_size=batch_size, shuffle=True)
    test_dataset = DataLoader(data.DefectivePatches(df, sett='test'), batch_size=batch_size, shuffle=True)

    model = model.PatchModel()
    model.load_state_dict(t.load('output/model_25.pt', map_location='cuda'))
    model = model.cuda()

    optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = t.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    loss_fn = nn.BCEWithLogitsLoss()

    scaler = t.cuda.amp.GradScaler()
    
    mlflow.start_run(run_name=time.strftime('%Y-%m-%d %H:%M:%S'))
    
    os.makedirs('output', exist_ok=True)

    for file in os.listdir('output'):
        os.remove(os.path.join('output', file))
    
    with t.autograd.set_detect_anomaly(False):
        
        # training
        for epoch in range(epochs):
            
            model.train()
            train_auc = AUROC(task='binary')
            running_loss = 0.0
            progress = tqdm(total=len(train_dataset.dataset), desc='Training', position=0, leave=True)

            for masks, labels in train_dataset:
                masks, labels = masks.cuda(), labels.cuda()

                optimizer.zero_grad()
                with t.cuda.amp.autocast():
                    output = model(masks).squeeze()
                    labels = labels.float()
                    loss = loss_fn(output, labels)
                    train_auc.update(output, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                masks, labels = masks.cpu(), labels.cpu()
                
                running_loss += loss
                progress.set_postfix(loss=f'{loss:.6f}')
                progress.update(len(masks))
                progress.refresh()

            scheduler.step()
            
            train_auc = train_auc.compute()
            epoch_loss = running_loss / (len(train_dataset.dataset) / train_dataset.batch_size)
            progress.set_postfix(loss=f'{epoch_loss:.6f}', epoch=f'{epoch+1}/{epochs}', auc=f'{train_auc:.6f}')
            progress.close()
            
            mlflow.log_metric('train_loss', epoch_loss, step=epoch)
            mlflow.log_metric('train_auc', train_auc, step=epoch)

            # validation
            if epoch > 0 and epoch % val_epochs == 0:

                model.eval()
                running_loss = 0.0
                val_auc = AUROC(task='binary')
                progress = tqdm(total=len(val_dataset.dataset), desc='Validation', position=0, leave=True)
                with t.no_grad():
                    for masks, labels in val_dataset:
                        masks, labels = masks.cuda(), labels.cuda()
            
                        with t.cuda.amp.autocast():
                            output = model(masks).squeeze()
                            labels = labels.float()
                            loss = loss_fn(output, labels)
                            val_auc.update(output, labels)
                        
                        masks, labels = masks.cpu(), labels.cpu()
                        
                        running_loss += loss
                        progress.set_postfix(loss=f'{loss:.6f}')
                        progress.update(len(masks))
                        progress.refresh()
                        
                    val_loss = running_loss / (len(val_dataset.dataset) / val_dataset.batch_size)
                    val_auc = val_auc.compute()
                    progress.set_postfix(loss=f'{val_loss:.6f}', auc=f'{val_auc:.6f}')
                    progress.close()
                    
                    mlflow.log_metric('val_loss', val_loss, step=epoch)
                    mlflow.log_metric('val_auc', val_auc, step=epoch)
            
            t.save(model.state_dict(), os.path.join('output', f'model_{epoch}.pt'))
        
        # test
        model.eval()
        test_auc = AUROC(task='binary')
        running_loss = 0.0
        progress = tqdm(total=len(test_dataset.dataset), desc='Testing', position=0, leave=True)
        with t.no_grad():
            for masks, labels in test_dataset:
                masks, labels = masks.cuda(), labels.cuda()

                with t.cuda.amp.autocast():
                    output = model(masks).squeeze()
                    labels = labels.float()
                    loss = loss_fn(output, labels)
                    test_auc.update(output, labels)

                masks, labels = masks.cpu(), labels.cpu()
                
                running_loss += loss
                progress.set_postfix(loss=f'{loss:.6f}')
                progress.update(len(masks))
                progress.refresh()

            test_loss = running_loss / (len(test_dataset.dataset) / test_dataset.batch_size)
            test_auc = test_auc.compute()
            progress.set_postfix(loss=f'{test_loss:.6f}', auc=f'{test_auc:.6f}')
            progress.close()
            
            mlflow.log_metric('test_loss', test_loss)
            mlflow.log_metric('test_auc', test_auc)
            
        # confusion matrix
        model.eval()
        y_true = []
        y_pred = []
        
        with t.no_grad():
            for masks, labels in test_dataset:
                masks, labels = masks.cuda(), labels.cuda()
                output = model(masks).squeeze()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(t.sigmoid(output).cpu().numpy())
            
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        matrix = confusion_matrix(y_true, y_pred > 0.5)
        
        
        
        class_names = ['Healthy', 'Defective']
        
        sns.set_style('whitegrid')
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('output/confusion_matrix.png')

    t.save(model.state_dict(), os.path.join('output', f'model_final.pt'))
    
    mlflow.log_artifacts('output')
    mlflow.end_run()
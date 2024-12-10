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
from torchmetrics import AUROC
import multiprocessing

import util
from data import KinaDataset
import model

warnings.filterwarnings('ignore')

def get_dataset(storage_options):
    df = dl.DeltaTable(
        table_uri='s3a://coffee-dataset/lake/leaf_annotations',
        storage_options=storage_options
    ).to_pandas()

    annotations = dl.DeltaTable(
        table_uri='s3a://coffee-dataset/lake/raw_annotations',
        storage_options=storage_options
    ).to_pandas()
    
    annotations = annotations[annotations['category_id'] == 'leaf']
    annotations = annotations[annotations['area'] > annotations['area'].quantile(0.05)]
    annotations['contour'] = annotations['segmentation'].apply(lambda x: np.array(x).reshape(-1, 2).astype(np.int32))
    annotations = annotations[annotations['contour'].apply(lambda x: util.is_valid_contour(x))]
    annotations['hash'] = annotations.apply(util.compute_hash, axis=1)
    annotations = annotations[['contour', 'hash', 'image_path']]
    annotations = annotations.merge(df, on='hash', how='left')
    annotations = annotations[['image_path_x', 'contour', 'defective']]
    annotations = annotations.rename(columns={'image_path_x': 'image_path'})
    annotations = annotations[annotations['defective'] != -1]
    annotations['defective'] = annotations['defective'].astype(int)

    return annotations


def balance(df):

    df_majority = df[df['defective'] == 1]
    df_minority = df[df['defective'] == 0]

    print(f'Total Defective Annotations: {len(df_majority)}')
    print(f'Total Healthy Annotations: {len(df_minority)}')
    
    if len(df_majority) < len(df_minority):
        df_majority, df_minority = df_minority, df_majority

    half_size = int(len(df_majority) / 2)
    df_majority_downsampled = resample(df_majority,
                                        replace=False,
                                        n_samples=half_size,
                                        random_state=seed)
    df_minority_upsampled = resample(df_minority,
                                        replace=True,
                                        n_samples=half_size,
                                        random_state=seed)

    # combine majority class with upsampled minority class
    df = pd.concat([df_majority_downsampled, df_minority_upsampled])

    # set column 'sett' to be training, validation, or test
    df['sett'] = np.random.choice(['train', 'val', 'test'], size=len(df), p=[0.7, 0.1, 0.2])
    
    return df


if __name__ == "__main__":
    
    multiprocessing.set_start_method('spawn', force=True)
    
    mlflow_uri = 'http://127.0.0.1:8081'
    seed = 8149385
    epochs = 50
    save_epochs = 20
    val_epochs = 5
    learning_rate = 0.001
    batch_size = 64
    
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("Kina Leaf Defect Detection")

    np.random.seed(seed)
    t.manual_seed(seed)

    session = boto3.Session(profile_name='default')
    credentials = session.get_credentials().get_frozen_credentials()

    storage_options = {
        'AWS_REGION': 'us-west-1',
        'AWS_ACCESS_KEY_ID': credentials.access_key,
        'AWS_SECRET_ACCESS_KEY': credentials.secret_key,
        'AWS_S3_ALLOW_UNSAFE_RENAME': 'true'
    }
        
    df = get_dataset(storage_options)
    df = balance(df)
    
    print(df.groupby(['sett', 'defective']).size())
    print()

    train_dataset = DataLoader(KinaDataset(df, storage_options, sett='train'), batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataset = DataLoader(KinaDataset(df, storage_options, sett='val'), batch_size=batch_size, shuffle=True)
    test_dataset = DataLoader(KinaDataset(df, storage_options, sett='test'), batch_size=batch_size, shuffle=True)

    model = model.MobileNetV3_FPN_Model()

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
            
            train_auc = AUROC(task="binary")
            running_loss = 0.0
            progress = tqdm(total=len(train_dataset.dataset), desc='Training', position=0, leave=True)

            for masks, labels in train_dataset:
                masks, labels = masks.cuda(), labels.cuda()

                optimizer.zero_grad()
                with t.cuda.amp.autocast():
                    output = model(masks)
                    loss = loss_fn(output.squeeze(), labels.float())
                    train_auc.update(output.squeeze(), labels)

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

            # validation
            if epoch > 0 and epoch % val_epochs == 0:

                model.eval()
                val_auc = AUROC(task="binary")
                running_loss = 0.0
                progress = tqdm(total=len(val_dataset.dataset), desc='Validation', position=0, leave=True)
                with t.no_grad():
                    for masks, labels in val_dataset:
                        masks, labels = masks.cuda(), labels.cuda()
            
                        with t.cuda.amp.autocast():
                            output = model(masks)
                            loss = loss_fn(output.squeeze(), labels.float())
                            val_auc.update(output.squeeze(), labels)
                        
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
                    
            if epoch > 0 and epoch % save_epochs == 0:
                path = os.path.join('output', f'model_{epoch}.pt')
                t.save(model.model.state_dict(), path)
        
        # test

        model.eval()
        test_auc = AUROC(task="binary")
        running_loss = 0.0
        progress = tqdm(total=len(test_dataset.dataset), desc='Testing', position=0, leave=True)
        with t.no_grad():
            for masks, labels in test_dataset:
                masks, labels = masks.cuda(), labels.cuda()

                with t.cuda.amp.autocast():
                    output = model(masks)
                    loss = loss_fn(output.squeeze(), labels.float())
                    test_auc.update(output.squeeze(), labels)

                masks, labels = masks.cpu(), labels.cpu()
                
                running_loss += loss
                progress.set_postfix(loss=f'{loss:.6f}')
                progress.update(len(masks))
                progress.refresh()

            test_loss = running_loss / (len(test_dataset.dataset) / test_dataset.batch_size)
            progress.set_postfix(loss=f'{test_loss:.6f}')
            progress.close()
            
            mlflow.log_metric('test_loss', test_loss)

        test_auc = test_auc.compute()
        mlflow.log_metric('test_auc', test_auc)

    path = os.path.join('output', f'model_final.pt')
    t.save(model.model.state_dict(), path)
    
    mlflow.log_artifacts('output')
    mlflow.end_run()
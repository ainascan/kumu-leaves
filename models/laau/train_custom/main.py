import argparse
import yaml
import logging
import random
import time
import os
import json
import gc

os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())

import mlflow
import numpy as np
import torch as t
import cv2
from torch.utils.data import DataLoader

np.bool = np.bool_
np.complex = np.complex_

from data import LaauDataset, LaauBatchSampler
from model import LaauModel


def set_optimizations():
    num_workers = os.cpu_count()

    cv2.setNumThreads(num_workers)

    # set precision for matrix multiplications
    # set precision for convolution operations
    # https://github.com/facebookresearch/detectron2/blob/5b72c27ae39f99db75d43f18fd1312e1ea934e60/detectron2/engine/defaults.py#L176
    t.set_float32_matmul_precision("medium")
    t.backends.cudnn.allow_tf32 = True
    t.backends.cuda.matmul.allow_tf32 = True
    
    # https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
    t.backends.cudnn.benchmark = True
    
    t.jit.enable_onednn_fusion(True)


def load_coco(configuration):
    directory = configuration['coco_dir']
    with open(os.path.join(directory, 'coco.json')) as f:
        coco = json.load(f)

    images = coco['images']
    annotation_map = {}

    for a in coco['annotations']:
        if a['image_id'] not in annotation_map:
            annotation_map[a['image_id']] = []
        annotation_map[a['image_id']].append(a)
        
    # filter out images with no annotations
    images = [i for i in images if i['id'] in annotation_map]
    # remove images that don't have a corresponding file
    images = [i for i in images if os.path.exists(os.path.join(directory, os.path.basename(i['file_name'])))]

    all_indices = np.arange(len(images))
    all_indices = sorted(all_indices, key=lambda x: f"{images[x]['height']},{images[x]['width']}")

    train_indices = np.random.choice(all_indices, int(len(all_indices) * 0.8), replace=False)
    all_indices = np.setdiff1d(all_indices, train_indices)
    test_indices = np.random.choice(all_indices, int(len(all_indices) * 0.5), replace=False)
    val_indices = np.setdiff1d(all_indices, test_indices)
    
    return images, annotation_map, train_indices, val_indices, test_indices


if __name__ == "__main__":
    """
    Train LAAU model
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S'
    )

    parser = argparse.ArgumentParser(description='Train Laau Model')
    parser.add_argument('--config', type=str, default='configuration.yaml', help='Yaml configuration file')
    config = parser.parse_args().__dict__
    
    with open(config['config'], 'r') as file:
        configuration = yaml.safe_load(file)
    
    logging.info(f"Configuration: {configuration}")
    
    # set random seeds
    random.seed(configuration['seed'])
    t.manual_seed(configuration['seed'])
    np.random.seed(configuration['seed'])
    
    set_optimizations()
    
    mlflow.set_tracking_uri(configuration['mlflow']['tracking_uri'])
    mlflow.set_experiment(configuration['mlflow']['experiment_name'])
    
    run_name = time.strftime('%Y-%m-%d %H:%M:%S')
    
    configuration['save_path'] = os.path.join(configuration['save_path'], run_name)
    
    logging.info(f"Starting Experiment: '{configuration['mlflow']['experiment_name']}' run: '{run_name}'")
    
    mlflow.start_run(run_name=run_name)
    mlflow.set_tag("mlflow.note.content", configuration['mlflow']['run_description'])
    for tag in configuration['mlflow']['tags']:
        mlflow.set_tag(tag['name'], tag['value'])

    for key, value in configuration.items():
        mlflow.log_param(key, value)
        
    mlflow.log_param('torch_version', t.__version__)
    mlflow.log_param('numpy_version', np.__version__)
    mlflow.log_param('opencv_version', cv2.__version__)

    collate_fn = lambda batch: tuple(zip(*batch))
    
    images, annotation_map, train_indices, val_indices, test_indices = load_coco(configuration)
    
    train_dataset = LaauDataset(configuration, images, annotation_map, train_indices)
    train_dataset.preload()

    val_dataset = LaauDataset(configuration, images, annotation_map, val_indices)
    val_dataset.preload()
    
    test_dataset = LaauDataset(configuration, images, annotation_map, test_indices)
    test_dataset.preload()
    
    train_loader = DataLoader(
        dataset=train_dataset,
        collate_fn=collate_fn,
        num_workers=0,
        batch_sampler=LaauBatchSampler(
            dataset=train_dataset,
            batch_size=configuration['batch_size'],                       
        ),
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        collate_fn=collate_fn,
        num_workers=0,
        batch_sampler=LaauBatchSampler(
            dataset=val_dataset,
            batch_size=configuration['batch_size'],
        ),
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        collate_fn=collate_fn,
        num_workers=0,
        batch_sampler=LaauBatchSampler(
            dataset=test_dataset,
            batch_size=configuration['batch_size'],
        ),
    )
    
    logging.info(f'Training on {len(train_loader.dataset)} samples')
    logging.info(f'Validating on {len(val_loader.dataset)} samples')
    logging.info(f'Testing on {len(test_loader.dataset)} samples\n')
    
    gc.collect()
    
    model = LaauModel(
        configuration['model_path'],
        configuration['save_path'],
        backbone=configuration['backbone'],
        freeze_at=configuration['freeze_at'],
        anchor_sizes=configuration['anchor_sizes'],
        anchor_ratios=configuration['anchor_ratios'],
        max_size=configuration['max_size'],
        min_size=configuration['min_size'],
        resume=configuration['resume']
    )
    
    model.train(
        train_loader,
        val_loader,
        test_loader,
        save_epochs=configuration['save_epochs'],
        learning_rate=configuration['learning_rate'],
        epochs=configuration['epochs'],
        val_epochs=configuration['val_epochs'],
        iou_threshold=configuration['iou_threshold'],
        confidence_threshold=configuration['confidence_threshold'],
    )
    
    mlflow.end_run()
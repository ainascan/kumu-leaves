import os
import json
import uuid
from hashlib import md5

import numpy as np
import sqlite3

import util


def generate_coco_file(db: sqlite3.Connection, config: dict):
    coco_data = {
        "info": {
            "description": "Compiled Coffee Dataset",
            "config": config
        },
        'images': [],
        'annotations': [],
        'categories': [
            {
                "id": config['category_id'],
                "name": config['category']
            },
        ]
    }
    
    new_images = db.execute(f'SELECT coco_json FROM images WHERE run_id = "{config["run_id"]}"').fetchall()
    new_annotations = db.execute(f'SELECT coco_json FROM annotations WHERE run_id = "{config["run_id"]}"').fetchall()

    new_images = [json.loads(i[0]) for i in new_images]
    new_annotations = [json.loads(a[0]) for a in new_annotations]

    image_id = 0
    annotation_id = 0
    total_corrupted = 0

    annotation_map = {}
    for annotation in new_annotations:
        if annotation['image_id'] not in annotation_map:
            annotation_map[annotation['image_id']] = []
        annotation_map[annotation['image_id']].append(annotation)

    for image in new_images:
        annotations = annotation_map[image['id']]

        # override the image and annotation id again because we need to use numbers for the coco format
        image['id'] = image_id
        
        for annotation in annotations:
            contour = np.array(annotation['segmentation'][0]).reshape(-1, 2)
            if util.is_invalid_contour(contour):
                total_corrupted += 1
                continue
            annotation['id'] = annotation_id
            annotation['image_id'] = image_id
            annotation_id += 1
        
        coco_data['images'].append(image)
        
        for annotation in annotations:
            coco_data['annotations'].append(annotation)
        
        image_id += 1

    print(f'Total Images: {len(coco_data["images"])}')
    print(f'Total Annotations: {len(coco_data["annotations"])}')
    print(f'Total Corrupted: {total_corrupted}\n')

    with open(os.path.join(config['output_dir'], 'coco.json'), 'w') as f:
        json.dump(coco_data, f)
        
    train_data, test_data, validation_data = train_test_split(coco_data)
    
    with open(os.path.join(config['output_dir'], 'train.json'), 'w') as f:
        json.dump(train_data, f)

    with open(os.path.join(config['output_dir'], 'test.json'), 'w') as f:
        json.dump(test_data, f)
    
    with open(os.path.join(config['output_dir'], 'validation.json'), 'w') as f:
        json.dump(validation_data, f)


def train_test_split(coco_data):
    train_split = 0.8
    test_split = 0.1
    val_split = 0.1

    train_data = {
        'images': [],
        'annotations': [],
        'categories': []
    }

    test_data = {
        'images': [],
        'annotations': [],
        'categories': []
    }

    validation_data = {
        'images': [],
        'annotations': [],
        'categories': []
    }

    train_data['categories'] = coco_data['categories']
    test_data['categories'] = coco_data['categories']
    validation_data['categories'] = coco_data['categories']
    
    grouped_default = []
    grouped_augmented = []

    mapped_annotations = {}
    for annotation in coco_data['annotations']:
        if annotation['image_id'] not in mapped_annotations:
            mapped_annotations[annotation['image_id']] = []
        mapped_annotations[annotation['image_id']].append(annotation)

    for image in coco_data['images']:
        annotations = mapped_annotations.get(image['id'])
        if annotations is not None:
            for annotation in annotations:
                if annotation.get('extras', {}).get('augmented') is True:
                    grouped_augmented.append((image, annotation))
                else:
                    grouped_default.append((image, annotation))
                
    np.random.shuffle(grouped_augmented)
    np.random.shuffle(grouped_default)
    
    print(f'Default Annotations: {len(grouped_default)}')
    print(f'Augmented Annotations: {len(grouped_augmented)}')
    
    total_annotations = len(grouped_default) + len(grouped_augmented)

    training_size = int(total_annotations * train_split)
    test_size = int(total_annotations * test_split)
    validation_size = int(total_annotations * val_split)
    
    all_grouped = grouped_augmented + grouped_default

    # select all augmented and some default for training
    training_set = all_grouped[:training_size] 

    # slice = max(0, training_size - len(grouped_augmented))
    # training_set += grouped_default[:slice]

    # select the rest for validation
    slice = training_size
    validation_set = all_grouped[slice:slice + validation_size]

    # select some default for test
    slice = slice + validation_size
    test_set = all_grouped[slice:slice + test_size]
    
    for image, annotation in training_set:
        train_data['images'].append(image)
        train_data['annotations'].append(annotation)

    for image, annotation in test_set:
        test_data['images'].append(image)
        test_data['annotations'].append(annotation)

    for image, annotation in validation_set:
        validation_data['images'].append(image)
        validation_data['annotations'].append(annotation)
        
    # remove duplicate images
    train_data['images'] = list({v['id']:v for v in train_data['images']}.values())
    test_data['images'] = list({v['id']:v for v in test_data['images']}.values())
    validation_data['images'] = list({v['id']:v for v in validation_data['images']}.values())
    
    print(f'Train Images: {len(train_data["images"])}, Annotations: {len(train_data["annotations"])}')
    print(f'Test Images: {len(test_data["images"])}, Annotations: {len(test_data["annotations"])}')
    print(f'Validation Images: {len(validation_data["images"])}, Annotations: {len(validation_data["annotations"])}')
        
    return train_data, test_data, validation_data
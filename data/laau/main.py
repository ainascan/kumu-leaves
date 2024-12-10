import argparse
import os
import warnings
import psutil
import uuid
import json
import pprint
import tempfile

import boto3
import numpy as np
import sqlite3
from tqdm import tqdm
import cv2
import imgaug.augmenters as A
import deltalake as dl
import s3fs

np.bool = np.bool_
np.complex = np.complex_

import coco
import masks
import util
from augment import apply_regular_augmentations


def get_aws_credentials():
    session = boto3.Session(profile_name='default')
    credentials = session.get_credentials()
    credentials = credentials.get_frozen_credentials()

    storage_options = {
        'AWS_REGION': 'us-west-1',
        'AWS_ACCESS_KEY_ID': credentials.access_key,
        'AWS_SECRET_ACCESS_KEY': credentials.secret_key,
        'AWS_S3_ALLOW_UNSAFE_RENAME': 'true'
    }
    
    return storage_options


def get_sqlite_connection():
    db = sqlite3.connect('annotations.db')
    
    # create table images if it does not exist
    db.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id TEXT PRIMARY KEY,
            file_path TEXT,
            run_id TEXT,
            coco_json TEXT
        )
    ''')
    
    db.execute('''
        CREATE TABLE IF NOT EXISTS annotations (
            id TEXT PRIMARY KEY,
            image_id TEXT,
            run_id TEXT,
            coco_json TEXT,
            FOREIGN KEY (image_id) REFERENCES images (id)
        )
    ''')
    
    db.commit()
    
    return db


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2



if __name__ == "__main__":
    """
    python3 main.py \
        --output_dir /home/jack/Mounts/DiskOne/kona_coffee/datasets/compiled_v25 \
        --num_regular_augs 6 \
    """

    parser = argparse.ArgumentParser(description='Generate LAAU dataset')

    #parser.add_argument('--root_dir', type=str, help='Root directory to augment data from')
    parser.add_argument('--output_dir', type=str, help='Output directory to save augmented data')
    #parser.add_argument('--backdrop_dir', type=str, help='Directory to load backdrops from')
    #parser.add_argument('--backdrop_amount', type=int, default=100, help='Number of backdrops to load')
    parser.add_argument('--numpy_seed', type=int, default=102030303, help='Seed for numpy random generator')
    parser.add_argument('--category', type=str, default='leaf', help='Category to augment')
    parser.add_argument('--target_width', type=int, default=1080, help='Target width for images')
    parser.add_argument('--target_height', type=int, default=1920, help='Target height for images')
    parser.add_argument('--num_regular_augs', type=int, default=3, help='Number of regular augmentations to perform')
    
    #parser.add_argument('--num_regular_cutpaste_augs', type=int, default=5, help='Number of phy-cut-paste augmentations to perform')
    #parser.add_argument('--num_abstract_cutpaste_augs', type=int, default=100, help='Number of abstract phy-cut-paste augmentations to generate')
    #parser.add_argument('--cutpaste_clusters', type=int, default=10, help='Number of clusters to use for phy-cut-paste')
    #parser.add_argument('--cutpaste_mask_amount', type=int, default=2500, help='Number of masks to load for phy-cut-paste')
    
    config = parser.parse_args().__dict__
    
    config['run_id'] = str(uuid.uuid1())
    
    warnings.filterwarnings("ignore")
    
    storage_options = get_aws_credentials()
    
    s3 = s3fs.S3FileSystem(
        anon=False,
        use_ssl=False,
        key=storage_options['AWS_ACCESS_KEY_ID'],
        secret=storage_options['AWS_SECRET_ACCESS_KEY'],
        client_kwargs={
            'region_name': storage_options['AWS_REGION']
        }
    )
    
    np.random.seed(config['numpy_seed'])
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # image_map, annotation_map, category_map = coco.load_coco_annotations(config)
    
    category_map = {
        'leaf': 1,
    }
    
    config['category_id'] = category_map[config['category']]
    
    print('')
    pprint.pprint(config)
    print('')
    
    if os.listdir(config['output_dir']):
        raise ValueError('Output directory is not empty. Please provide an empty directory.')
    
    annotations = dl.DeltaTable(
        table_uri='s3a://coffee-dataset/lake/raw_annotations',
        storage_options=storage_options
    ).to_pandas()
    
    print(f'Number of Annotations Discovered: {len(annotations)}')

    annotations = annotations[annotations['category_id'] == config['category']]
    
    print(f'Number of Selected Annotations: {len(annotations)}')
    
    annotations['contour'] = annotations['segmentation'].apply(lambda x: np.array(x).reshape(-1, 2))
    annotations = annotations[annotations['contour'].apply(lambda x: not util.is_invalid_contour(x))]
    
    print(f'Number of Valid Annotations: {len(annotations)}')

    # mask_cache = masks.MaskCache(image_map, annotation_map, category_map)
    
    # mask_cache.load_cluster_model(config['category'], clusters=config['cutpaste_clusters'])
    # mask_cache.load_masks(config['category'], amount=config['cutpaste_mask_amount'])
    # mask_cache.load_backdrops(config['backdrop_dir'], amount=config['backdrop_amount'])
    
    # print('')
    # print('Mask Clustering Value Counts:')
    # print(mask_cache.mask_df.cluster.value_counts())
    # print('')

    db = get_sqlite_connection()
    
    transformer = A.Sequential([
        A.Affine(
            rotate=(-10, 10),
            scale=(0.9, 1.1),
            shear=(-16, 16),
            mode=["wrap"]
        ),
        A.AverageBlur(k=(2, 7)),
        A.Fliplr(0.5),
        A.AddToHueAndSaturation((-20, 20)),
        A.LinearContrast((0.8, 1.6)),
        A.JpegCompression(compression=(70, 99)),
        A.Sometimes(0.3, A.Grayscale(alpha=(0.0, 1.0))),
        A.Sometimes(0.4, A.OneOf([
            A.Snowflakes(),
            A.Rain(),
            #A.Fog(),
            #A.Clouds(),
        ])),
    ])
    
    loader = tqdm(total=len(annotations.image_path.unique()), desc='Applying Augmentation to Images...')

    for image_path, group in annotations.groupby('image_path'):
        
        if len(group) <= 0:
            continue

        with tempfile.NamedTemporaryFile() as f:
            with s3.open(image_path, 'rb') as g:
                f.write(g.read())
            f.seek(0)
            original_frame = cv2.imread(f.name)

        image_name = os.path.basename(image_path)
        new_image_id = str(uuid.uuid1())
        new_file_name = f"{image_name.split('.')[0]}.jpg"
        new_file_path = os.path.join(config['output_dir'], new_file_name)
        
        contours = group['contour'].values
        resized_frame, resized_contours = util.resize_frame_and_contours(original_frame, contours, config['target_width'], config['target_height'])

        image_row = (new_image_id, new_file_path, config['run_id'], json.dumps({
            'id': new_image_id,
            'file_name': new_file_name,
            'height': resized_frame.shape[0],
            'width': resized_frame.shape[1],
            'extras': {
                'augmented': False,
            },
        }))
        
        annotation_rows = [
            (str(uuid.uuid1()), new_image_id, config['run_id'], json.dumps({
                'id': str(uuid.uuid1()),
                'image_id': new_image_id,
                'category_id': config['category_id'],
                'segmentation': [contour.flatten().tolist()],
                'area': cv2.contourArea(contour.astype(np.int32)),
                'bbox': cv2.boundingRect(contour.astype(np.int32)),
                'iscrowd': 0,
                'extras': {
                    'augmented': False,
                },
            }))
            for contour in resized_contours
        ]
        
        cv2.imwrite(new_file_path, resized_frame)
        db.execute('INSERT INTO images VALUES (?, ?, ?, ?)', image_row)
        db.executemany('INSERT INTO annotations VALUES (?, ?, ?, ?)', annotation_rows)
        db.commit()

        augment_iter = apply_regular_augmentations(resized_frame, resized_contours, transformer, config)
        for file_path, augmented_image, image_row, annotation_rows in augment_iter:
            cv2.imwrite(file_path, augmented_image)
            db.execute('INSERT INTO images VALUES (?, ?, ?, ?)', image_row)
            db.executemany('INSERT INTO annotations VALUES (?, ?, ?, ?)', annotation_rows)
            db.commit()
        
        # augment_iter = apply_cutpaste_augmentations(resized_frame, resized_contours, mask_cache, config)
        # for file_path, augmented_image, image_row, annotation_rows in augment_iter:
        #     cv2.imwrite(file_path, augmented_image)
        #     db.execute('INSERT INTO images VALUES (?, ?, ?, ?)', image_row)
        #     db.executemany('INSERT INTO annotations VALUES (?, ?, ?, ?)', annotation_rows)
        #     db.commit()

        loader.update(1)
    
    loader.close()

    coco.generate_coco_file(db, config)
import argparse
import os
import warnings
import psutil
import uuid
import json
import pprint
import tempfile
from collections import OrderedDict

from PIL import Image, ImageOps
import boto3
import numpy as np
import pandas as pd
import cv2
import deltalake as dl
import s3fs

np.bool = np.bool_
np.complex = np.complex_

import util

os.environ['AWS_EC2_METADATA_DISABLED'] = 'true'


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


def get_annotations(storage_options):
    annotations = dl.DeltaTable(
        table_uri='s3a://coffee-dataset/lake/raw_annotations',
        storage_options=storage_options
    ).to_pandas()
    
    print('Total Contour Annotations:', len(annotations))

    try:
        patch_annos = dl.DeltaTable(
            table_uri='s3a://coffee-dataset/lake/clear_leaf_patch_annotations',
            storage_options=storage_options
        ).to_pandas()
    except:
        patch_annos = pd.DataFrame(columns=['image_path', 'patch', 'defective', 'hash'])

    print('Total Patch Annotations:', len(patch_annos))
    print('Total Defective Patches:', len(patch_annos[patch_annos['defective'] == 1]))
    print('Total Healthy Patches:', len(patch_annos[patch_annos['defective'] == 0]))

    annotations = annotations[annotations['category_id'] == 'leaf']
    annotations = annotations[annotations['area'] > annotations['area'].quantile(0.05)]
    annotations['contour'] = annotations['segmentation'].apply(lambda x: np.array(x).reshape(-1, 2).astype(np.int32))
    annotations = annotations[annotations['contour'].apply(lambda x: not util.is_invalid_contour(x))]
    annotations = annotations[['contour', 'image_path']]
    annotations['hash'] = annotations.apply(util.compute_hash, axis=1)
    
    annotations.reset_index(drop=True, inplace=True)
    patch_annos.reset_index(drop=True, inplace=True)
    
    # remove any annotations that are already in the patch_annos
    #annotations = annotations[annotations['hash'].isin(patch_annos['hash'])]
    
    # shuffle the annotations
    annotations = annotations.sample(frac=1).reset_index(drop=True)

    return annotations, patch_annos


def load_image(s3, image_path):
    with s3.open(image_path, 'rb') as g:
        image = Image.open(g)
        image = ImageOps.exif_transpose(image)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
    return image


if __name__ == "__main__":
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
        
    annotations, patch_annos = get_annotations(storage_options)
    image_cache = OrderedDict()
    masked_frame = None
    index = 0
    new_patches = []


    def get_masked_frame(index):
        annotation = annotations.iloc[index]
        mask_patches = patch_annos[patch_annos['hash'] == annotation['hash']]
        
        print(f'Image Path: {annotation["image_path"]}')
        print(f'Total Patches: {len(mask_patches)}')

        if annotation['image_path'] in image_cache:
            frame = image_cache[annotation['image_path']]
        else:
            frame = load_image(s3, annotation['image_path'])
            image_cache[annotation['image_path']] = frame

            if len(image_cache) > 100:
                image_cache.popitem(last=False)

        mask = util.create_masked_frame(frame, annotation['contour'])
        
        if not mask_patches.empty:
            for _, row in mask_patches.iterrows():
                x1, y1, x2, y2 = row['patch']
                if row['defective']:
                    cv2.rectangle(mask, (x1, y1), (x2, y2), (0, 0, 255), 1)
                else:
                    cv2.rectangle(mask, (x1, y1), (x2, y2), (0, 255, 0), 1)

        return mask



    def on_mouse_click(event, x, y, flags, param):
        annotation = annotations.iloc[index]
        nx, ny = x - 32, y - 32
        patch = [nx, ny, nx + 64, ny + 64]

        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.rectangle(masked_frame, (nx, ny), (nx + 64, ny + 64), (0, 0, 255), 1)
            new_patches.append({
                'image_path': annotation['image_path'],
                'hash': annotation['hash'],
                'defective': 1,
                'patch': patch,
            })
        elif event == cv2.EVENT_RBUTTONDOWN:
            cv2.rectangle(masked_frame, (nx, ny), (nx + 64, ny + 64), (0, 255, 0), 1)
            new_patches.append({
                'image_path': annotation['image_path'],
                'hash': annotation['hash'],
                'defective': 0,
                'patch': patch,
            })


    masked_frame = get_masked_frame(index)
    
    cv2.namedWindow('frame', cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback('frame', on_mouse_click)

    while True:
        cv2.imshow('frame', masked_frame)

        # update title
        cv2.setWindowTitle('frame', f'{index}/{len(annotations)}')
        
        key = cv2.waitKey(20) & 0xFF
        
        # quit
        if key == ord('q'):
            break
        
        # remove any patch from the current image
        elif key == ord('r'):
            annotation = annotations.iloc[index]
            image_path, hash = annotation['image_path'], annotation['hash']

            # remove any patches that match the current patch image path and hash
            new_patches[:] = [patch for patch in new_patches if not (patch['image_path'] == image_path and patch['hash'] == hash)]
            # remove any from patch_annos
            patch_annos.drop(patch_annos[(patch_annos['image_path'] == image_path) & (patch_annos['hash'] == hash)].index, inplace=True)
            # reset the masked frame
            masked_frame = get_masked_frame(index)
    
        elif key == ord('a'):
            index -= 1
            if index < 0:
                index = 0

            masked_frame = get_masked_frame(index)
            
        elif key == ord('d'):
            index += 1
            if index == len(annotations):
                index = len(annotations) - 1
                
            masked_frame = get_masked_frame(index)
        
        if index == len(annotations):
            break
        
    cv2.destroyAllWindows()
    
    new_patches = pd.DataFrame(new_patches)
    
    # combine the new patches with the old patches
    new_patches = pd.concat([patch_annos, new_patches], ignore_index=True)
    
    dl.write_deltalake(
        table_or_uri='s3a://coffee-dataset/lake/clear_leaf_patch_annotations',
        data=new_patches,
        mode='overwrite',
        storage_options=storage_options,
        custom_metadata={
            'catalog_name': 'Leaf Patch Annotations',
            'catalog_description': 'This contains patches of 64x64 pixels from the leafs where there is defects or no defects.',
        }
    )
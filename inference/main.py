import argparse
import logging
import os
import time

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
import torch
import cv2
import numpy as np
from tqdm import tqdm
import deltalake as dl
from deltalake.exceptions import TableNotFoundError
import boto3
import s3fs
import pandas as pd

import util
import laau
import kina

os.environ['AWS_EC2_METADATA_DISABLED'] = 'true'

def load_credentials(config):
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


def load_laau_model(config):
    cfg = get_cfg()
    cfg.merge_from_file(config['laau_config_path'])
    cfg.MODEL.DEVICE = 'cuda' if config['gpu'] else 'cpu'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config['laau_score_threshold']
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = config['laau_nms_threshold']
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = config['laau_model_path']
    
    laau_model = DefaultPredictor(cfg)

    return laau_model


def load_kina_model(config):
    kina_model = kina.Model()
    map_location = 'cuda' if config['gpu'] else 'cpu'
    kina_model.load_state_dict(torch.load(config['kina_model_path'], map_location=map_location))
    kina_model = kina_model.cuda() if config['gpu'] else kina_model.cpu()
    kina_model.eval()
    return kina_model


def load_images(config):
    input_table = config['input_table']
    output_table = config['output_table']
    storage_options = config['storage_options']
    
    input_df = dl.DeltaTable(
        table_uri=input_table,
        storage_options=storage_options
    ).to_pandas(columns=['image_path'])
    
    try:
        output_df = dl.DeltaTable(
            table_uri=output_table,
            storage_options=storage_options
        ).to_pandas(columns=['image_path'])
    except TableNotFoundError:
        output_df = pd.DataFrame(columns=['image_path'])
    
    # remove images that are in the output table
    input_df = input_df[~input_df['image_path'].isin(output_df['image_path'])]
    
    # only if "mountain_thunder" is in the image path
    input_df = input_df[input_df['image_path'].str.contains('mountain_thunder')]
    
    return input_df
    


if __name__ == "__main__":
    """
    python3 main.py \
        --input_table 's3a://coffee-dataset/lake/raw_images_v2' \
        --output_table 's3a://coffee-dataset/lake/alpha_bronze_pipeline_results' \
        --laau_model_path ./models/model_final.pth \
        --laau_config_path ./models/model_config.yaml \
        --kina_model_path ./models/kina_25.pt \
        --laau_score_threshold 0.7 \
        --laau_nms_threshold 0.5 \
        --kina_score_threshold 0.5 \
        --kina_blur 63 \
        --kina_stride 16 \
        --debug=true
    """
    
    parser = argparse.ArgumentParser(description='Kumu Pipeline')
    parser.add_argument('--debug', type=bool, default=False, help='Debug mode')
    parser.add_argument('--input_table', type=str, help='Input table with images')
    parser.add_argument('--output_table', type=str, help='Output table with results')
    parser.add_argument('--laau_score_threshold', type=float, default=0.7, help='Laau Score/Confidence Threshold')
    parser.add_argument('--laau_nms_threshold', type=float, default=0.5, help='Laau NMS threshold')
    parser.add_argument('--kina_score_threshold', type=float, default=0.5, help='Kina Score/Confidence Threshold')
    parser.add_argument('--kina_stride', type=int, default=8, help='kina stride')
    parser.add_argument('--kina_blur', type=int, default=63, help='kina blur kernel size')
    parser.add_argument('--laau_model_path', type=str, help='Laau model path')
    parser.add_argument('--laau_config_path', type=str, help='Laau config path')
    parser.add_argument('--kina_model_path', type=str, help='kina model path')
    
    config = parser.parse_args().__dict__
    
    logging.basicConfig(level=logging.DEBUG if config['debug'] else logging.INFO)
    setup_logger()
    
    config['gpu'] = torch.cuda.is_available()
    config['storage_options'] = load_credentials(config)
    
    logging.info(f'Config: {config}')
    
    os.makedirs('output', exist_ok=True)
    
    s3 = s3fs.S3FileSystem(
        anon=False,
        use_ssl=False,
        key=config['storage_options']['AWS_ACCESS_KEY_ID'],
        secret=config['storage_options']['AWS_SECRET_ACCESS_KEY'],
        client_kwargs={
            'region_name': config['storage_options']['AWS_REGION']
        }
    )
    
    laau_model = load_laau_model(config)
    kina_model = load_kina_model(config)
    
    logging.info('Models loaded')
    
    image_paths = load_images(config)

    logging.info(f'Found {len(image_paths)} new images to process')
    
    for row in image_paths.itertuples():
        start_time = time.perf_counter()
        
        torch.cuda.empty_cache()
        
        image_path = row.image_path
        image_name = os.path.basename(image_path)
        
        try:
            with s3.open(image_path, 'rb') as f:
                image = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            logging.error(f'Error reading {image_name}: {e}')
            continue
        
        original = image.copy()

        logging.info(f'Processing {image_name} with shape {image.shape}')
        
        resized = util.resize_shorted_edge(image, 1080, 1920)
        
        results = laau.inference(laau_model, resized)
        
        logging.info(f'Segmented {len(results)} leaves')
        
        if len(results) == 0:
            continue
        
        compiled = np.zeros_like(original)
        
        loader = tqdm(total=len(results), desc='Processing Kina Masks')
        
        final_results = []
        
        try:

            for result in kina.inference(kina_model, original, results, config):
                bbox, heatmap, mask = result['bbox'], result['heatmap'], result['mask']
                
                masked_frame = cv2.bitwise_and(original, original, mask=mask)
                masked_frame = cv2.addWeighted(masked_frame, 0.6, heatmap, 0.4, 0)
                compiled = cv2.bitwise_or(compiled, masked_frame)
                
                final_results.append(result)
                
                loader.update(1)
                torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f'Error processing {image_name}: {e}')
            loader.close()
            continue

        loader.close()
        
        binary_mask = cv2.cvtColor(compiled, cv2.COLOR_BGR2GRAY)
        binary_mask = cv2.threshold(binary_mask, 0, 255, cv2.THRESH_BINARY)[1]
        background = cv2.bitwise_and(original, original, mask=cv2.bitwise_not(binary_mask))
        final = cv2.bitwise_or(compiled, background)
        
        # draw the percentage of the defects in the middle of the bounding box
        for result in final_results:
            bbox = result['bbox']
            percentage = result['defective_percentage'] * 100
            
            cv2.rectangle(final, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 1)
            
            cv2.putText(final, f'{percentage:.2f}%', (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.imwrite(f'output/{image_name}', final)
        
        output = [
            {
                'image_path': image_path,
                'laau_confidence': result['confidence'],
                'laau_bbox': result['bbox'],
                'laau_contour': result['contour'].flatten().tolist(),
                'kina_mask_shape': result['segmentation_mask'].shape,
                'kina_segmentation_mask': result['segmentation_mask'].flatten().tolist(),
                'kina_defective_percentage': result['defective_percentage'],
            }
            for result in final_results
        ]
        
        dl.write_deltalake(
            table_or_uri=config['output_table'],
            data=pd.DataFrame(output),
            mode='append',
            storage_options=config['storage_options'],
            custom_metadata={
                'catalog_name': 'Alpha Bronze Pipeline Results',
                'catalog_description': 'The alpha release base results of the full ainascan pipeline',
            }
        )
        
        logging.info(f'Processed {image_name} in {time.perf_counter() - start_time:.2f}s')
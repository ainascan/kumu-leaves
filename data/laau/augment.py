import os
import json
import uuid
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import phy_cut_paste
from tqdm import tqdm
from imgaug.augmentables.polys import Polygon
import shapely as sh


def fill_with_masks(frame, mask_cache, min_amount=20, max_amount=50):
    mask_amount = np.random.randint(min_amount, max_amount)
    total_frame_area = frame.shape[0] * frame.shape[1]
    used_frame_area = 0
    
    masks = []
    for _ in range(mask_amount):
        mask = mask_cache.fetch_mask()
        # if mask dims are even 70% of the frame, ignore
        if mask.shape[0] > 0.7 * frame.shape[0] or mask.shape[1] > 0.7 * frame.shape[1]:
            continue

        area = np.count_nonzero(mask)
        used = used_frame_area + area
        if used / total_frame_area > 0.75:
            break
        used_frame_area = used
        masks.append(mask)

    return masks


def clip_out_contours(frame_shape, poly):
    if not poly.is_valid:
        return None
    
    poly = poly.to_shapely_polygon()
    clipped = sh.clip_by_rect(poly, 0, 0, frame_shape[1], frame_shape[0])
    
    # if area is less than 30% of original, skip
    if clipped.area < 0.3 * poly.area:
        return None
    
    if isinstance(clipped, sh.geometry.MultiPolygon):
        for geom in clipped.geoms:
            if geom.is_valid and geom.area > 6000:
                return np.array(geom.exterior.coords).astype(np.int32)

    elif clipped.is_valid and clipped.area > 6000:
        return np.array(clipped.exterior.coords).astype(np.int32)


def apply_augmentation(frame, polygons, transformer):
    frame_aug, polygons_aug = transformer(image=frame, polygons=polygons)
    
    pool = ThreadPoolExecutor(max_workers=len(polygons_aug))
    futures = []
    
    for poly in polygons_aug:
        futures.append(pool.submit(clip_out_contours, frame_aug.shape, poly))
        
    contours = []
    for future in as_completed(futures):
        result = future.result()
        if result is not None:
            contours.append(result)
    
    return frame_aug, contours


def apply_abstract_cutpaste_augmentations(mask_cache, transformer, config):

    loader = tqdm(total=config['num_abstract_cutpaste_augs'], desc='Applying Abstract Cut-And-Paste to Images...', position=0, leave=True)
    
    for _ in range(config['num_abstract_cutpaste_augs']):
        image_id = str(uuid.uuid1())
        aug_file_name = f"abstract_{image_id}.cpaug.jpg"
        file_path = os.path.join(config['output_dir'], aug_file_name)

        backdrop = mask_cache.fetch_backdrop()
        masks = fill_with_masks(backdrop, mask_cache)
        
        original_total_contours = len(masks)

        augmented_frame, augmented_contours = phy_cut_paste.simulate_masks(
            masks=masks,
            backdrop=backdrop,
            strict=False,
            threads=16,
            iterations=2
        )

        # don't save if there are no contours
        if len(augmented_contours) == 0:
            continue
        
        augmented_contours = [Polygon(contour) for contour in augmented_contours]
        augmented_frame, augmented_contours = apply_augmentation(augmented_frame, augmented_contours, transformer)
        
        loader.set_postfix(augmented=len(augmented_contours), original=original_total_contours)

        image_row = (image_id, file_path, config['run_id'], json.dumps({
            'id': image_id,
            'file_name': aug_file_name,
            'height': augmented_frame.shape[0],
            'width': augmented_frame.shape[1],
            'extras': {
                'augmented': True,
            },
        }))
        
        annotation_rows = [
            (str(uuid.uuid1()), image_id, config['run_id'], json.dumps({
                'id': str(uuid.uuid1()),
                'image_id': image_id,
                'category_id': config['category_id'],
                'segmentation': [np.round(contour, 3).flatten().tolist()],
                'area': cv2.contourArea(contour.astype(np.int32)),
                'bbox': cv2.boundingRect(contour.astype(np.int32)),
                'iscrowd': 0,
                'extras': {
                    'augmented': True,
                },
            }))
            for contour in augmented_contours
        ]
        
        yield file_path, augmented_frame, image_row, annotation_rows
        
        loader.update(1)


def apply_cutpaste_augmentations(frame, contours, mask_cache, config):
    total_frame_area = frame.shape[0] * frame.shape[1]
    used_frame_area = sum([cv2.contourArea(contour.astype(np.int32)) for contour in contours])
    
    if used_frame_area / total_frame_area > 0.75:
        # ignore, no more space left in the frame
        return []
    
    boxes = [cv2.boundingRect(c.astype(np.int32)) for c in contours]
    avg_contour_width = np.mean([b[2] for b in boxes])
    avg_contour_height = np.mean([b[3] for b in boxes])
    
    for i in range(config['num_regular_cutpaste_augs']):
        image_id = str(uuid.uuid1())
        aug_file_name = f"{image_id}_{i}.cpoaug.jpg"
        file_path = os.path.join(config['output_dir'], aug_file_name)
        
        mask_amount = np.random.randint(20, 50)

        used_masks = []
        for _ in range(mask_amount):
            mask = mask_cache.fetch_mask(width=avg_contour_width, height=avg_contour_height)
            area = np.count_nonzero(mask)
            used = used_frame_area + area
            if used / total_frame_area > 0.75:
                break
            used_frame_area = used
            used_masks.append(mask)

        augmented_frame, augmented_contours = phy_cut_paste.simulate_masks(
            masks=used_masks,
            backdrop=frame.copy(),
            contour_boundaries=contours,
            strict=True,
            contour_boundary_add_delay=1000,
            timesteps=1000,
            threads=16,
            iterations=5
        )
        
        if len(augmented_contours) == 0:
            continue
        
        augmented_contours = [np.round(contour, 3) for contour in augmented_contours]
        
        image_row = (image_id, file_path, config['run_id'], json.dumps({
            'id': image_id,
            'file_name': aug_file_name,
            'height': augmented_frame.shape[0],
            'width': augmented_frame.shape[1],
            'extras': {
                'augmented': True,
            },
        }))
        
        annotation_rows = [
            (str(uuid.uuid1()), image_id, config['run_id'], json.dumps({
                'id': str(uuid.uuid1()),
                'image_id': image_id,
                'category_id': config['category_id'],
                'segmentation': [contour.flatten().tolist()],
                'area': cv2.contourArea(contour.astype(np.int32)),
                'bbox': cv2.boundingRect(contour.astype(np.int32)),
                'iscrowd': 0,
                'extras': {
                    'augmented': True,
                },
            }))
            for contour in contours
        ]
        annotation_rows.extend([
            (str(uuid.uuid1()), image_id, config['run_id'], json.dumps({
                'id': str(uuid.uuid1()),
                'image_id': image_id,
                'category_id': config['category_id'],
                'segmentation': [contour.flatten().tolist()],
                'area': cv2.contourArea(contour.astype(np.int32)),
                'bbox': cv2.boundingRect(contour.astype(np.int32)),
                'iscrowd': 0,
                'extras': {
                    'augmented': True,
                },
            }))
            for contour in augmented_contours
        ])
        
        yield file_path, augmented_frame, image_row, annotation_rows


def apply_regular_augmentations(frame, contours, transformer, config):
    
    polygons = [Polygon(contour) for contour in contours]
    
    for _ in range(config['num_regular_augs']):

        image_id = str(uuid.uuid1())

        aug_file_name = f"{image_id}.rgaug.jpg"

        file_path = os.path.join(config['output_dir'], aug_file_name)

        augmented_frame, augmented_contours = apply_augmentation(frame, polygons, transformer)
        
        image_row = (image_id, file_path, config['run_id'], json.dumps({
            'id': image_id,
            'file_name': aug_file_name,
            'height': augmented_frame.shape[0],
            'width': augmented_frame.shape[1],
            'extras': {
                'augmented': True,
            },
        }))
        
        annotation_rows = [
            (str(uuid.uuid1()), image_id, config['run_id'], json.dumps({
                'id': str(uuid.uuid1()),
                'image_id': image_id,
                'category_id': config['category_id'],
                'segmentation': [contour.flatten().tolist()],
                'area': cv2.contourArea(contour.astype(np.int32)),
                'bbox': cv2.boundingRect(contour.astype(np.int32)),
                'iscrowd': 0,
                'extras': {
                    'augmented': True,
                },
            }))
            for contour in augmented_contours
        ]
        
        yield file_path, augmented_frame, image_row, annotation_rows
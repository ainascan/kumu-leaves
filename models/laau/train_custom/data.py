import os
import json
import numpy as np
from tqdm import tqdm
import cv2
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import Dataset, BatchSampler
from torchvision.io import read_image
from torchvision.tv_tensors import Image, BoundingBoxes, Mask


def parse_segmentation(frame_shape, annotation):
    contour = np.array(annotation['segmentation']).reshape(-1, 2)

    # we can recreate the mask by drawing the contour
    mask = np.zeros((frame_shape[1], frame_shape[2]), dtype=np.uint8)
    mask = cv2.drawContours(mask, [contour], -1, 1, -1)
    mask = Mask(mask)
    
    # bounding box is in xywh format traditionally for COCO formats.
    # torchvision typically uses xyxy format.
    bbox = annotation['bbox']
    bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
    bbox = BoundingBoxes(bbox, format='xyxy', canvas_size=(frame_shape[2], frame_shape[1]))
    
    return bbox, mask


def load_annotations(image_path, annotations, parallel=False, instances_per_image='all'):
    frame = Image(read_image(image_path))

    # shuffle annotations to get random instances
    # with the instances per image, only get
    # a subset of the annotations. So every training
    # loop will have a different set of instances
    if instances_per_image == 'all':
        instances_per_image = len(annotations)
    else:
        np.random.shuffle(annotations)

    boxes = []
    masks = []
    labels = []
    
    if parallel:
        pool = ThreadPoolExecutor(max_workers=os.cpu_count())
        futures = []
        for annotation in annotations[:instances_per_image]:
            futures.append(pool.submit(parse_segmentation, frame.shape, annotation))

        for future in as_completed(futures):
            try:
                result = future.result()
                boxes.append(result[0])
                masks.append(result[1])
                labels.append(1)
            except:
                pass

    else:
        for annotation in annotations[:instances_per_image]:
            try:
                result = parse_segmentation(frame.shape, annotation)
                boxes.append(result[0])
                masks.append(result[1])
                labels.append(1)
            except:
                pass

    return frame, boxes, masks, labels


class LaauBatchSampler(BatchSampler):
    """
    Return batches of indicies where each batch contains images of the same size.
    """
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # group indices by image size
        self.indices_map = {}
        for index in np.arange(len(dataset)):
            shape = dataset.frame_shape(index)
            if shape not in self.indices_map:
                self.indices_map[shape] = []
            self.indices_map[shape].append(index)

        # for each shape, group indices into batches of size batch_size
        self.batches = []
        for shape, indices in self.indices_map.items():
            for i in range(0, len(indices), batch_size):
                batch = indices[i:i+batch_size]
                # may remove batches with less than batch_size
                if len(batch) == batch_size:
                    self.batches.append(batch)

        np.random.shuffle(self.batches)


    def __iter__(self):
        for batch in self.batches:
            yield batch


    def __len__(self):
        return len(self.batches)



class LaauDataset(Dataset):

    def __init__(self, configuration, images, annotations, indicies):
        self.images = images
        self.annotation_map = annotations
        self.directory = configuration['coco_dir']
        self.instances_per_image = configuration['instances_per_image']
        self.indices = indicies
        self.image_cache = {}
        self.cache_size = 100

    def preload(self):
        def try_load_annotation(index, image_path, annotations):
            try:
                results = load_annotations(image_path, annotations)
                if len(results[-1]) == 0:
                    return None
                return index
            except:
                return None

        loader = tqdm(total=len(self.indices), desc='Preloading', position=0, leave=True)
        pool = ThreadPoolExecutor(max_workers=os.cpu_count())
        futures = []
        new_indices = []
        bad_annotations = 0
        
        for index in self.indices:
            image = self.images[index]
            image_path = os.path.join(self.directory, os.path.basename(image['file_name']))
            annotations = self.annotation_map[image['id']]
            futures.append(pool.submit(try_load_annotation, index, image_path, annotations))
        
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                new_indices.append(result)
            else:
                bad_annotations += 1

            loader.update(1)
            loader.set_postfix(bad_annotations=bad_annotations)
        
        loader.close()

        logging.warning(f"Skipped {bad_annotations} images with bad annotations")
        self.indices = new_indices


    def frame_shape(self, index):
        return f"{self.images[self.indices[index]]['height']},{self.images[self.indices[index]]['width']}"


    def __len__(self):
        return len(self.indices)


    def __getitem__(self, index):
        image = self.images[self.indices[index]]
        image_path = os.path.join(self.directory, os.path.basename(image['file_name']))
        annotations = self.annotation_map[image['id']]
        
        return load_annotations(
            image_path,
            annotations,
            parallel=True,
            instances_per_image=self.instances_per_image
        )

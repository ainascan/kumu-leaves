import os

import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import util

class MaskCache():
    
    def __init__(self, image_map, annotation_map, category_map):
        self.image_map = image_map
        self.annotation_map = annotation_map
        self.category_map = category_map
        
        self.mask_cache = []
        self.mask_df = []
        self.backdrop_cache = []
        
    
    def load_cluster_model(self, category, clusters=10):
        annotations = [a for a in self.annotation_map.values()]
        annotations = [item for sublist in annotations for item in sublist]
        annotations = [a for a in annotations if a['category_id'] == self.category_map[category]]
        
        df = []
        for annotation in annotations:
            bbox = annotation['bbox']
            df.append((bbox[2], bbox[3]))
            
        df = pd.DataFrame(df, columns=['width', 'height'])
        
        x = df[['width', 'height']]
        
        model = Pipeline([  
            ('scaler', StandardScaler()),
            ('kmeans', KMeans(n_clusters=clusters))
        ])
        
        model.fit(x)

        self.cluster_model = model


    def fetch_mask(self, width=None, height=None):
        total_clusters = self.cluster_model.named_steps['kmeans'].n_clusters
        
        clusters = []
        if width is not None and height is not None:
            pred = self.cluster_model.predict([[width, height]])[0]
            clusters.append(pred)
            
            if pred == 0:
                clusters.append(1)
                clusters.append(2)
            elif pred == total_clusters - 1:
                clusters.append(total_clusters - 2)
                clusters.append(total_clusters - 3)
            else:
                clusters.append(pred - 1)
                clusters.append(pred + 1)
        
        else:
            clusters = list(range(total_clusters))
            np.random.shuffle(clusters)
            clusters = clusters[:3]
        
    
        # filter to get the masks of the same clusters
        masks = self.mask_df[self.mask_df.cluster.isin(clusters)]
        
        # randomly select a row from the filtered masks
        row = masks.iloc[np.random.randint(0, len(masks))]
        
        return self.mask_cache[row['index']].copy()


    def fetch_backdrop(self):
        return self.backdrop_cache[np.random.randint(0, len(self.backdrop_cache))].copy()


    def load_backdrops(self, backdrop_dir, amount=100):
        loader = tqdm(total=amount, desc='Loading backdrops...')
        
        if amount <= 0:
            loader.close()
            return
        
        def recursive_list(dir):
            files = os.listdir(dir)
            files = [os.path.join(dir, f) for f in files]
            files = [f for f in files if os.path.isfile(f) and f.endswith('.jpg')]
            dirs = [os.path.join(dir, d) for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
            for d in dirs:
                files.extend(recursive_list(d))
            return files
        
        images = recursive_list(backdrop_dir)
        np.random.shuffle(images)
        
        for image in images[:amount]:
            image_path = os.path.join(backdrop_dir, image)
            frame = cv2.imread(image_path)
            frame = util.resize_shortest_edge(frame, min_size=1080, max_size=1920)
            self.backdrop_cache.append(frame)
            loader.update(1)
        
        loader.close()


    def load_masks(self, category, amount=2500):
        image_keys = list(self.image_map.keys())
        np.random.shuffle(image_keys)
        
        loader = tqdm(total=amount, desc='Loading masks...')
        
        for image_id in image_keys:
            image = self.image_map[image_id]
            
            if image_id not in self.annotation_map:
                continue
            
            annotations = self.annotation_map[image_id]
            annotations = [a for a in annotations if not util.is_invalid_segmentation(a)]
            annotations = [a for a in annotations if a['category_id'] == self.category_map[category]]

            if len(annotations) == 0:
                continue

            np.random.shuffle(annotations)
            
            frame = cv2.imread(image['full_path'])
            contours = [np.array(a['segmentation']).reshape(-1, 2) for a in annotations]
            frame, contours = util.resize_frame_and_contours(frame, contours, min_size=1080, max_size=1920)
            
            for contour in contours:
                mask = util.generate_cropped_mask(frame, contour)
                self.mask_cache.append(mask)
                self.mask_df.append({
                    'index': len(self.mask_cache) - 1,
                    'cluster': self.cluster_model.predict([[mask.shape[1], mask.shape[0]]])[0],
                    'width': mask.shape[1],
                    'height': mask.shape[0]
                })
                loader.update(1)
                
                if len(self.mask_cache) >= amount:
                    break
            
            if len(self.mask_cache) >= amount:
                break
        
        loader.close()
    
        self.mask_df = pd.DataFrame(self.mask_df)
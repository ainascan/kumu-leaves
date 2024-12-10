from PIL import Image, ImageOps
import numpy as np
import torch as t
from torch.utils.data import Dataset
from torchvision.transforms import v2
import s3fs
import cv2
import os

_cache_dir = '/tmp/kina'

if not os.path.exists(_cache_dir):
    os.makedirs(_cache_dir)



def get_image(s3, image_path):
    image_name = os.path.basename(image_path)
    disk_path = os.path.join(_cache_dir, image_name)
        
    if os.path.exists(disk_path):
        with open(disk_path, 'rb') as f:
            image = Image.open(f)
            image = ImageOps.exif_transpose(image)
            image = np.array(image)
            return image
    
    with s3.open(image_path, 'rb') as g:
        image = Image.open(g)
        image = ImageOps.exif_transpose(image)
        
    image.save(disk_path)

    return np.array(image)



class DefectivePatches(Dataset):
    def __init__(self, df, sett='train'):
        self.frame = df
        self.sett = sett
        
        self.augments = v2.Compose([
            #v2.ToTensor(),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomAdjustSharpness(2.0),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),
        ])
        
        # self.transform = v2.Compose([
        #     v2.Resize(64, interpolation=Image.BILINEAR),
        #     v2.CenterCrop(64),
        #     v2.ToTensor(),
        #     # v2.ToDtype(t.float32, scale=True),
        #     # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])
        
        self.frame = self.frame[self.frame['sett'] == sett]

    
    def __len__(self):
        return len(self.frame)


    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        
        image_path = row['image_path']
        label = row['defective']
        
        frame = cv2.imread(image_path)
        #frame = cv2.detailEnhance(frame)

        if self.sett == 'train':
            frame = self.augments(frame)

        
        # convert h,w,c to c,h,w. 
        frame = t.tensor(frame, dtype=t.float32).permute(2, 0, 1) / 255.0
        
        #print(frame.shape, frame.min(), frame.max())
        
        # frame = t.tensor(frame, dtype=t.float32)
        
        return frame, t.tensor(label, dtype=t.long)


class DefectiveImages(Dataset):
    def __init__(self, df, storage_options, sett='train'):
        self.frame = df
        self.sett = sett
        
        self.augments = v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(90),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        ])
        
        self.transform = v2.Compose([
            v2.Resize(320, interpolation=Image.BICUBIC),
            v2.CenterCrop(300),
            v2.ToTensor(),
            v2.ToDtype(t.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.frame = self.frame[self.frame['sett'] == sett]
        self.s3 = s3fs.S3FileSystem(
            anon=False,
            use_ssl=False,
            key=storage_options['AWS_ACCESS_KEY_ID'],
            secret=storage_options['AWS_SECRET_ACCESS_KEY'],
            client_kwargs={
                'region_name': storage_options['AWS_REGION']
            }
        )


    def __len__(self):
        return len(self.frame)


    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        
        image_path = row['image_path']
        label = row['defective']
        contour = row['contour']
        
        frame = get_image(self.s3, image_path)
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        x, y, w, h = cv2.boundingRect(contour)
        masked_frame = masked_frame[y:y+h, x:x+w]
        
        if self.sett == 'train' and np.random.rand() > 0.5:
            masked_frame = self.augments(masked_frame)
        
        masked_frame = Image.fromarray(masked_frame)
        masked_frame = self.transform(masked_frame)
        
        return masked_frame, t.tensor(label, dtype=t.long)



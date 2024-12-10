from shapely.geometry import Polygon
from PIL import Image
import numpy as np
import logging
import cv2
import time
import numba
import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from pytorch_toolbelt.inference.tiles import ImageSlicer, CudaTileMerger, TileMerger
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, to_numpy

import util

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.features[-2:].parameters():
            param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1280, 1)
        )
    
    def forward(self, x):
        return self.model(x)


def extractor(image, amount=64) -> np.ndarray:
    image = Image.fromarray(image)
    paletted = image.convert('P', palette=Image.ADAPTIVE, colors=amount)
    palette = paletted.getpalette()
    color_idxs = paletted.getcolors()
    colors = np.array([palette[idx*3:idx*3+3] for _, idx in color_idxs]) / 255
    colors = colors[np.argsort(np.linalg.norm(colors, axis=1))]
    colors = colors.flatten().tolist()
    colors += [0] * ((amount*3) - len(colors))
    return colors


@numba.jit(nopython=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# @numba.jit(nopython=True)
# def defective_percent(heatmap):
#     #heatmap = heatmap / np.max(heatmap)
#     #return np.sum(heatmap > 0.5) / np.sum(heatmap > 0)


@numba.jit(nopython=True)
def extract_patches(frame, t=64, stride=8):
    pos = []
    tiles = []
    
    for x in range(0, frame.shape[1] - t, stride):
        for y in range(0, frame.shape[0] - t, stride):
            # colors = extractor(frame[y:y+t, x:x+t])
            # if np.all(colors == 0):
            #     continue
            # features.append((x, y, colors))
            tile = frame[y:y+t, x:x+t]
            #if np.all(tile == 0):
            #    continue

            pos.append([x, y])
            tiles.append(tile)

    return tiles, pos


def resize_to(bbox, contour, mask, shape):
    hr = shape[1] / mask.shape[1]
    wr = shape[0] / mask.shape[0]
    mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    contour = contour * np.array([hr, wr]) 
    bbox = bbox * np.array([hr, wr, hr, wr])
    bbox = bbox.astype(np.int32)
    return bbox, contour, mask


def inference(kina_model, image, results, config, t=64):
    
    blur = config['kina_blur']
    thresold = config['kina_score_threshold']
    stride = config['kina_stride']

    for result in results:
        start_time = time.perf_counter()
        bbox, contour, mask = result['bbox'], result['contour'], result['mask']
        
        # Laau reduced the size of the image, we need to upscale to match the original image
        bbox, contour, mask = resize_to(bbox, contour, mask, image.shape)
        
        original_contour = contour.copy()
        
        masked_frame = cv2.bitwise_and(image, image, mask=mask)
        masked_frame = masked_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        
        # offset contour to be cropped to the masked_frame
        contour -= [bbox[0], bbox[1]]
        contour = contour.reshape(-1, 2).astype(np.int32)
        
        # mask that we can use to filter out any values outside of the contour
        contour_mask = np.zeros(masked_frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, (255), -1)
        
        logging.debug(f'Preprocessed image in {time.perf_counter() - start_time:.2f}s')
        
        start_time = time.perf_counter()
        
        ############################################
        #patches, positions = extract_patches(masked_frame, t, stride)        
        #patches = torch.stack([torch.tensor(p, dtype=torch.float32).permute(2, 0, 1) / 255.0 for p in patches])
        
        #if config['gpu']:
        #    patches = patches.cuda()
        
        tiler = ImageSlicer(masked_frame.shape, tile_size=(t, t), tile_step=(stride, stride))
        tiles = torch.stack([torch.tensor(tile, dtype=torch.float32).permute(2, 0, 1) / 255.0 for tile in tiler.split(masked_frame)])

        # move to gpu. But this may fail. Catch error and move to cpu if it fails

        if config['gpu']:
            tiles = tiles.cuda()
            merger = TileMerger(tiler.target_shape, 1, tiler.weight, device='cuda')
        else:
            merger = TileMerger(tiler.target_shape, 1, tiler.weight, device='cpu')

        logging.debug(f'Extracted {len(tiles)} patches in {time.perf_counter() - start_time:.2f}s')
        

        start_time = time.perf_counter()
        # #predictions = kina_model.predict_proba([p[2] for p in patches])[:, 1] > thresold
        # with torch.no_grad():
        #     predictions = kina_model(patches)

        # predictions = sigmoid(predictions.cpu().detach().numpy().flatten())
        # patches = patches.cpu()
        
        # all at once
        predictions = kina_model(tiles)
        predictions = torch.sigmoid(predictions)
        
        # using the tiler, we merge the predictions for each tile
        # using a weighted average. Then, the merge() normalizes it.
        merger.integrate_batch(predictions, tiler.crops)

        logging.debug(f'Predicted {len(predictions)} patches in {time.perf_counter() - start_time:.2f}s')
        
        heatmap = np.moveaxis(to_numpy(merger.merge()), 0, -1)
        heatmap = tiler.crop_to_orignal_size(heatmap)
        
        if config['gpu']:
            tiles = tiles.cpu()
            predictions = predictions.cpu()

        # heatmap = np.zeros(masked_frame.shape[:2], dtype=np.float32)
        # for i in range(len(predictions)):
        #     x, y = positions[i]
        #     heatmap[y:y+t, x:x+t] += predictions[i]

        # # normalize the heatmap in order to blend the strides together
        # heatmap = heatmap / np.max(heatmap)
        
        # set to 0 the values below the thresold
        heatmap[heatmap <= thresold] = 0
        
        # set to 0 the values outside of the contour
        heatmap = cv2.bitwise_and(heatmap, heatmap, mask=contour_mask)

        # percentage of leaf that is defective. total pixels not 0 within contour / total pixels within the contour
        defective_percentage = np.sum(heatmap > 0) / np.sum(contour_mask > 0)
        
        segmentation_mask = heatmap.copy().astype(np.uint8)
        
        # implement this for a binary segmentation mask
        # heatmap[heatmap > 0] = 255
        # heatmap = heatmap.astype(np.uint8)
        # heatmap = cv2.merge([heatmap, heatmap, heatmap])
        
        # implement this for a colored heatmap with dramatic effect
        heatmap *= 255.0
        heatmap = heatmap.astype(np.uint8)    
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.GaussianBlur(heatmap, (blur, blur), 0)
        heatmap = cv2.bitwise_and(heatmap, heatmap, mask=contour_mask)
        
        # apply heatmap to the final image. This will be the heatmap on a background
        new_mask = np.zeros_like(image)
        new_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = heatmap

        logging.debug(f'Generated heatmap in {time.perf_counter() - start_time:.2f}s')
        
        logging.debug(f'Defective percentage: {defective_percentage:.2f}')

        yield {
            **result,
            'bbox': bbox,
            'contour': original_contour,
            'mask': mask,
            'heatmap': new_mask,
            'segmentation_mask': segmentation_mask,
            'defective_percentage': defective_percentage
        }
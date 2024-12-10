import cv2
import numpy as np

def inference(laau_model, image):

    masks = []
    results = laau_model(image)
    results = results['instances'].to('cpu')

    for i, pred_mask in enumerate(results.pred_masks):
        mask = pred_mask.numpy().astype(np.uint8)
        bbox = results.pred_boxes[i].tensor.numpy().astype(np.int32)[0]
        score = results.scores[i].item()
        
        mask[:bbox[1], :] = 0
        mask[bbox[3]:, :] = 0
        mask[:, :bbox[0]] = 0
        mask[:, bbox[2]:] = 0
        
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours[0], key=cv2.contourArea)
        
        submask = np.zeros_like(mask)
        cv2.drawContours(submask, [contour], -1, 1, -1)

        masks.append({
            'contour': contour,
            'bbox': bbox,
            'mask': submask,
            'confidence': score,
        })
        
    return masks
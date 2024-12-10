import cv2
import numpy as np
from imgaug.augmentables.polys import Polygon

def scale_to_fit_in(image, max_size, interpolation=cv2.INTER_LINEAR):
    h, w = image.shape[:2]
    largest_edge = max(h, w)
    if largest_edge <= max_size:
        return image
    scale = max_size / largest_edge
    new_w, new_h = int(w * scale), int(h * scale)
    image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    return image


def resize_frame_and_contours(frame, contours, min_size, max_size, interpolation=cv2.INTER_LINEAR):
    h, w = frame.shape[:2]
    short_edge = min(h, w)
    if short_edge < min_size:
        scale = min_size / short_edge
    elif short_edge > max_size:
        scale = max_size / short_edge
    else:
        return frame, contours
    new_w, new_h = int(w * scale), int(h * scale)
    frame = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)
    for contour in contours:
        contour[:, 0] = contour[:, 0] * scale
        contour[:, 1] = contour[:, 1] * scale
    return frame, contours


def resize_shortest_edge(frame, min_size=1080, max_size=1920, interpolation=cv2.INTER_LINEAR):
    h, w = frame.shape[:2]
    short_edge = min(h, w)
    if short_edge < min_size:
        scale = min_size / short_edge
    elif short_edge > max_size:
        scale = max_size / short_edge
    else:
        return frame
    new_w, new_h = int(w * scale), int(h * scale)
    frame = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)
    return frame


def generate_cropped_mask(frame, contour):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour.astype(np.int32)], -1, (255, 255, 255), -1)
    mask = cv2.bitwise_and(frame, frame, mask=mask)
    bbox = cv2.boundingRect(contour.astype(np.int32))
    mask = mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
    return mask


def generate_mask(frame, contours):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [c.astype(np.int32) for c in contours], -1, (255, 255, 255), -1)
    mask = cv2.bitwise_and(frame, frame, mask=mask)
    return mask


def is_invalid_contour(contour):
    """Tests if the annotation has an invalid segmentation. Either a semantic segmentation or the polygon is invalid."""
    try:
        if len(contour) < 24:
            return True

        if Polygon(contour).is_valid is False:
            return True
    except:
        return True
    
    return False
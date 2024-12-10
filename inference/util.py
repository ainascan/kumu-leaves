from shapely.geometry import Polygon
import cv2

def resize_shorted_edge(image, min_size, max_size, interpolation=cv2.INTER_LINEAR):
    h, w = image.shape[:2]
    short_edge = min(h, w)
    if short_edge < min_size:
        scale = min_size / short_edge
    elif short_edge > max_size:
        scale = max_size / short_edge
    else:
        return image
    new_w, new_h = int(w * scale), int(h * scale)
    image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    return image


def is_valid_contour(contour):
    try:
        if len(contour) < 24:
            return False

        if Polygon(contour).is_valid is False:
            return False

    except Exception as e:
        return False
    
    return True

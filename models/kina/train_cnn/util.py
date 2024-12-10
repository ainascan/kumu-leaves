from shapely.geometry import Polygon
import hashlib
import cv2

def is_valid_contour(contour):
    try:
        if contour is None:
            return False
            
        if len(contour) < 24:
            return False

        if not Polygon(contour).is_valid:
            return False

    except:
        return False
    
    return True


def compute_hash(row):
    image_path = row['image_path']
    contour = row['contour'].copy()
    x, y, w, h = cv2.boundingRect(contour)
    contour[:, 0] -= x
    contour[:, 1] -= y

    contour_hash = hashlib.md5(contour.flatten().astype('uint8')).hexdigest()
    image_path_hash = hashlib.md5(image_path.encode()).hexdigest()
    mask_hash = hashlib.md5(f'{contour_hash}{image_path_hash}'.encode()).hexdigest()
    return mask_hash
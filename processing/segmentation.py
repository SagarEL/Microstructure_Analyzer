import cv2
import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage

def segment_grains(image: np.ndarray) -> np.ndarray:
    """Segment grains using threshold + watershed."""
    # Threshold
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Distance transform
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    # Local peaks
    coords = peak_local_max(dist, footprint=np.ones((3, 3)), labels=thresh)
    mask = np.zeros(dist.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers = ndimage.label(mask)[0]
    # Watershed
    labels = watershed(-dist, markers, mask=thresh)
    return labels
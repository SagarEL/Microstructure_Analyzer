import cv2
import numpy as np
from utils.config import CLAHE_CLIP, CLAHE_TILE

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Enhance microstructure image: grayscale, contrast, denoise, smooth."""
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # CLAHE contrast
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    enhanced = clahe.apply(gray)
    # Median blur
    denoised = cv2.medianBlur(enhanced, 5)
    # Bilateral filter
    filtered = cv2.bilateralFilter(denoised, 9, 75, 75)
    return filtered
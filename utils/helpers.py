import os

def list_sample_images(path='samples'):
    """Return list of image file paths in samples/ folder."""
    return [os.path.join(path, f) for f in os.listdir(path)
            if f.lower().endswith(('.png', '.jpg', '.tif', '.tiff'))]
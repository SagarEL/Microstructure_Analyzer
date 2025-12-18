import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from typing import Optional, Tuple


def analyze_microstructure(labels: np.ndarray, um_per_pixel: Optional[float] = None) -> Tuple[pd.DataFrame, dict]:
    """Extract region properties and optionally convert pixel measures to micrometers.

    Args:
        labels: Labeled segmentation image (integer labels per region).
        um_per_pixel: Optional scale to convert pixels to micrometers (µm/px).

    Returns:
        df: DataFrame with measured properties (px), and extra columns in µm if scale provided.
        stats: dict with summary statistics, includes 'um_per_pixel' when provided.
    """
    props = regionprops_table(
        labels,
        properties=['area', 'equivalent_diameter',
                    'major_axis_length', 'minor_axis_length', 'orientation']
    )
    df = pd.DataFrame(props)

    # Safe aspect ratio (avoid division by zero)
    df['aspect_ratio'] = np.where(df['minor_axis_length'] > 0,
                                  df['major_axis_length'] / df['minor_axis_length'],
                                  np.nan)

    # Cell size classification (uses area in pixels)
    bins = [0, 200, 500, np.inf]
    labels_size = ['Small', 'Medium', 'Large']
    df['size_class'] = pd.cut(df['area'], bins=bins, labels=labels_size)

    # Cell shape classification
    df['shape_class'] = np.where(df['aspect_ratio'] < 1.3, 'Round', 'Elongated')

    stats = {
        'cell_count': int(len(df)),
        'avg_size': float(np.mean(df['equivalent_diameter'])) if not df.empty else 0,
        'size_std': float(np.std(df['equivalent_diameter'])) if not df.empty else 0,
        'shape_counts': df['shape_class'].value_counts().to_dict() if not df.empty else {},
        'size_counts': df['size_class'].value_counts().to_dict() if not df.empty else {}
    }

    # Physical conversion if scale provided
    if um_per_pixel is not None and um_per_pixel > 0 and not df.empty:
        df['equivalent_diameter_um'] = df['equivalent_diameter'] * um_per_pixel
        df['area_um2'] = df['area'] * (um_per_pixel ** 2)
        stats['um_per_pixel'] = float(um_per_pixel)
        stats['avg_size_um'] = float(np.mean(df['equivalent_diameter_um']))
        stats['size_std_um'] = float(np.std(df['equivalent_diameter_um']))

    return df, stats
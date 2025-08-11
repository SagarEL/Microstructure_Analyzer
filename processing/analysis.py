import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

def analyze_microstructure(labels: np.ndarray) -> (pd.DataFrame, dict):
    props = regionprops_table(
        labels,
        properties=['area', 'equivalent_diameter',
                    'major_axis_length', 'minor_axis_length', 'orientation']
    )
    df = pd.DataFrame(props)
    df['aspect_ratio'] = df['major_axis_length'] / df['minor_axis_length']

    # Cell size classification
    bins = [0, 200, 500, np.inf]
    labels_size = ['Small', 'Medium', 'Large']
    df['size_class'] = pd.cut(df['area'], bins=bins, labels=labels_size)

    # Cell shape classification
    df['shape_class'] = np.where(df['aspect_ratio'] < 1.3, 'Round', 'Elongated')

    stats = {
        'cell_count': int(len(df)),  # <-- Add this line
        'avg_size': float(np.mean(df['equivalent_diameter'])) if not df.empty else 0,
        'size_std': float(np.std(df['equivalent_diameter'])) if not df.empty else 0,
        'shape_counts': df['shape_class'].value_counts().to_dict() if not df.empty else {},
        'size_counts': df['size_class'].value_counts().to_dict() if not df.empty else {}
    }
    return df, stats
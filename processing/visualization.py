import plotly.express as px


def plot_results(df):
    """Return a Plotly Figure with grain size histogram.

    If `equivalent_diameter_um` exists in `df`, plot that with µm labels; otherwise use pixels.
    """
    if 'equivalent_diameter_um' in df.columns:
        fig = px.histogram(
            df, x='equivalent_diameter_um', nbins=30,
            title='Grain Size Distribution',
            labels={'equivalent_diameter_um': 'Equivalent Diameter (µm)'}
        )
    else:
        fig = px.histogram(
            df, x='equivalent_diameter', nbins=30,
            title='Grain Size Distribution',
            labels={'equivalent_diameter': 'Equivalent Diameter (px)'}
        )
    return fig
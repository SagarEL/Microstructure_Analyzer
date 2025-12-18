import plotly.express as px


def plot_results(df):
    """Return a Plotly Figure with histogram and scatter."""
    fig = px.histogram(
        df, x='equivalent_diameter', nbins=30,
        title='Grain Size Distribution',
        labels={'equivalent_diameter': 'Equivalent Diameter (px)'}
    )
    return fig
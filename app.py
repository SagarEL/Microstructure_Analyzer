import streamlit as st
import cv2
import numpy as np
import concurrent.futures
from utils.helpers import list_sample_images
from processing.preprocessing import preprocess_image
from processing.segmentation import segment_grains
from processing.analysis import analyze_microstructure
from processing.visualization import plot_results
from skimage.color import label2rgb
import matplotlib.pyplot as plt
from skimage import measure

st.title("🧊 Microstructure Image Analyzer")
st.markdown("Upload an SEM/TEM image or use sample images below.")


# Sidebar
st.sidebar.markdown("### Comparison Mode")
compare_mode = st.sidebar.checkbox("Enable Comparison Mode")

uploaded = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "tif"])
samples = list_sample_images()
sample_choice = st.sidebar.selectbox("Or choose sample image", [None] + samples)

@st.cache_data
def cached_preprocess_image(img):
    return preprocess_image(img)

@st.cache_data
def cached_segment_grains(proc):
    return segment_grains(proc)

@st.cache_data
def cached_analyze_microstructure(labels):
    return analyze_microstructure(labels)

def process_all(img):
    proc = cached_preprocess_image(img)
    labels = cached_segment_grains(proc)
    df, stats = cached_analyze_microstructure(labels)
    return proc, labels, df, stats

def overlay_labels(image, labels):
    # Overlay segmentation labels on the grayscale image
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    overlay = label2rgb(labels, image=gray, bg_label=0, alpha=0.4)
    return (overlay * 255).astype(np.uint8)

def _to_rgb(image):
    """Convert an OpenCV BGR image to RGB for display in Streamlit.
    If image is None or not a 3-channel image, return it unchanged.
    """
    if image is None:
        return None
    try:
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception:
        # In case image has unexpected shape/type, return as-is and let
        # later checks surface the problem to the user.
        return image
    return image

def st_image_safe(*args, **kwargs):
    """Display an image with Streamlit while supporting older/newer kw names.

    Tries `use_container_width` (newer Streamlit). If that raises a TypeError
    (unexpected keyword), falls back to `use_column_width` (older Streamlit),
    and finally calls `st.image` without width kwargs.
    """
    try:
        return st.image(*args, **{**kwargs, **{"use_container_width": True}})
    except TypeError as e:
        # Some Streamlit versions use `use_column_width` instead
        try:
            return st.image(*args, **{**kwargs, **{"use_column_width": True}})
        except TypeError:
            # Last resort: call without width argument
            return st.image(*args, **kwargs)

def plot_contours_overlay(image, labels):
    # Convert to grayscale if needed
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    contours = measure.find_contours(labels, 0.5)
    fig, ax = plt.subplots()
    ax.imshow(gray, cmap='gray')
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='red')
    ax.axis('off')
    st.pyplot(fig)

if compare_mode:
    uploaded2 = st.sidebar.file_uploader("Upload Second Image", type=["png", "jpg", "tif"], key="second")
    if (uploaded or sample_choice) and uploaded2:
        # Read first image
        if uploaded:
            data1 = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img1 = cv2.imdecode(data1, cv2.IMREAD_COLOR)
        else:
            img1 = cv2.imread(sample_choice)
        # Validate reads
        if img1 is None:
            st.error("Could not read the first image. It may be corrupted or an unsupported format.")
            st.stop()
        # Read second image
        data2 = np.asarray(bytearray(uploaded2.read()), dtype=np.uint8)
        img2 = cv2.imdecode(data2, cv2.IMREAD_COLOR)
        if img2 is None:
            st.error("Could not read the second image. It may be corrupted or an unsupported format.")
            st.stop()
        # Async processing
        with st.spinner("Processing both images..."):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future1 = executor.submit(process_all, img1)
                future2 = executor.submit(process_all, img2)
                proc1, labels1, df1, stats1 = future1.result()
                proc2, labels2, df2, stats2 = future2.result()
        # Display side-by-side
        st.markdown("## Side-by-Side Comparison")
        colA, colB = st.columns(2)
        with colA:
            st_image_safe(_to_rgb(img1), caption="Image 1: Original")
            st_image_safe(overlay_labels(img1, labels1), caption="Image 1: Segmentation Overlay")
            st.dataframe(df1.describe())
            st.metric("Cell Count", f"{stats1['cell_count']:,}")
        with colB:
            st_image_safe(_to_rgb(img2), caption="Image 2: Original")
            st_image_safe(overlay_labels(img2, labels2), caption="Image 2: Segmentation Overlay")
            st.dataframe(df2.describe())
            st.metric("Cell Count", f"{stats2['cell_count']:,}")
        # Show difference
        st.markdown(f"### Difference in Cell Count: {stats2['cell_count'] - stats1['cell_count']}")
        st.stop()

if uploaded is not None or sample_choice:
    try:
        # Read image
        if uploaded:
            data = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(sample_choice)
        if img is None:
            st.error("Could not read the selected image. It may be corrupted or an unsupported format.")
            st.stop()
        # Async processing
        with st.spinner("Processing image..."):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(process_all, img)
                proc, labels, df, stats = future.result()
        if df.empty or labels is None or np.max(labels) == 0:
            st.warning("No cells or grains detected in this image.")
            st.stop()
        # Display
        tab1, tab2, tab3 = st.tabs(["Images", "Plots", "Statistics"])

        with tab1:
            st.markdown("### Input & Segmentation")
            col_img1, col_img2, col_img3 = st.columns(3)
            col_img1.image(_to_rgb(img), caption="Original")
            col_img2.image(_to_rgb(proc), caption="Processed")
            col_img3.image(overlay_labels(img, labels), caption="Segmentation Overlay")
            st.markdown("### Contour Overlay")
            plot_contours_overlay(img, labels)

        with tab2:
            st.markdown("### Grain Size Distribution")
            st.plotly_chart(plot_results(df), use_container_width=True)
            st.markdown("### Aspect Ratio vs. Size")
            import plotly.express as px
            scatter = px.scatter(
                df, x="equivalent_diameter", y="aspect_ratio",
                title="Aspect Ratio vs. Equivalent Diameter",
                labels={"equivalent_diameter": "Equivalent Diameter (px)", "aspect_ratio": "Aspect Ratio"},
                hover_data=["area", "major_axis_length", "minor_axis_length"]
            )
            st.plotly_chart(scatter, use_container_width=True)

        with tab3:
            st.markdown("### Descriptive Statistics")
            if df.empty:
                st.warning("No cells or grains detected in this image.")
            else:
                st.dataframe(df.describe())
                st.subheader("Key Metrics")
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                cell_count = stats.get('cell_count', 0)
                avg_size = stats.get('avg_size', 0)
                size_std = stats.get('size_std', 0)
                shape_counts = stats.get('shape_counts', {})
                size_counts = stats.get('size_counts', {})

                col_stats1.metric("Cell Count", f"{cell_count:,}", help="Total number of cells detected")
                col_stats2.metric("Avg. Size", f"{avg_size:.2f}", help="Average equivalent diameter")
                col_stats3.metric("Size Std. Dev.", f"{size_std:.2f}", help="Standard deviation of cell size")

                import plotly.express as px
                if shape_counts:
                    fig_shape = px.pie(names=list(shape_counts.keys()), values=list(shape_counts.values()), title="Shape Distribution")
                    st.plotly_chart(fig_shape, use_container_width=True)
                if size_counts:
                    fig_size = px.pie(names=list(size_counts.keys()), values=list(size_counts.values()), title="Size Distribution")
                    st.plotly_chart(fig_size, use_container_width=True)

                with st.expander("Show raw statistics JSON"):
                    st.json(stats)
                st.download_button("Download CSV", df.to_csv().encode(), file_name="analysis.csv")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.stop()
else:
    st.info("Upload or select an image to begin analysis.")
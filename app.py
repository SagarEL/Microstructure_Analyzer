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

# Scale / Calibration controls
with st.sidebar.expander("Scale / Calibration", expanded=False):
    mag = st.number_input("Magnification (x)", min_value=1, value=1000, step=1)
    camera_px = st.number_input("Camera pixel size (µm)", min_value=0.0, value=0.0, format="%.4f")
    umpp_input = st.number_input("µm per pixel (µm/px) (optional)", min_value=0.0, value=0.0, format="%.6f")
    auto_apply = st.checkbox("Auto-apply scale when changed", value=False, help="Automatically apply the scale when you change the inputs (no need to press Apply)")

    # Manual apply button
    if st.button("Apply scale"):
        candidate = None
        if umpp_input and umpp_input > 0:
            candidate = float(umpp_input)
        elif camera_px and camera_px > 0 and mag and mag > 0:
            candidate = float(camera_px)/float(mag)
        if candidate:
            st.session_state['umpp'] = candidate
            st.success(f"Scale applied: {st.session_state['umpp']:.6f} µm/px")
            # If we have cached labels from a prior run, re-run analysis with new scale and update cached stats
            if 'last_labels' in st.session_state and st.session_state['last_labels'] is not None:
                # Call analyze_microstructure directly (avoid passing numpy arrays through @st.cache_data which can raise hashing/serialization issues)
                up = float(st.session_state.get('umpp')) if st.session_state.get('umpp', 0.0) and st.session_state.get('umpp', 0.0) > 0 else None
                df_new, stats_new = analyze_microstructure(st.session_state['last_labels'], um_per_pixel=up)
                st.session_state['last_df'] = df_new
                st.session_state['last_stats'] = stats_new
                st.experimental_rerun()
        else:
            st.warning("Provide either µm/px or camera pixel size and magnification to apply scale.")

    # Auto-apply behavior
    if auto_apply:
        candidate = None
        if umpp_input and umpp_input > 0:
            candidate = float(umpp_input)
        elif camera_px and camera_px > 0 and mag and mag > 0:
            candidate = float(camera_px)/float(mag)
        if candidate:
            prev = st.session_state.get('umpp', 0.0)
            if abs(prev - candidate) > 1e-12:
                st.session_state['umpp'] = candidate
                st.success(f"Scale auto-applied: {st.session_state['umpp']:.6f} µm/px")
                # Recompute analysis using existing labels if available
                if 'last_labels' in st.session_state and st.session_state['last_labels'] is not None:
                    up = float(st.session_state.get('umpp')) if st.session_state.get('umpp', 0.0) and st.session_state.get('umpp', 0.0) > 0 else None
                    df_new, stats_new = analyze_microstructure(st.session_state['last_labels'], um_per_pixel=up)
                    st.session_state['last_df'] = df_new
                    st.session_state['last_stats'] = stats_new
                    st.experimental_rerun()

    if 'umpp' in st.session_state and st.session_state['umpp'] and st.session_state['umpp'] > 0:
        st.caption(f"Applied scale: {st.session_state['umpp']:.6f} µm/px")
        if st.button('Clear scale'):
            st.session_state['umpp'] = 0.0

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
def cached_analyze_microstructure(labels, um_per_pixel: float = 0.0):
    up = float(um_per_pixel) if um_per_pixel and um_per_pixel > 0 else None
    return analyze_microstructure(labels, um_per_pixel=up)

def process_all(img, um_per_pixel: float = 0.0):
    proc = cached_preprocess_image(img)
    labels = cached_segment_grains(proc)
    df, stats = cached_analyze_microstructure(labels, um_per_pixel)
    return proc, labels, df, stats

# overlay and contour plotting functions are defined later to avoid duplication

if compare_mode:
    uploaded2 = st.sidebar.file_uploader("Upload Second Image", type=["png", "jpg", "tif"], key="second")
    if (uploaded or sample_choice) and uploaded2:
        # Read first image
        if uploaded:
            data1 = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img1 = cv2.imdecode(data1, cv2.IMREAD_COLOR)
        else:
            img1 = cv2.imread(sample_choice)
        # Read second image
        data2 = np.asarray(bytearray(uploaded2.read()), dtype=np.uint8)
        img2 = cv2.imdecode(data2, cv2.IMREAD_COLOR)
        # Process both images sequentially and pass applied per-image scales if present
        umpp1_to_use = st.session_state.get('umpp', 0.0)
        umpp2_to_use = st.session_state.get('umpp', 0.0)
        with st.spinner("Processing both images..."):
            proc1, labels1, df1, stats1 = process_all(img1, umpp1_to_use)
            proc2, labels2, df2, stats2 = process_all(img2, umpp2_to_use)
        # Display side-by-side
        st.markdown("## Side-by-Side Comparison")
        colA, colB = st.columns(2)
        with colA:
            st.image(img1, caption="Image 1: Original")
            st.image(overlay_labels(img1, labels1), caption="Image 1: Segmentation Overlay")
            st.dataframe(df1.describe())
            st.metric("Cell Count", f"{stats1['cell_count']:,}")
        with colB:
            st.image(img2, caption="Image 2: Original")
            st.image(overlay_labels(img2, labels2), caption="Image 2: Segmentation Overlay")
            st.dataframe(df2.describe())
            st.metric("Cell Count", f"{stats2['cell_count']:,}")
        # Show difference
        st.markdown(f"### Difference in Cell Count: {stats2['cell_count'] - stats1['cell_count']}")
        st.stop()

# Main image processing and display occurs later in the file (single consolidated block).

def overlay_labels(image, labels):
    # Overlay segmentation labels on the grayscale image
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    overlay = label2rgb(labels, image=gray, bg_label=0, alpha=0.4)
    return (overlay * 255).astype(np.uint8)

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
        # Read second image
        data2 = np.asarray(bytearray(uploaded2.read()), dtype=np.uint8)
        img2 = cv2.imdecode(data2, cv2.IMREAD_COLOR)
        # Determine µm/px per image (manual overrides computed)
        umpp1_to_use = 0.0
        if 'umpp1' in locals() and umpp1 > 0:
            umpp1_to_use = float(umpp1)
        elif 'camera_px1' in locals() and camera_px1 > 0:
            umpp1_to_use = float(camera_px1) / float(mag1)

        umpp2_to_use = 0.0
        if 'umpp2' in locals() and umpp2 > 0:
            umpp2_to_use = float(umpp2)
        elif 'camera_px2' in locals() and camera_px2 > 0:
            umpp2_to_use = float(camera_px2) / float(mag2)

        # Process both images sequentially (avoid thread-related Streamlit context errors)
        with st.spinner("Processing both images..."):
            proc1, labels1, df1, stats1 = process_all(img1, umpp1_to_use)
            proc2, labels2, df2, stats2 = process_all(img2, umpp2_to_use)
        # Display side-by-side
        st.markdown("## Side-by-Side Comparison")
        colA, colB = st.columns(2)
        with colA:
            st.image(img1, caption="Image 1: Original")
            st.image(overlay_labels(img1, labels1), caption="Image 1: Segmentation Overlay")
            st.dataframe(df1.describe())
            st.metric("Cell Count", f"{stats1['cell_count']:,}")
        with colB:
            st.image(img2, caption="Image 2: Original")
            st.image(overlay_labels(img2, labels2), caption="Image 2: Segmentation Overlay")
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
        # Determine µm/px to use (prefer applied session state, then sidebar inputs)
        umpp_to_use = st.session_state.get('umpp', 0.0)
        if not umpp_to_use or umpp_to_use <= 0:
            if umpp_input and umpp_input > 0:
                umpp_to_use = float(umpp_input)
            elif camera_px and camera_px > 0 and mag and mag > 0:
                umpp_to_use = float(camera_px) / float(mag)

        # Process image (synchronously to avoid thread context issues)
        with st.spinner("Processing image..."):
            proc, labels, df, stats = process_all(img, umpp_to_use)
            st.session_state['last_proc'] = proc
            st.session_state['last_labels'] = labels
            st.session_state['last_df'] = df
            st.session_state['last_stats'] = stats
            # Prefer cached last_df/last_stats if they were set (e.g., via Apply scale re-analysis)
            df = st.session_state.get('last_df', df)
            stats = st.session_state.get('last_stats', stats)
        if df.empty or labels is None or np.max(labels) == 0:
            st.warning("No cells or grains detected in this image.")
            st.stop()
        # Display
        tab1, tab2, tab3 = st.tabs(["Images", "Plots", "Statistics"])

        with tab1:
            st.markdown("### Input & Segmentation")
            col_img1, col_img2, col_img3 = st.columns(3)
            col_img1.image(img, caption="Original")
            col_img2.image(proc, caption="Processed")
            col_img3.image(overlay_labels(img, labels), caption="Segmentation Overlay")
            st.markdown("### Contour Overlay")
            plot_contours_overlay(img, labels)

        with tab2:
            st.markdown("### Grain Size Distribution")
            st.plotly_chart(plot_results(df))
            st.markdown("### Aspect Ratio vs. Size")
            import plotly.express as px
            # Choose physical units if available
            if 'equivalent_diameter_um' in df.columns:
                x_col = 'equivalent_diameter_um'
                x_label = 'Equivalent Diameter (µm)'
                hover = ['area_um2', 'major_axis_length', 'minor_axis_length']
            else:
                x_col = 'equivalent_diameter'
                x_label = 'Equivalent Diameter (px)'
                hover = ['area', 'major_axis_length', 'minor_axis_length']

            scatter = px.scatter(
                df, x=x_col, y="aspect_ratio",
                title="Aspect Ratio vs. Equivalent Diameter",
                labels={x_col: x_label, "aspect_ratio": "Aspect Ratio"},
                hover_data=hover
            )
            st.plotly_chart(scatter)

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
                if 'avg_size_um' in stats:
                    # Show µm-based average when available
                    col_stats2.metric("Avg. Size (µm)", f"{stats['avg_size_um']:.2f}", help="Average equivalent diameter in µm")
                    col_stats3.metric("Size Std. Dev. (µm)", f"{stats['size_std_um']:.2f}", help="Standard deviation of cell size in µm")
                else:
                    col_stats2.metric("Avg. Size (px)", f"{avg_size:.2f}", help="Average equivalent diameter in pixels")
                    col_stats3.metric("Size Std. Dev.", f"{size_std:.2f}", help="Standard deviation of cell size")

                import plotly.express as px
                if shape_counts:
                    fig_shape = px.pie(names=list(shape_counts.keys()), values=list(shape_counts.values()), title="Shape Distribution")
                    st.plotly_chart(fig_shape)
                if size_counts:
                    fig_size = px.pie(names=list(size_counts.keys()), values=list(size_counts.values()), title="Size Distribution")
                    st.plotly_chart(fig_size)

                with st.expander("Show raw statistics JSON"):
                    st.json(stats)
                st.download_button("Download CSV", df.to_csv().encode(), file_name="analysis.csv")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.stop()
else:
    st.info("Upload or select an image to begin analysis.")
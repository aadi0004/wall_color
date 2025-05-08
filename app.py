
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from datetime import datetime
from wall_utils import load_image, apply_color, recommend_colors, rgb_to_hex, detect_wall_area, detect_windows
import logging
import os
import shutil

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Wall Color Recommender", layout="wide")
st.title("🎨 Wall Color Visualizer")

# --- Sidebar Instructions ---
with st.sidebar:
    st.markdown("## 📌 How to Use")
    st.markdown("""
    1. Upload a clear, well-lit room image with a visible wall.
    2. The wall will be detected using edge and contour analysis.
    3. View 2 color suggestions with previews.
    **Tips**:
    - Ensure the image is clear and well-lit for accurate wall detection.
    - Clear cache if previews fail.
    - Check console logs for errors (run with --logger.level=DEBUG).
    """)
    st.markdown("---")
    st.markdown("## 🛠️ Advanced Options")
    show_wall_mask = st.checkbox("Show detected wall mask", value=True)
    if st.button("Clear Cache"):
        try:
            cache_path = os.path.join(".streamlit", "cache")
            if os.path.exists(cache_path):
                shutil.rmtree(cache_path)
                st.success("Cache cleared successfully!")
            else:
                st.info("No cache found.")
        except Exception as e:
            st.error(f"Error clearing cache: {str(e)}")
            logger.error(f"Cache clear error: {str(e)}")

# --- Session State Initialization ---
if 'app_image_bgr' not in st.session_state:
    st.session_state.app_image_bgr = None
if 'app_wall_mask' not in st.session_state:
    st.session_state.app_wall_mask = None
if 'app_uploaded_file_bytes' not in st.session_state:
    st.session_state.app_uploaded_file_bytes = None
if 'selected_color' not in st.session_state:
    st.session_state.selected_color = None
if 'windows' not in st.session_state:
    st.session_state.windows = []

# --- Image Upload ---
uploaded_file = st.file_uploader("📷 Upload a room image", type=["jpg", "jpeg", "png"])

# --- Process Image Upload ---
if uploaded_file:
    file_bytes = uploaded_file.read()
    current_file_id = getattr(uploaded_file, 'id', None) or uploaded_file.name
    if (st.session_state.app_image_bgr is None or 
        current_file_id != st.session_state.app_uploaded_file_bytes):
        # Reset session state
        st.session_state.app_image_bgr = load_image(file_bytes, uploaded_file.name)
        st.session_state.app_wall_mask = None
        st.session_state.app_uploaded_file_bytes = current_file_id
        st.session_state.selected_color = None
        st.session_state.windows = []

    if st.session_state.app_image_bgr is None:
        st.error("Could not load the image. Please try a different file or format.")
        st.stop()

    # Display original image
    h, w = st.session_state.app_image_bgr.shape[:2]
    st.image(cv2.cvtColor(st.session_state.app_image_bgr, cv2.COLOR_BGR2RGB), 
             caption=f"Original Image ({w}x{h} pixels)", width=None)

    # Wall Detection and Color Suggestions
    st.subheader("🎨 Wall Detection and Color Suggestions")
    with st.spinner("Detecting wall area and generating color suggestions..."):
        try:
            progress_bar = st.progress(0)
            wall_mask, error, debug_edges, debug_contours_green_blue, debug_contours_green_red = detect_wall_area(st.session_state.app_image_bgr)
            progress_bar.progress(30)
            
            if error:
                st.error(error)
                if debug_edges is not None:
                    st.image(debug_edges, caption="Debug: Edge Detection", width=None)
                if debug_contours_green_blue is not None:
                    st.image(debug_contours_green_blue, caption="Debug: Contours (Green = Windows, Blue = Near-Rectangular)", width=None)
                if debug_contours_green_red is not None:
                    st.image(debug_contours_green_red, caption="Debug: Contours (Green = Windows, Red = Aspect Outliers)", width=None)
                st.markdown("""
                **Suggestions**:
                - Upload a clear, well-lit image with a distinct wall.
                """)
                st.stop()
            
            st.session_state.app_wall_mask = wall_mask
            st.success("Wall area detected!")
            
            # Display debug images
            if debug_edges is not None:
                st.image(debug_edges, caption="Debug: Edge Detection", width=None)
            if debug_contours_green_blue is not None:
                st.image(debug_contours_green_blue, caption="Debug: Contours (Green = Windows, Blue = Near-Rectangular)", width=None)
            if debug_contours_green_red is not None:
                st.image(debug_contours_green_red, caption="Debug: Contours (Green = Windows, Red = Aspect Outliers)", width=None)
            
            # Display wall mask if selected
            if show_wall_mask:
                wall_mask_display = cv2.cvtColor(wall_mask, cv2.COLOR_GRAY2RGB)
                st.image(wall_mask_display, caption="Detected Wall Mask (White = Wall)", width=None)
            
            progress_bar.progress(50)
            
            # Color Suggestions and Previews
            color_recommendations, color_error = recommend_colors(st.session_state.app_image_bgr, wall_mask, num_colors=2)
            progress_bar.progress(75)
            if color_recommendations is None:
                st.warning(f"Could not generate color recommendations: {color_error}")
            else:
                st.write("Here are 2 suggested colors for your wall with previews:")
                cols = st.columns(2)
                for i, color_bgr in enumerate(color_recommendations):
                    hex_code = rgb_to_hex(color_bgr)
                    with cols[i]:
                        # Display color swatch
                        st.markdown(f"<div style='background-color:{hex_code}; width:100%; height:60px; border-radius:8px;'></div>", 
                                    unsafe_allow_html=True)
                        st.write(f"{hex_code}")
                        
                        # Generate preview
                        preview_image = apply_color(st.session_state.app_image_bgr, wall_mask, color_bgr)
                        st.image(cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB), 
                                 caption=f"Preview: Wall in {hex_code}", width=None)
                        
                        # Download preview
                        download_filename = f"wall_color_{hex_code[1:]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        img_pil = Image.fromarray(cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB))
                        buffer = io.BytesIO()
                        img_pil.save(buffer, format="PNG")
                        buffer.seek(0)
                        st.download_button(
                            label=f"📥 Download {hex_code} Preview",
                            data=buffer,
                            file_name=download_filename,
                            mime="image/png",
                            key=f"download_color_{i}"
                        )
            progress_bar.progress(100)
        except Exception as e:
            st.error(f"Unexpected error during wall detection or color recommendation: {str(e)}")
            logger.error(f"Wall detection/color recommendation error: {str(e)}")
            st.stop()

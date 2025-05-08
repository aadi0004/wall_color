import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from datetime import datetime
from wall_utils import load_image, apply_color, recommend_colors, rgb_to_hex, suggest_accessories, apply_accessory, suggest_accessory_placement, detect_wall_area, detect_windows, suggest_curtains, create_test_png
from streamlit_drawable_canvas import st_canvas
import logging
import os
import shutil

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Wall Color & Decor Recommender", layout="wide")
st.title("🎨 Wall Color, Accessory & Curtains Visualizer")

# --- Sidebar Instructions ---
with st.sidebar:
    st.markdown("## 📌 How to Use")
    st.markdown("""
    1. Upload a clear, well-lit room image with a visible wall and windows.
    2. The wall will be detected using edge and contour analysis, followed by 2 color suggestions with previews.
    3. View automatic curtain suggestions based on detected windows.
    4. Click 'Suggest Accessories' for sofa/chair recommendations.
    5. Adjust accessory/curtain placement, size, or rotation using the canvas.
    **Tips**:
    - Ensure assets/ contains sofa.png, chair.png, curtain.png (1024x1024, gray #808080, transparent).
    - Clear cache if previews fail.
    - Use 'Display Raw PNGs' to verify assets.
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
    if st.button("Create Test PNGs"):
        try:
            for accessory in ["sofa", "chair", "curtain"]:
                create_test_png(accessory)
            st.success("Test PNGs (sofa.png, chair.png, curtain.png) created in assets/")
        except Exception as e:
            st.error(f"Error creating test PNGs: {str(e)}")
            logger.error(f"Test PNG creation error: {str(e)}")
    if st.button("Display Raw PNGs"):
        for accessory in ["sofa", "chair", "curtain"]:
            path = f"assets/{accessory}.png"
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    if img.shape[2] == 4:
                        st.image(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA), caption=f"Raw {accessory}.png", width=200)
                    else:
                        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Raw {accessory}.png (No Alpha)", width=200)
                    logger.debug(f"Displayed raw {accessory}.png: {img.shape}")
                else:
                    st.warning(f"Failed to load {accessory}.png")
            else:
                st.warning(f"{accessory}.png not found in assets/")

# --- Session State Initialization ---
if 'app_image_bgr' not in st.session_state:
    st.session_state.app_image_bgr = None
if 'app_wall_mask' not in st.session_state:
    st.session_state.app_wall_mask = None
if 'app_uploaded_file_bytes' not in st.session_state:
    st.session_state.app_uploaded_file_bytes = None
if 'selected_color' not in st.session_state:
    st.session_state.selected_color = None
if 'accessory_positions' not in st.session_state:
    st.session_state.accessory_positions = {}
if 'accessory_suggestions' not in st.session_state:
    st.session_state.accessory_suggestions = []
if 'curtain_suggestion' not in st.session_state:
    st.session_state.curtain_suggestion = None
if 'curtain_position' not in st.session_state:
    st.session_state.curtain_position = None
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
        st.session_state.accessory_positions = {}
        st.session_state.accessory_suggestions = []
        st.session_state.curtain_suggestion = None
        st.session_state.curtain_position = None
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

    # Automatic Curtains Suggestion
    st.subheader("🪟 Curtains Suggestions")
    with st.spinner("Detecting windows and suggesting curtains..."):
        try:
            progress_bar = st.progress(0)
            windows, debug_edges, debug_contours_green_blue, debug_contours_green_red = detect_windows(st.session_state.app_image_bgr, wall_mask)
            st.session_state.windows = windows
            progress_bar.progress(30)
            if debug_edges is not None:
                st.image(debug_edges, caption="Debug: Edge Detection", width=None)
            if debug_contours_green_blue is not None:
                st.image(debug_contours_green_blue, caption="Debug: Contours (Green = Windows, Blue = Near-Rectangular)", width=None)
            if debug_contours_green_red is not None:
                st.image(debug_contours_green_red, caption="Debug: Contours (Green = Windows, Red = Aspect Outliers)", width=None)
            curtain_suggestion = suggest_curtains(st.session_state.app_image_bgr, wall_mask, st.session_state.selected_color, windows)
            st.session_state.curtain_suggestion = curtain_suggestion
            st.session_state.curtain_position = None
            progress_bar.progress(60)

            if not curtain_suggestion["success"]:
                st.warning(f"No curtains recommended: {curtain_suggestion['reason']}")
            else:
                curtain_type = curtain_suggestion["type"]
                color_bgr = curtain_suggestion["color"]
                hex_code = rgb_to_hex(color_bgr)
                st.success(f"Recommended: {curtain_type.capitalize()} curtains in {hex_code}")
                st.write(f"Reason: {curtain_suggestion['reason']}")
                if windows:
                    st.write(f"Detected {len(windows)} window(s) for curtain placement.")
                    highlight_img = st.session_state.app_image_bgr.copy()
                    for window in windows:
                        x1, y1, x2, y2 = window
                        cv2.rectangle(highlight_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    st.image(cv2.cvtColor(highlight_img, cv2.COLOR_BGR2RGB), 
                             caption="Detected Windows for Curtains", width=None)
                else:
                    st.warning("No windows detected; using default placement.")

                # Initial preview (apply to first window or default)
                logger.debug(f"Generating initial curtain preview")
                placement_rect = windows[0] if windows else None
                preview_image = apply_accessory(st.session_state.app_image_bgr, "curtain", color_bgr, placement_rect)
                if np.array_equal(preview_image, st.session_state.app_image_bgr):
                    st.warning("Failed to apply curtains. Check assets/curtain.png (1024x1024, gray #808080, transparent).")
                    logger.error("Initial curtain preview returned unchanged image.")
                    debug_img = st.session_state.app_image_bgr.copy()
                    x, y = (w // 4, h // 4) if not placement_rect else (placement_rect[0], placement_rect[1])
                    cv2.rectangle(debug_img, (x, y), (x + w // 4, y + h // 4), (0, 255, 0), 2)
                    st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), 
                             caption="Debug: Curtain Placement", width=None)
                else:
                    st.image(cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB), 
                             caption=f"Initial Preview: {curtain_type.capitalize()} Curtains ({hex_code})", width=None)
                    logger.debug("Initial curtain preview displayed.")

                # Adjust placement
                st.markdown("### Adjust Curtains Placement")
                st.write("Drag, resize, or rotate the rectangle (applies to first window).")
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",
                    stroke_width=2,
                    stroke_color="blue",
                    background_image=Image.fromarray(cv2.cvtColor(st.session_state.app_image_bgr, cv2.COLOR_BGR2RGB)),
                    height=h,
                    width=w,
                    drawing_mode="transform",
                    key="canvas_curtain",
                )
                if canvas_result.json_data and canvas_result.json_data["objects"]:
                    rect = canvas_result.json_data["objects"][0]
                    x1, y1 = int(rect["left"]), int(rect["top"])
                    new_w, new_h = int(rect["width"] * rect["scaleX"]), int(rect["height"] * rect["scaleY"])
                    rotation = rect.get("angle", 0)
                    x2, y2 = x1 + new_w, y1 + new_h
                    if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h:
                        st.session_state.curtain_position = (x1, y1, x2, y2, rotation)
                        st.success(f"New curtain position: ({x1}, {y1}) to ({x2}, {y2}), Rotation: {rotation}°")
                    else:
                        st.warning("Invalid adjustment. Using suggested placement.")
                        st.session_state.curtain_position = placement_rect
                # Reset to suggested placement
                if st.button("Reset Curtain to Suggested Placement"):
                    st.session_state.curtain_position = placement_rect
                    st.success("Reset to suggested curtain placement.")

                # Adjusted preview
                if st.session_state.curtain_position:
                    logger.debug(f"Generating adjusted curtain preview at {st.session_state.curtain_position}")
                    preview_image = apply_accessory(st.session_state.app_image_bgr, "curtain", color_bgr, st.session_state.curtain_position)
                    if np.array_equal(preview_image, st.session_state.app_image_bgr):
                        st.warning("Failed to apply adjusted curtains. Check assets/curtain.png.")
                        logger.error("Adjusted curtain preview returned unchanged image.")
                        debug_img = st.session_state.app_image_bgr.copy()
                        x, y = st.session_state.curtain_position[0], st.session_state.curtain_position[1]
                        cv2.rectangle(debug_img, (x, y), (x + w // 4, y + h // 4), (0, 255, 0), 2)
                        st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), 
                                 caption="Debug: Curtain Adjusted Placement", width=None)
                    else:
                        st.image(cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB), 
                                 caption=f"Adjusted Preview: {curtain_type.capitalize()} Curtains ({hex_code})", width=None)
                        logger.debug("Adjusted curtain preview displayed.")

                # Download preview
                download_filename = f"curtain_{hex_code[1:]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                img_pil = Image.fromarray(cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB))
                buffer = io.BytesIO()
                img_pil.save(buffer, format="PNG")
                buffer.seek(0)
                st.download_button(
                    label="📥 Download Curtains Preview",
                    data=buffer,
                    file_name=download_filename,
                    mime="image/png"
                )
            progress_bar.progress(100)
        except Exception as e:
            st.error(f"Error generating curtain suggestions: {str(e)}")
            logger.error(f"Curtain suggestion error: {str(e)}")

    # Accessory Suggestion Section
    st.subheader("🛋️ Accessory Suggestions")
    if st.button("Suggest Accessories"):
        with st.spinner("Analyzing for accessory suggestions..."):
            try:
                progress_bar = st.progress(0)
                accessory_suggestions = suggest_accessories(st.session_state.app_image_bgr, wall_mask, st.session_state.selected_color, max_suggestions=2)
                st.session_state.accessory_suggestions = accessory_suggestions
                st.session_state.accessory_positions = {}
                progress_bar.progress(50)

                if not accessory_suggestions:
                    st.warning("No accessories recommended.")
                else:
                    for idx, suggestion in enumerate(accessory_suggestions):
                        if not suggestion["success"]:
                            st.warning(f"Accessory {idx+1}: {suggestion['reason']}")
                            continue
                        accessory_type = suggestion["type"]
                        color_bgr = suggestion["color"]
                        hex_code = rgb_to_hex(color_bgr)
                        st.success(f"Accessory {idx+1}: {accessory_type.capitalize()} in {hex_code}")
                        st.write(f"Reason: {suggestion['reason']}")

                        accessory_key = accessory_type
                        placement_rect = suggest_accessory_placement(st.session_state.app_image_bgr, wall_mask, accessory_type)
                        if placement_rect:
                            x1, y1, x2, y2 = placement_rect
                            st.write(f"Suggested placement: ({x1}, {y1}) to ({x2}, {y2})")
                            highlight_img = st.session_state.app_image_bgr.copy()
                            cv2.rectangle(highlight_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            st.image(cv2.cvtColor(highlight_img, cv2.COLOR_BGR2RGB), 
                                     caption=f"Accessory {idx+1} Suggested Placement", width=None)
                        else:
                            st.warning("Using default placement.")
                            placement_rect = None

                        # Initial preview
                        logger.debug(f"Generating initial preview for {accessory_type}")
                        preview_image = apply_accessory(st.session_state.app_image_bgr, accessory_key, color_bgr, placement_rect)
                        if np.array_equal(preview_image, st.session_state.app_image_bgr):
                            st.warning(f"Failed to apply {accessory_type}. Check assets/{accessory_key}.png (1024x1024, gray #808080, transparent).")
                            logger.error(f"Initial preview for {accessory_type} returned unchanged image.")
                            debug_img = st.session_state.app_image_bgr.copy()
                            x, y = (w // 4, 3 * h // 4) if not placement_rect else (placement_rect[0], placement_rect[1])
                            cv2.rectangle(debug_img, (x, y), (x + w // 4, y + h // 4), (0, 255, 0), 2)
                            st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), 
                                     caption=f"Debug: {accessory_type} Placement", width=None)
                        else:
                            st.image(cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB), 
                                     caption=f"Initial Preview: {accessory_type.capitalize()} ({hex_code})", width=None)
                            logger.debug(f"Initial preview for {accessory_type} displayed.")

                        # Adjust placement
                        st.markdown(f"### Adjust {accessory_type.capitalize()} Placement")
                        st.write("Drag, resize, or rotate the rectangle.")
                        canvas_result = st_canvas(
                            fill_color="rgba(255, 165, 0, 0.3)",
                            stroke_width=2,
                            stroke_color="blue",
                            background_image=Image.fromarray(cv2.cvtColor(st.session_state.app_image_bgr, cv2.COLOR_BGR2RGB)),
                            height=h,
                            width=w,
                            drawing_mode="transform",
                            key=f"canvas_accessory_{idx}",
                        )
                        if canvas_result.json_data and canvas_result.json_data["objects"]:
                            rect = canvas_result.json_data["objects"][0]
                            x1, y1 = int(rect["left"]), int(rect["top"])
                            new_w, new_h = int(rect["width"] * rect["scaleX"]), int(rect["height"] * rect["scaleY"])
                            rotation = rect.get("angle", 0)
                            x2, y2 = x1 + new_w, y1 + new_h
                            if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h:
                                st.session_state.accessory_positions[idx] = (x1, y1, x2, y2, rotation)
                                st.success(f"New position: ({x1}, {y1}) to ({x2}, {y2}), Rotation: {rotation}°")
                            else:
                                st.warning("Invalid adjustment. Using suggested placement.")
                                st.session_state.accessory_positions[idx] = placement_rect
                        # Reset to suggested placement
                        if st.button(f"Reset to Suggested Placement", key=f"reset_{idx}"):
                            st.session_state.accessory_positions[idx] = placement_rect
                            st.success("Reset to suggested placement.")

                        # Adjusted preview
                        if idx in st.session_state.accessory_positions:
                            pos = st.session_state.accessory_positions[idx]
                            logger.debug(f"Generating adjusted preview for {accessory_type} at {pos}")
                            preview_image = apply_accessory(st.session_state.app_image_bgr, accessory_key, color_bgr, pos)
                            if np.array_equal(preview_image, st.session_state.app_image_bgr):
                                st.warning(f"Failed to apply adjusted {accessory_type}. Check assets/{accessory_key}.png.")
                                logger.error(f"Adjusted preview for {accessory_type} returned unchanged image.")
                                debug_img = st.session_state.app_image_bgr.copy()
                                x, y = pos[0], pos[1]
                                cv2.rectangle(debug_img, (x, y), (x + w // 4, y + h // 4), (0, 255, 0), 2)
                                st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), 
                                         caption=f"Debug: {accessory_type} Adjusted Placement", width=None)
                            else:
                                st.image(cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB), 
                                         caption=f"Adjusted Preview: {accessory_type.capitalize()} ({hex_code})", width=None)
                                logger.debug(f"Adjusted preview for {accessory_type} displayed.")

                        # Download preview
                        download_filename = f"{accessory_key}_{hex_code[1:]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        img_pil = Image.fromarray(cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB))
                        buffer = io.BytesIO()
                        img_pil.save(buffer, format="PNG")
                        buffer.seek(0)
                        st.download_button(
                            label=f"📥 Download {accessory_type.capitalize()} Preview",
                            data=buffer,
                            file_name=download_filename,
                            mime="image/png"
                        )
                progress_bar.progress(100)
            except Exception as e:
                st.error(f"Error generating accessory suggestions: {str(e)}")
                logger.error(f"Accessory suggestion error: {str(e)}")

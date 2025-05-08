
import cv2
import numpy as np
from PIL import Image
import io
from sklearn.cluster import KMeans
from scipy.ndimage import binary_dilation
import logging

# Configure logging
logger = logging.getLogger(__name__)

def load_image(file_bytes, file_name):
    """Load image from bytes and validate format."""
    try:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image")
        if len(img_bgr.shape) != 3 or img_bgr.shape[2] != 3:
            raise ValueError("Image must be a 3-channel color image")
        logger.debug(f"Loaded image: {file_name}, shape: {img_bgr.shape}")
        return img_bgr
    except Exception as e:
        logger.error(f"Error loading image {file_name}: {str(e)}")
        return None

def apply_color(image_bgr, wall_mask, color_bgr):
    """Apply the specified color to the wall area."""
    try:
        result = image_bgr.copy()
        colored_wall = np.zeros_like(result)
        for c in range(3):
            colored_wall[:, :, c] = color_bgr[c]
        result[wall_mask == 255] = colored_wall[wall_mask == 255]
        logger.debug(f"Applied color {color_bgr} to wall")
        return result
    except Exception as e:
        logger.error(f"Error applying color: {str(e)}")
        return image_bgr

def recommend_colors(image_bgr, wall_mask, num_colors=2):
    """Recommend exactly 2 colors (analogous and complementary) based on the wall's color."""
    try:
        # Extract wall pixels to determine the dominant wall color
        wall_pixels = image_bgr[wall_mask == 255]
        if len(wall_pixels) < 100:
            logger.warning("Not enough wall pixels for color recommendation")
            return None, "Insufficient wall area."
        
        # Compute the average color of the wall
        avg_color = np.mean(wall_pixels, axis=0).astype(np.uint8)
        
        # Convert to HSV for color manipulation
        avg_color_rgb = np.uint8([[avg_color]])
        avg_color_hsv = cv2.cvtColor(avg_color_rgb, cv2.COLOR_BGR2HSV)[0][0]
        
        # Suggest two colors
        colors = []
        # 1. Analogous color (hue +30)
        analogous_hue = (avg_color_hsv[0] + 30) % 180
        analogous_hsv = np.array([analogous_hue, avg_color_hsv[1], avg_color_hsv[2]], dtype=np.uint8)
        analogous_color = cv2.cvtColor(np.uint8([[analogous_hsv]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(analogous_color)
        
        # 2. Complementary color (hue +150)
        comp_hue = (avg_color_hsv[0] + 150) % 180
        comp_hsv = np.array([comp_hue, avg_color_hsv[1], avg_color_hsv[2]], dtype=np.uint8)
        comp_color = cv2.cvtColor(np.uint8([[comp_hsv]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(comp_color)
        
        logger.debug(f"Recommended colors: {colors}")
        return colors, None
    except Exception as e:
        logger.error(f"Error recommending colors: {str(e)}")
        return None, f"Color recommendation failed: {str(e)}"

def rgb_to_hex(color_bgr):
    """Convert BGR color to hex code."""
    try:
        b, g, r = color_bgr
        hex_code = f"#{r:02x}{g:02x}{b:02x}"
        return hex_code
    except Exception as e:
        logger.error(f"Error converting to hex: {str(e)}")
        return "#FFFFFF"

def detect_wall_area(image_bgr):
    """
    Detect wall area using edge detection and contour analysis, similar to detect_windows.
    Returns wall mask, debug edges, and debug contour images.
    """
    try:
        h, w = image_bgr.shape[:2]
        # Step 1: Edge detection on the entire image
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.dilate(edges, None, iterations=2)
        debug_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        logger.debug(f"Edge detection completed: {np.sum(edges > 0)} edge pixels")
        
        # Step 2: Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        windows = []
        near_rectangular = []
        
        debug_contours_green_blue = image_bgr.copy()
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < (h * w * 0.01) or area > (h * w * 0.5):
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:  # Near-rectangular shapes
                x, y, w_contour, h_contour = cv2.boundingRect(approx)
                aspect_ratio = w_contour / (h_contour + 1e-6)
                if 0.5 <= aspect_ratio <= 2.0:  # Reasonable aspect ratio for windows
                    windows.append((x, y, x + w_contour, y + h_contour))
                    cv2.rectangle(debug_contours_green_blue, (x, y), (x + w_contour, y + h_contour), (0, 255, 0), 2)
                near_rectangular.append((x, y, x + w_contour, y + h_contour))
                cv2.rectangle(debug_contours_green_blue, (x, y), (x + w_contour, y + h_contour), (255, 0, 0), 2)
        
        # Step 3: Filter windows by aspect ratio
        debug_contours_green_red = debug_contours_green_blue.copy()
        for rect in near_rectangular:
            x1, y1, x2, y2 = rect
            aspect_ratio = (x2 - x1) / (y2 - y1 + 1e-6)
            if not (0.5 <= aspect_ratio <= 2.0):
                cv2.rectangle(debug_contours_green_red, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Step 4: Create wall mask by excluding window-like areas
        wall_mask = np.ones((h, w), dtype=np.uint8) * 255
        for window in windows:
            x1, y1, x2, y2 = window
            wall_mask[y1:y2, x1:x2] = 0
        
        # Step 5: Find the largest remaining contour (the wall)
        contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.error("No contours found after excluding windows")
            return None, "No wall contours detected.", debug_edges, debug_contours_green_blue, debug_contours_green_red
        
        wall_contour = max(contours, key=cv2.contourArea)
        wall_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(wall_mask, [wall_contour], -1, 255, -1)
        
        # Step 6: Refine mask
        wall_mask = cv2.erode(wall_mask, None, iterations=2)
        wall_mask = cv2.dilate(wall_mask, None, iterations=4)
        
        # Step 7: Ensure the mask is significant
        wall_area = np.sum(wall_mask == 255)
        if wall_area < (h * w * 0.05):
            logger.error("Wall area too small after refinement")
            return None, "Detected wall area too small.", debug_edges, debug_contours_green_blue, debug_contours_green_red
        
        logger.debug(f"Wall mask generated: {wall_area} pixels")
        return wall_mask, None, debug_edges, debug_contours_green_blue, debug_contours_green_red
    except Exception as e:
        logger.error(f"Error detecting wall area: {str(e)}")
        return None, f"Wall detection failed: {str(e)}", None, None, None

def detect_windows(image_bgr, wall_mask):
    """Detect windows on the wall and return bounding boxes with debug images."""
    try:
        h, w = image_bgr.shape[:2]
        wall_mask_binary = wall_mask.copy()
        wall_mask_binary[wall_mask_binary > 0] = 1
        
        # Step 1: Edge detection on the wall area
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        roi = gray * wall_mask_binary
        edges = cv2.Canny(roi, 100, 200)
        edges = cv2.dilate(edges, None, iterations=2)
        debug_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        logger.debug(f"Edge detection completed: {np.sum(edges > 0)} edge pixels")
        
        # Step 2: Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        windows = []
        near_rectangular = []
        
        debug_contours_green_blue = image_bgr.copy()
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < (h * w * 0.01) or area > (h * w * 0.5):
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:  # Near-rectangular shapes
                x, y, w_contour, h_contour = cv2.boundingRect(approx)
                # Check if the contour is mostly on the wall
                roi_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(roi_mask, [approx], -1, 255, -1)
                overlap = cv2.bitwise_and(roi_mask, wall_mask)
                overlap_area = np.sum(overlap == 255)
                if overlap_area < 0.5 * area:
                    continue
                near_rectangular.append((x, y, x + w_contour, y + h_contour))
                cv2.rectangle(debug_contours_green_blue, (x, y), (x + w_contour, y + h_contour), (0, 255, 0), 2)
        
        # Step 3: Filter windows by aspect ratio
        debug_contours_green_red = debug_contours_green_blue.copy()
        for rect in near_rectangular:
            x1, y1, x2, y2 = rect
            aspect_ratio = (x2 - x1) / (y2 - y1 + 1e-6)
            if 0.5 <= aspect_ratio <= 2.0:  # Reasonable aspect ratio for windows
                windows.append(rect)
            else:
                cv2.rectangle(debug_contours_green_red, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Draw confirmed windows in green
        for rect in windows:
            x1, y1, x2, y2 = rect
            cv2.rectangle(debug_contours_green_blue, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(debug_contours_green_red, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        logger.debug(f"Detected {len(windows)} windows, {len(near_rectangular)} near-rectangular, {len(near_rectangular) - len(windows)} aspect outliers")
        return windows, debug_edges, debug_contours_green_blue, debug_contours_green_red
    except Exception as e:
        logger.error(f"Error detecting windows: {str(e)}")
        return [], None, None, None
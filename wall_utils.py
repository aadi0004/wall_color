
# Utility functions for wall detection, color recommendation, and accessory suggestion
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

def suggest_accessories(image_bgr, wall_mask, wall_color_bgr, max_suggestions=2):
    """Suggest accessories based on wall color and room layout."""
    try:
        suggestions = []
        h, w = image_bgr.shape[:2]
        wall_area = np.sum(wall_mask == 255)
        if wall_area < (h * w * 0.1):
            logger.warning("Wall area too small for accessory suggestion")
            return [{"success": False, "reason": "Wall area too small"}] * max_suggestions
        
        # Convert wall color to HSV
        wall_rgb = np.uint8([[wall_color_bgr]])
        wall_hsv = cv2.cvtColor(wall_rgb, cv2.COLOR_BGR2HSV)[0][0]
        
        # Suggest sofa
        sofa_hue = (wall_hsv[0] + 30) % 180  # Analogous color
        sofa_hsv = np.array([sofa_hue, wall_hsv[1], wall_hsv[2]], dtype=np.uint8)
        sofa_color = cv2.cvtColor(np.uint8([[sofa_hsv]]), cv2.COLOR_HSV2BGR)[0][0]
        suggestions.append({
            "success": True,
            "type": "sofa",
            "color": sofa_color,
            "reason": "Analogous color to complement wall"
        })
        
        # Suggest chair
        chair_hue = (wall_hsv[0] + 150) % 180  # Complementary color
        chair_hsv = np.array([chair_hue, wall_hsv[1], wall_hsv[2]], dtype=np.uint8)
        chair_color = cv2.cvtColor(np.uint8([[chair_hsv]]), cv2.COLOR_HSV2BGR)[0][0]
        suggestions.append({
            "success": True,
            "type": "chair",
            "color": chair_color,
            "reason": "Complementary color to contrast wall"
        })
        
        logger.debug(f"Accessory suggestions: {suggestions}")
        return suggestions
    except Exception as e:
        logger.error(f"Error suggesting accessories: {str(e)}")
        return [{"success": False, "reason": str(e)}] * max_suggestions

def apply_accessory(image_bgr, accessory_type, color_bgr, placement_rect=None):
    """Apply accessory to the image at the specified position."""
    try:
        h, w = image_bgr.shape[:2]
        accessory_path = f"assets/{accessory_type}.png"
        accessory_img = cv2.imread(accessory_path, cv2.IMREAD_UNCHANGED)
        if accessory_img is None:
            logger.error(f"Accessory image not found: {accessory_path}")
            return image_bgr
        
        # Validate accessory image
        if accessory_img.shape[2] != 4:
            logger.error(f"Accessory image {accessory_path} must have alpha channel")
            return image_bgr
        if accessory_img.shape[0] != 1024 or accessory_img.shape[1] != 1024:
            logger.warning(f"Accessory image {accessory_path} should be 1024x1024")
        
        # Determine placement
        if placement_rect:
            x1, y1, x2, y2 = placement_rect[:4]
            rotation = placement_rect[4] if len(placement_rect) > 4 else 0
            new_w = x2 - x1
            new_h = y2 - y1
        else:
            new_w, new_h = w // 4, h // 4
            x1, y1 = w // 4, 3 * h // 4
            rotation = 0
        
        # Resize accessory
        accessory_resized = cv2.resize(accessory_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Apply rotation if needed
        if rotation != 0:
            center = (new_w // 2, new_h // 2)
            M = cv2.getRotationMatrix2D(center, rotation, 1.0)
            accessory_resized = cv2.warpAffine(accessory_resized, M, (new_w, new_h))
        
        # Color the accessory (apply color to non-transparent areas)
        alpha = accessory_resized[:, :, 3] / 255.0
        for c in range(3):
            accessory_resized[:, :, c] = (1 - alpha) * accessory_resized[:, :, c] + alpha * color_bgr[c]
        
        # Overlay accessory on the image
        result = image_bgr.copy()
        x2, y2 = x1 + new_w, y1 + new_h
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            logger.warning(f"Accessory placement out of bounds: ({x1}, {y1}) to ({x2}, {y2})")
            return image_bgr
        
        roi = result[y1:y2, x1:x2]
        alpha_roi = alpha[:, :, np.newaxis]
        for c in range(3):
            roi[:, :, c] = (1 - alpha_roi) * roi[:, :, c] + alpha_roi * accessory_resized[:, :, c]
        result[y1:y2, x1:x2] = roi
        
        logger.debug(f"Applied {accessory_type} at ({x1}, {y1}) to ({x2}, {y2}), color: {color_bgr}")
        return result
    except Exception as e:
        logger.error(f"Error applying accessory: {str(e)}")
        return image_bgr

def suggest_accessory_placement(image_bgr, wall_mask, accessory_type):
    """Suggest placement for an accessory on the wall."""
    try:
        h, w = image_bgr.shape[:2]
        wall_pixels = np.where(wall_mask == 255)
        if len(wall_pixels[0]) < 100:
            logger.warning("Not enough wall pixels for accessory placement")
            return None
        
        # Find the bottom center of the wall area
        y_coords = wall_pixels[0]
        x_coords = wall_pixels[1]
        bottom_y = np.max(y_coords)
        bottom_x = int(np.mean(x_coords[y_coords == bottom_y]))
        
        # Adjust placement based on accessory type
        accessory_w, accessory_h = w // 4, h // 4
        x1 = bottom_x - accessory_w // 2
        y1 = bottom_y - accessory_h
        x2 = x1 + accessory_w
        y2 = y1 + accessory_h
        
        # Ensure within bounds
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        logger.debug(f"Suggested placement for {accessory_type}: ({x1}, {y1}) to ({x2}, {y2})")
        return (x1, y1, x2, y2)
    except Exception as e:
        logger.error(f"Error suggesting accessory placement: {str(e)}")
        return None

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

def suggest_curtains(image_bgr, wall_mask, wall_color_bgr, windows):
    """Suggest curtains based on wall color and window positions."""
    try:
        h, w = image_bgr.shape[:2]
        wall_area = np.sum(wall_mask == 255)
        if wall_area < (h * w * 0.1):
            return {"success": False, "reason": "Wall area too small"}
        if not windows:
            logger.warning("No windows detected for curtain suggestion")
            return {"success": False, "reason": "No windows detected"}
        
        # Convert wall color to HSV
        wall_rgb = np.uint8([[wall_color_bgr if wall_color_bgr is not None else [255, 255, 255]]])
        wall_hsv = cv2.cvtColor(wall_rgb, cv2.COLOR_BGR2HSV)[0][0]
        
        # Suggest curtain color (analogous to wall color)
        curtain_hue = (wall_hsv[0] + 30) % 180
        curtain_hsv = np.array([curtain_hue, wall_hsv[1], wall_hsv[2]], dtype=np.uint8)
        curtain_color = cv2.cvtColor(np.uint8([[curtain_hsv]]), cv2.COLOR_HSV2BGR)[0][0]
        
        # Suggest curtain type based on window size
        largest_window = max(windows, key=lambda rect: (rect[2] - rect[0]) * (rect[3] - rect[1]))
        window_area = (largest_window[2] - largest_window[0]) * (largest_window[3] - largest_window[1])
        curtain_type = "sheer" if window_area > (h * w * 0.1) else "drapes"
        
        logger.debug(f"Suggested {curtain_type} curtains, color: {curtain_color}")
        return {
            "success": True,
            "type": curtain_type,
            "color": curtain_color,
            "reason": f"Based on window size and wall color harmony"
        }
    except Exception as e:
        logger.error(f"Error suggesting curtains: {str(e)}")
        return {"success": False, "reason": str(e)}

def create_test_png(accessory_type):
    """Create a test PNG for the specified accessory."""
    try:
        size = (1024, 1024)
        img = np.zeros((size[0], size[1], 4), dtype=np.uint8)
        # Create a gray square (simulating the accessory) with transparency
        img[:, :, 0:3] = 128  # Gray color (#808080)
        img[:, :, 3] = 255    # Fully opaque
        # Make the center transparent
        margin = 200
        img[margin:-margin, margin:-margin, 3] = 0
        cv2.imwrite(f"assets/{accessory_type}.png", img)
        logger.debug(f"Created test PNG for {accessory_type}")
    except Exception as e:
        logger.error(f"Error creating test PNG for {accessory_type}: {str(e)}")
        raise

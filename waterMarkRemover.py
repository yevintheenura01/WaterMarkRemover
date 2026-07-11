import cv2
import numpy as np
import os
import subprocess
import tempfile
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Global variables for manual watermark selection
watermark_areas = []
selecting = False
current_rect = None

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
PHOTO_REMOVAL_MODES = {
    "Manual brush surrounding pixels": "manual_brush_local",
    "Manual brush mask (least distortion)": "manual_brush",
    "Manual area (best for solid logos)": "manual_area",
    "Manual refined mask (best for photos)": "manual_refined",
    "Manual strong cleanup (stubborn photos)": "manual_strong",
    "Auto light watermark": "auto_light",
    "Auto dark watermark": "auto_dark",
    "Auto translucent watermark": "auto_translucent",
}

def mouse_callback(event, x, y, flags, param):
    """Mouse callback for manual watermark selection"""
    global selecting, current_rect, watermark_areas
    
    if event == cv2.EVENT_LBUTTONDOWN:
        selecting = True
        current_rect = (x, y, x, y)
    elif event == cv2.EVENT_MOUSEMOVE and selecting:
        current_rect = (current_rect[0], current_rect[1], x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        selecting = False
        if abs(current_rect[2] - current_rect[0]) > 0 and abs(current_rect[3] - current_rect[1]) > 0:
            # Add rectangle to watermark areas
            x1, y1, x2, y2 = current_rect
            watermark_areas.append((min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1)))
            print(f"Added watermark area: {watermark_areas[-1]}")

def select_watermark_areas(image):
    """Allow user to manually select watermark areas"""
    try:
        return select_watermark_areas_cv2(image)
    except cv2.error as err:
        print("OpenCV window support is not available. Using Tkinter selector instead.")
        print("OpenCV error:", err)
        return select_watermark_areas_tk(image)

def select_watermark_areas_cv2(image):
    """Allow user to manually select watermark areas in an OpenCV window."""
    global watermark_areas, selecting, current_rect
    watermark_areas = []
    selecting = False
    current_rect = None
    
    display_img = image.copy()
    cv2.namedWindow('Select Watermark Areas', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Select Watermark Areas', mouse_callback)
    
    print("Instructions:")
    print("- Click and drag to select watermark areas")
    print("- Press 'r' to reset all selections")
    print("- Press 'd' to delete last selection")
    print("- Press 's' to save selections and continue")
    print("- Press 'q' to quit without saving")
    
    while True:
        temp_img = display_img.copy()
        
        # Draw existing selections
        for i, (x, y, w, h) in enumerate(watermark_areas):
            cv2.rectangle(temp_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(temp_img, f"Area {i+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw current selection
        if current_rect:
            x1, y1, x2, y2 = current_rect
            cv2.rectangle(temp_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.imshow('Select Watermark Areas', temp_img)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return []
        elif key == ord('r'):
            watermark_areas = []
        elif key == ord('d') and watermark_areas:
            watermark_areas.pop()
    
    cv2.destroyAllWindows()
    return watermark_areas

def select_watermark_areas_tk(image):
    """Allow user to manually select watermark areas in a Tkinter window."""
    selections = []
    current_canvas_rect = {"id": None}
    drag_start = {"x": 0, "y": 0}

    height, width = image.shape[:2]
    max_width = 1100
    max_height = 720
    scale = min(max_width / width, max_height / height, 1.0)
    display_width = max(1, int(width * scale))
    display_height = max(1, int(height * scale))

    display_img = cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_AREA)
    temp_preview = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    temp_preview.close()
    cv2.imwrite(temp_preview.name, display_img)

    root = tk.Tk()
    root.title("Select Watermark Areas")
    root.resizable(True, True)

    status_var = tk.StringVar(
        value="Drag over watermark areas. Use Save when done, or Cancel to skip manual selection."
    )

    canvas = tk.Canvas(root, width=display_width, height=display_height, cursor="crosshair")
    preview_image = tk.PhotoImage(file=temp_preview.name)
    canvas.create_image(0, 0, anchor="nw", image=preview_image)
    canvas.preview_image = preview_image
    canvas.grid(row=0, column=0, columnspan=4, padx=10, pady=(10, 6))

    def canvas_to_image_rect(x1, y1, x2, y2):
        left = int(max(0, min(x1, x2)) / scale)
        top = int(max(0, min(y1, y2)) / scale)
        right = int(min(display_width, max(x1, x2)) / scale)
        bottom = int(min(display_height, max(y1, y2)) / scale)
        return left, top, max(1, right - left), max(1, bottom - top)

    def redraw_selections():
        canvas.delete("selection")
        for index, (x, y, w, h) in enumerate(selections):
            x1 = int(x * scale)
            y1 = int(y * scale)
            x2 = int((x + w) * scale)
            y2 = int((y + h) * scale)
            canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2, tags="selection")
            canvas.create_text(
                x1 + 6,
                max(12, y1 - 10),
                text=f"Area {index + 1}",
                anchor="w",
                fill="red",
                tags="selection",
            )
        status_var.set(f"{len(selections)} area(s) selected.")

    def on_mouse_down(event):
        drag_start["x"] = max(0, min(event.x, display_width))
        drag_start["y"] = max(0, min(event.y, display_height))
        if current_canvas_rect["id"] is not None:
            canvas.delete(current_canvas_rect["id"])
        current_canvas_rect["id"] = canvas.create_rectangle(
            drag_start["x"],
            drag_start["y"],
            drag_start["x"],
            drag_start["y"],
            outline="lime",
            width=2,
        )

    def on_mouse_move(event):
        if current_canvas_rect["id"] is None:
            return
        x = max(0, min(event.x, display_width))
        y = max(0, min(event.y, display_height))
        canvas.coords(current_canvas_rect["id"], drag_start["x"], drag_start["y"], x, y)

    def on_mouse_up(event):
        if current_canvas_rect["id"] is None:
            return
        x = max(0, min(event.x, display_width))
        y = max(0, min(event.y, display_height))
        if abs(x - drag_start["x"]) >= 3 and abs(y - drag_start["y"]) >= 3:
            rect = canvas_to_image_rect(drag_start["x"], drag_start["y"], x, y)
            selections.append(rect)
            print(f"Added watermark area: {rect}")
        canvas.delete(current_canvas_rect["id"])
        current_canvas_rect["id"] = None
        redraw_selections()

    def reset():
        selections.clear()
        redraw_selections()

    def delete_last():
        if selections:
            selections.pop()
            redraw_selections()

    def save():
        root.destroy()

    def cancel():
        selections.clear()
        root.destroy()

    canvas.bind("<ButtonPress-1>", on_mouse_down)
    canvas.bind("<B1-Motion>", on_mouse_move)
    canvas.bind("<ButtonRelease-1>", on_mouse_up)

    tk.Label(root, textvariable=status_var, anchor="w").grid(row=1, column=0, padx=10, pady=(0, 8), sticky="w")
    tk.Button(root, text="Reset", command=reset, width=12).grid(row=1, column=1, padx=4, pady=(0, 8))
    tk.Button(root, text="Delete Last", command=delete_last, width=12).grid(row=1, column=2, padx=4, pady=(0, 8))
    tk.Button(root, text="Save", command=save, width=12).grid(row=2, column=1, padx=4, pady=(0, 10))
    tk.Button(root, text="Cancel", command=cancel, width=12).grid(row=2, column=2, padx=4, pady=(0, 10))

    root.protocol("WM_DELETE_WINDOW", cancel)
    root.mainloop()

    if os.path.exists(temp_preview.name):
        os.remove(temp_preview.name)

    return selections

def select_watermark_mask_tk(image):
    """Paint an exact watermark mask in a Tkinter window."""
    height, width = image.shape[:2]
    max_width = 1100
    max_height = 720
    scale = min(max_width / width, max_height / height, 1.0)
    display_width = max(1, int(width * scale))
    display_height = max(1, int(height * scale))

    display_img = cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_AREA)
    temp_preview = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    temp_preview.close()
    cv2.imwrite(temp_preview.name, display_img)

    mask = np.zeros((height, width), dtype=np.uint8)
    result = {"cancelled": False}
    last_point = {"x": None, "y": None}

    root = tk.Tk()
    root.title("Paint Watermark Mask")
    root.resizable(True, True)

    brush_size_var = tk.IntVar(value=18)
    erase_var = tk.BooleanVar(value=False)
    status_var = tk.StringVar(value="Paint only the watermark pixels, then click Save.")

    canvas = tk.Canvas(root, width=display_width, height=display_height, cursor="crosshair")
    preview_image = tk.PhotoImage(file=temp_preview.name)
    canvas.create_image(0, 0, anchor="nw", image=preview_image)
    canvas.preview_image = preview_image
    canvas.grid(row=0, column=0, columnspan=6, padx=10, pady=(10, 6))

    def clamp_point(event):
        return max(0, min(event.x, display_width - 1)), max(0, min(event.y, display_height - 1))

    def paint_at(display_x, display_y, previous=None):
        brush_display_radius = max(2, brush_size_var.get() // 2)
        image_x = int(display_x / scale)
        image_y = int(display_y / scale)
        image_radius = max(1, int(brush_display_radius / scale))
        color = 0 if erase_var.get() else 255
        outline = "#00c8ff" if erase_var.get() else "#ff2d2d"

        if previous:
            prev_image_x = int(previous[0] / scale)
            prev_image_y = int(previous[1] / scale)
            cv2.line(mask, (prev_image_x, prev_image_y), (image_x, image_y), color, image_radius * 2)
            canvas.create_line(
                previous[0],
                previous[1],
                display_x,
                display_y,
                fill=outline,
                width=max(2, brush_display_radius * 2),
                capstyle=tk.ROUND,
                tags="brush",
            )
        else:
            cv2.circle(mask, (image_x, image_y), image_radius, color, -1)
            canvas.create_oval(
                display_x - brush_display_radius,
                display_y - brush_display_radius,
                display_x + brush_display_radius,
                display_y + brush_display_radius,
                outline=outline,
                width=2,
                tags="brush",
            )

        if erase_var.get():
            cv2.circle(mask, (image_x, image_y), image_radius, 0, -1)
        else:
            cv2.circle(mask, (image_x, image_y), image_radius, 255, -1)

        status_var.set(f"Masked pixels: {cv2.countNonZero(mask)}")

    def on_mouse_down(event):
        x, y = clamp_point(event)
        last_point["x"] = x
        last_point["y"] = y
        paint_at(x, y)

    def on_mouse_move(event):
        x, y = clamp_point(event)
        previous = (last_point["x"], last_point["y"])
        paint_at(x, y, previous)
        last_point["x"] = x
        last_point["y"] = y

    def on_mouse_up(_event):
        last_point["x"] = None
        last_point["y"] = None

    def reset():
        mask[:, :] = 0
        canvas.delete("brush")
        status_var.set("Mask cleared.")

    def save():
        root.destroy()

    def cancel():
        result["cancelled"] = True
        mask[:, :] = 0
        root.destroy()

    canvas.bind("<ButtonPress-1>", on_mouse_down)
    canvas.bind("<B1-Motion>", on_mouse_move)
    canvas.bind("<ButtonRelease-1>", on_mouse_up)

    tk.Label(root, textvariable=status_var, anchor="w").grid(row=1, column=0, padx=10, pady=(0, 8), sticky="w")
    tk.Label(root, text="Brush").grid(row=1, column=1, padx=(4, 0), pady=(0, 8), sticky="e")
    tk.Scale(
        root,
        from_=4,
        to=60,
        orient="horizontal",
        variable=brush_size_var,
        length=140,
    ).grid(row=1, column=2, padx=4, pady=(0, 8), sticky="w")
    tk.Checkbutton(root, text="Erase", variable=erase_var).grid(row=1, column=3, padx=4, pady=(0, 8))
    tk.Button(root, text="Reset", command=reset, width=10).grid(row=1, column=4, padx=4, pady=(0, 8))
    tk.Button(root, text="Save", command=save, width=10).grid(row=2, column=2, padx=4, pady=(0, 10))
    tk.Button(root, text="Cancel", command=cancel, width=10).grid(row=2, column=3, padx=4, pady=(0, 10))

    root.protocol("WM_DELETE_WINDOW", cancel)
    root.mainloop()

    if os.path.exists(temp_preview.name):
        os.remove(temp_preview.name)

    if result["cancelled"]:
        return np.zeros((height, width), dtype=np.uint8)
    return smooth_mask(mask)

def create_watermark_mask(image, watermark_areas, padding=5):
    """Create mask from selected watermark areas"""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    for (x, y, w, h) in watermark_areas:
        # Add some padding around the selected area
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        mask[y1:y2, x1:x2] = 255
    
    return smooth_mask(mask)

def smooth_mask(mask):
    """Clean the mask while keeping it binary for stable inpainting."""
    if cv2.countNonZero(mask) == 0:
        return mask

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    return mask

def remove_small_mask_components(mask, min_area=8, max_area_ratio=0.18):
    """Remove tiny specks and large frame-wide regions that are unlikely to be watermarks."""
    if cv2.countNonZero(mask) == 0:
        return mask

    binary = smooth_mask(mask)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    image_area = mask.shape[0] * mask.shape[1]
    max_area = max(min_area, int(image_area * max_area_ratio))
    cleaned = np.zeros_like(mask)

    for label in range(1, component_count):
        area = stats[label, cv2.CC_STAT_AREA]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        if area < min_area or area > max_area:
            continue
        if width > mask.shape[1] * 0.92 or height > mask.shape[0] * 0.92:
            continue
        cleaned[labels == label] = 255

    return smooth_mask(cleaned)

def create_contrast_watermark_mask(image, target="both", strength=2):
    """Find watermark-like pixels by comparing the image with a blurred local background."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    strength = max(1, min(3, int(strength)))

    blur_size = 21 + (strength * 10)
    if blur_size % 2 == 0:
        blur_size += 1
    background = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    residual = cv2.subtract(gray, background)
    inverse_residual = cv2.subtract(background, gray)

    residual_threshold = 10 + (3 - strength) * 4
    if target == "light":
        contrast_mask = cv2.inRange(residual, residual_threshold, 255)
    elif target == "dark":
        contrast_mask = cv2.inRange(inverse_residual, residual_threshold, 255)
    else:
        contrast_mask = cv2.bitwise_or(
            cv2.inRange(residual, residual_threshold, 255),
            cv2.inRange(inverse_residual, residual_threshold, 255),
        )

    saturation = hsv[:, :, 1]
    low_saturation = cv2.inRange(saturation, 0, 95 + (strength * 18))

    mask = cv2.bitwise_and(contrast_mask, low_saturation)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=max(1, strength - 1))
    return remove_small_mask_components(mask, min_area=10 + (strength * 6), max_area_ratio=0.25)

def feathered_blend(original, repaired, mask, strength=2):
    """Blend repaired pixels back gently so surrounding photo detail is preserved."""
    if cv2.countNonZero(mask) == 0:
        return original

    strength = max(1, min(3, int(strength)))
    kernel = np.ones((3, 3), np.uint8)
    blend_mask = cv2.dilate(mask, kernel, iterations=max(1, strength - 1))
    blur_size = 5 if strength < 3 else 7
    alpha = cv2.GaussianBlur(blend_mask, (blur_size, blur_size), 0).astype(np.float32) / 255.0
    alpha *= 0.82 if strength == 1 else 0.92 if strength == 2 else 1.0
    alpha = alpha[:, :, None]
    return np.clip((repaired.astype(np.float32) * alpha) + (original.astype(np.float32) * (1.0 - alpha)), 0, 255).astype(np.uint8)

def create_refined_photo_mask(image, watermark_areas, strength=2):
    """Create a tighter mask for watermark-looking pixels inside selected photo areas."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    strength = max(1, min(3, int(strength)))

    for (x, y, w, h) in watermark_areas:
        padding = 3 + (strength * 2)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)

        roi_gray = gray[y1:y2, x1:x2]
        roi_hsv = hsv[y1:y2, x1:x2]
        roi_image = image[y1:y2, x1:x2]
        if roi_gray.size == 0:
            continue

        saturation = roi_hsv[:, :, 1]
        value = roi_hsv[:, :, 2]
        bright_percentile = 92 - (strength * 6)
        dark_percentile = 8 + (strength * 6)
        bright_limit = max(170, int(np.percentile(roi_gray, bright_percentile)))
        dark_limit = min(95, int(np.percentile(roi_gray, dark_percentile)))
        sat_limit = 85 + (strength * 18)
        bright_value_limit = max(170, 225 - (strength * 15))
        dark_value_limit = min(100, 45 + (strength * 18))

        bright_text = cv2.inRange(roi_gray, bright_limit, 255)
        dark_text = cv2.inRange(roi_gray, 0, dark_limit)
        low_saturation_logo = cv2.inRange(saturation, 0, sat_limit)
        high_or_low_value = cv2.bitwise_or(
            cv2.inRange(value, bright_value_limit, 255),
            cv2.inRange(value, 0, dark_value_limit),
        )

        roi_mask = cv2.bitwise_or(bright_text, dark_text)
        roi_mask = cv2.bitwise_or(roi_mask, cv2.bitwise_and(low_saturation_logo, high_or_low_value))
        translucent_mask = create_contrast_watermark_mask(roi_image, target="both", strength=strength)
        roi_mask = cv2.bitwise_or(roi_mask, translucent_mask)
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        roi_mask = cv2.dilate(roi_mask, kernel, iterations=max(1, strength - 1))
        roi_mask = remove_small_mask_components(roi_mask, min_area=6, max_area_ratio=0.35)

        mask[y1:y2, x1:x2] = roi_mask

    return smooth_mask(mask)

def create_auto_photo_mask(image, mode, strength=2):
    """Automatically detect simple light, dark, or translucent photo watermarks."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    kernel = np.ones((3, 3), np.uint8)
    strength = max(1, min(3, int(strength)))

    if mode == "auto_translucent":
        mask = create_contrast_watermark_mask(image, target="both", strength=strength)
    elif mode == "auto_dark":
        mask = cv2.inRange(gray, 0, 35 + (strength * 18))
        mask = cv2.bitwise_or(mask, create_contrast_watermark_mask(image, target="dark", strength=strength))
    else:
        mask = cv2.inRange(gray, 245 - (strength * 15), 255)
        mask = cv2.bitwise_or(mask, create_contrast_watermark_mask(image, target="light", strength=strength))

    low_saturation = cv2.inRange(saturation, 0, 75 + (strength * 18))
    mask = cv2.bitwise_and(mask, low_saturation)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=max(1, strength - 1))
    return remove_small_mask_components(mask, min_area=12 + (strength * 8), max_area_ratio=0.25)

def inpaint_photo(image, mask, strength=2):
    """Remove selected watermark pixels with an inpainting strategy based on strength."""
    strength = max(1, min(3, int(strength)))
    radius = 2 + strength

    if strength >= 3:
        first_pass = cv2.inpaint(image, mask, radius + 1, cv2.INPAINT_TELEA)
        repaired = cv2.inpaint(first_pass, mask, radius, cv2.INPAINT_NS)
        return feathered_blend(image, repaired, mask, strength)

    repaired = cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)
    return feathered_blend(image, repaired, mask, strength)

def inpaint_photo_locally(image, mask, strength=1):
    """Repair each masked area from its immediate surrounding pixels."""
    if cv2.countNonZero(mask) == 0:
        return image

    strength = max(1, min(3, int(strength)))
    result = image.copy()
    binary_mask = smooth_mask(mask)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    image_height, image_width = image.shape[:2]

    for label in range(1, component_count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < 4:
            continue

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        padding = max(18, min(80, int(max(w, h) * 0.45) + (strength * 8)))

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image_width, x + w + padding)
        y2 = min(image_height, y + h + padding)

        roi = result[y1:y2, x1:x2].copy()
        roi_mask = np.where(labels[y1:y2, x1:x2] == label, 255, 0).astype(np.uint8)
        roi_mask = smooth_mask(roi_mask)

        radius = 2 if strength == 1 else 3 if strength == 2 else 4
        repaired_roi = cv2.inpaint(roi, roi_mask, radius, cv2.INPAINT_TELEA)
        if strength >= 3:
            repaired_roi = cv2.inpaint(repaired_roi, roi_mask, radius, cv2.INPAINT_NS)

        result[y1:y2, x1:x2] = feathered_blend(roi, repaired_roi, roi_mask, strength)

    return result

def remove_watermark_image(image_path, custom_name="", removal_mode="manual_refined", removal_strength=2):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image:", image_path)
        return
    removal_strength = max(1, min(3, int(removal_strength)))

    if removal_mode in ("manual_brush", "manual_brush_local"):
        print("Paint the watermark pixels to remove:")
        mask = select_watermark_mask_tk(img)
    elif removal_mode in ("auto_light", "auto_dark"):
        print("Using automatic photo watermark detection...")
        mask = create_auto_photo_mask(img, removal_mode, removal_strength)
    else:
        print("Select watermark areas to remove:")
        watermark_areas = select_watermark_areas(img)
        
        if not watermark_areas:
            print("No watermark areas selected. Using automatic light watermark detection...")
            mask = create_auto_photo_mask(img, "auto_light", removal_strength)
        elif removal_mode == "manual_area":
            print(f"Using {len(watermark_areas)} full manually selected watermark areas")
            mask = create_watermark_mask(img, watermark_areas, padding=removal_strength)
        elif removal_mode == "manual_strong":
            print(f"Using strong photo cleanup inside {len(watermark_areas)} selected watermark areas")
            mask = create_refined_photo_mask(img, watermark_areas, strength=3)
        else:
            print(f"Using refined photo mask inside {len(watermark_areas)} selected watermark areas")
            mask = create_refined_photo_mask(img, watermark_areas, strength=removal_strength)

    if cv2.countNonZero(mask) == 0:
        print("No watermark pixels detected. Try brush mask mode for this photo.")
        return

    if removal_mode == "manual_brush_local":
        result = inpaint_photo_locally(img, mask, strength=removal_strength)
    else:
        result = inpaint_photo(img, mask, strength=3 if removal_mode == "manual_strong" else removal_strength)

    # Save result with custom name or default
    if custom_name:
        # If user provided custom name, use it
        if not custom_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp')):
            custom_name += '.png'  # Default to PNG for lossless quality
        out_path = os.path.join(os.path.dirname(image_path), custom_name)
    else:
        # Default naming - preserve original format for quality
        base_name = os.path.splitext(image_path)[0]
        original_ext = os.path.splitext(image_path)[1].lower()
        if original_ext in ['.jpg', '.jpeg']:
            out_path = base_name + "_nowm.jpg"
        elif original_ext in ['.png']:
            out_path = base_name + "_nowm.png"
        elif original_ext in ['.tiff', '.tif']:
            out_path = base_name + "_nowm.tiff"
        elif original_ext in ['.webp']:
            out_path = base_name + "_nowm.webp"
        else:
            out_path = base_name + "_nowm.png"  # Default to PNG for lossless
    
    # High-quality image saving with appropriate parameters
    if out_path.lower().endswith(('.jpg', '.jpeg')):
        # For JPEG, use high quality (95-100)
        cv2.imwrite(out_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif out_path.lower().endswith('.png'):
        # For PNG, use maximum compression level for quality
        cv2.imwrite(out_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    elif out_path.lower().endswith(('.tiff', '.tif')):
        # For TIFF, use lossless compression
        cv2.imwrite(out_path, result, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
    elif out_path.lower().endswith('.webp'):
        cv2.imwrite(out_path, result, [cv2.IMWRITE_WEBP_QUALITY, 95])
    else:
        # Default high-quality save
        cv2.imwrite(out_path, result)
    
    print("High-quality processed image saved to", out_path)

def remove_watermark_video(video_path, custom_name="", aspect_ratio=None, crop_enabled=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use more compatible codec - try multiple options
    # Try H.264 first, fallback to MP4V if not available
    try:
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        # Test if this codec works by creating a test writer
        test_writer = cv2.VideoWriter('test.mp4', fourcc, fps, (width, height))
        if not test_writer.isOpened():
            raise Exception("H264 codec not available")
        test_writer.release()
        if os.path.exists('test.mp4'):
            os.remove('test.mp4')
    except:
        # Fallback to MP4V codec which is more widely supported
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print("Using MP4V codec (H.264 not available)")
    
    # Get original video properties for quality preservation
    original_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    original_bitrate = cap.get(cv2.CAP_PROP_BITRATE)
    
    # Save result with custom name or default
    if custom_name:
        # If user provided custom name, use it
        if not custom_name.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            custom_name += '.mp4'  # Default to MP4 if no extension
        out_path = os.path.join(os.path.dirname(video_path), custom_name)
    else:
        # Default naming
        out_path = os.path.splitext(video_path)[0] + "_nowm.mp4"
    
    # Get first frame for watermark selection
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read first frame from video")
        cap.release()
        return
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print("Select watermark areas on the first frame:")
    watermark_areas = select_watermark_areas(first_frame)
    
    if not watermark_areas:
        print("No watermark areas selected. Using automatic translucent watermark detection...")
        mask = create_auto_photo_mask(first_frame, "auto_translucent", strength=2)
    else:
        print(f"Using {len(watermark_areas)} manually selected watermark areas")
        mask = create_watermark_mask(first_frame, watermark_areas, padding=3)
    
    # Compute crop rectangle based on selected aspect ratio.
    # If crop is disabled or no ratio is provided, preserve original frame size.
    if crop_enabled and aspect_ratio and aspect_ratio[0] > 0 and aspect_ratio[1] > 0:
        target_aspect = aspect_ratio[0] / float(aspect_ratio[1])
        current_aspect = width / float(height) if height != 0 else target_aspect
        if current_aspect > target_aspect:
            # Too wide: reduce width
            crop_h = height
            crop_w = int(round(crop_h * target_aspect))
        else:
            # Too tall: reduce height
            crop_w = width
            crop_h = int(round(crop_w / target_aspect))
        # Ensure even dimensions for codec compatibility
        crop_w = max(2, crop_w - (crop_w % 2))
        crop_h = max(2, crop_h - (crop_h % 2))
        crop_x = max(0, (width - crop_w) // 2)
        crop_y = max(0, (height - crop_h) // 2)
        print(f"Cropping video to aspect ratio {aspect_ratio[0]}:{aspect_ratio[1]}")
    else:
        crop_x, crop_y = 0, 0
        crop_w, crop_h = width, height
        print("Cropping disabled. Keeping original video dimensions.")

    # Create temporary video file without audio first, using cropped size
    temp_video = tempfile.NamedTemporaryFile(suffix="_no_audio.mp4", delete=False).name
    out = cv2.VideoWriter(temp_video, fourcc, fps, (crop_w, crop_h))
    
    # Check if VideoWriter was successfully created
    if not out.isOpened():
        print("Error: Could not create VideoWriter. Trying alternative codec...")
        # Try alternative codecs
        alternative_codecs = ['mp4v', 'XVID', 'MJPG']
        for codec in alternative_codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(temp_video, fourcc, fps, (crop_w, crop_h))
                if out.isOpened():
                    print(f"Successfully using {codec} codec")
                    break
            except:
                continue
        
        if not out.isOpened():
            print("Error: Could not initialize any video codec. Please check your OpenCV installation.")
            cap.release()
            return

    print("Processing video frames...")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # High-quality inpainting for each frame
        if watermark_areas:
            # Apply the same selected watermark areas to each frame.
            result = inpaint_photo(frame, mask, strength=2)
        else:
            # Recalculate automatic masks per frame for moving or fading overlays.
            frame_mask = create_auto_photo_mask(frame, "auto_translucent", strength=2)
            if cv2.countNonZero(frame_mask) == 0:
                result = frame
            else:
                result = inpaint_photo(frame, frame_mask, strength=2)
        
        # Center-crop to 4:5 aspect before writing
        cropped = result[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        out.write(cropped)
        frame_count += 1
        
        if frame_count % 30 == 0:  # Progress indicator
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    
    # Now merge with original audio using ffmpeg
    print("Merging with original audio...")
    try:
        # Check if ffmpeg is available
        ffmpeg_check = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if ffmpeg_check.returncode != 0:
            raise FileNotFoundError("FFmpeg not found")
        
        # First, check if original video has audio
        check_audio_cmd = [
            'ffprobe', '-v', 'quiet', '-select_streams', 'a', 
            '-show_entries', 'stream=codec_name', '-of', 'csv=p=0', video_path
        ]
        
        audio_check = subprocess.run(check_audio_cmd, capture_output=True, text=True)
        has_audio = audio_check.returncode == 0 and audio_check.stdout.strip()
        
        if has_audio:
            print("Original video has audio. Merging with processed video...")
            # Merge video with original audio using high-quality settings
            cmd = [
                'ffmpeg', '-y',  # -y to overwrite output file
                '-i', temp_video,  # Input video (no audio)
                '-i', video_path,  # Original video (for audio)
                '-c:v', 'libx264',  # H.264 codec for best compatibility and quality
                '-c:a', 'aac',     # AAC audio codec
                '-map', '0:v:0',   # Use video from first input
                '-map', '1:a:0',   # Use audio from second input
                '-shortest',       # End when shortest stream ends
                '-preset', 'medium', # Use medium preset for better compatibility
                '-crf', '23',      # Constant Rate Factor 23 (good quality, more compatible)
                '-pix_fmt', 'yuv420p',  # Ensure compatibility
                '-movflags', '+faststart',  # Optimize for streaming
                out_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("Processed video with audio saved to", out_path)
            else:
                print("Warning: Could not merge audio. Error:", result.stderr)
                print("Saving video without audio...")
                shutil.move(temp_video, out_path)
        else:
            print("Original video has no audio. Saving processed video...")
            shutil.move(temp_video, out_path)
            
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("Warning: ffmpeg not found or error occurred.")
        print("To preserve audio, install ffmpeg: https://ffmpeg.org/download.html")
        print("Error details:", str(e))
        # Fallback: copy temp video to final location
        shutil.move(temp_video, out_path)
    
    # Clean up temporary file
    if os.path.exists(temp_video):
        os.remove(temp_video)


def get_ratio_selection(ratio_label, custom_width_text, custom_height_text):
    """Parse ratio selection from GUI into (aspect_ratio, crop_enabled)."""
    presets = {
        "4:5": (4, 5),
        "1:1": (1, 1),
        "16:9": (16, 9),
        "9:16": (9, 16),
    }

    if ratio_label == "Original (No crop)":
        return None, False

    if ratio_label == "Custom":
        try:
            w = int(custom_width_text)
            h = int(custom_height_text)
            if w <= 0 or h <= 0:
                raise ValueError
            return (w, h), True
        except ValueError:
            raise ValueError("Custom ratio must use positive whole numbers.")

    if ratio_label in presets:
        return presets[ratio_label], True

    return None, False


def launch_simple_gui():
    """Collect input path and processing options with a simple GUI."""
    result = {}

    root = tk.Tk()
    root.title("Watermark Remover")
    root.geometry("600x450")
    root.resizable(False, False)

    file_path_var = tk.StringVar()
    output_name_var = tk.StringVar()
    ratio_var = tk.StringVar(value="Original (No crop)")
    photo_mode_var = tk.StringVar(value="Manual brush surrounding pixels")
    photo_strength_var = tk.IntVar(value=2)
    custom_w_var = tk.StringVar(value="4")
    custom_h_var = tk.StringVar(value="5")

    def browse_file():
        selected = filedialog.askopenfilename(
            title="Choose Image or Video",
            filetypes=[
                ("Media Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp *.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All Files", "*.*"),
            ],
        )
        if selected:
            file_path_var.set(selected)

    def update_custom_ratio_state(*_args):
        state = "normal" if ratio_var.get() == "Custom" else "disabled"
        custom_w_entry.config(state=state)
        custom_h_entry.config(state=state)

    def process_and_close():
        path = file_path_var.get().strip().strip('"\'')
        if not path:
            messagebox.showerror("Missing Input", "Please choose an image or video file.")
            return
        if not os.path.isfile(path):
            messagebox.showerror("Invalid Path", "The selected file does not exist.")
            return

        ext = os.path.splitext(path)[1].lower()

        if ext not in IMAGE_EXTENSIONS + VIDEO_EXTENSIONS:
            messagebox.showerror("Unsupported File", "Please select a supported image or video file.")
            return

        try:
            aspect_ratio, crop_enabled = get_ratio_selection(
                ratio_var.get(), custom_w_var.get().strip(), custom_h_var.get().strip()
            )
        except ValueError as err:
            messagebox.showerror("Invalid Ratio", str(err))
            return

        result["path"] = path
        result["custom_name"] = output_name_var.get().strip()
        result["photo_removal_mode"] = PHOTO_REMOVAL_MODES.get(
            photo_mode_var.get(), "manual_refined"
        )
        result["photo_removal_strength"] = photo_strength_var.get()
        result["aspect_ratio"] = aspect_ratio
        result["crop_enabled"] = crop_enabled
        root.destroy()

    def cancel_and_close():
        result["cancelled"] = True
        root.destroy()

    tk.Label(root, text="Input file:", anchor="w").grid(row=0, column=0, padx=12, pady=(14, 6), sticky="w")
    tk.Entry(root, textvariable=file_path_var, width=62).grid(row=1, column=0, padx=12, pady=4, sticky="w")
    tk.Button(root, text="Browse", command=browse_file, width=12).grid(row=1, column=1, padx=6, pady=4)

    tk.Label(root, text="Output filename (optional):", anchor="w").grid(row=2, column=0, padx=12, pady=(10, 6), sticky="w")
    tk.Entry(root, textvariable=output_name_var, width=62).grid(row=3, column=0, padx=12, pady=4, sticky="w")

    tk.Label(root, text="Photo watermark mode:", anchor="w").grid(row=4, column=0, padx=12, pady=(12, 6), sticky="w")
    photo_mode_combo = ttk.Combobox(
        root,
        textvariable=photo_mode_var,
        values=list(PHOTO_REMOVAL_MODES.keys()),
        state="readonly",
        width=34,
    )
    photo_mode_combo.grid(row=5, column=0, padx=12, pady=4, sticky="w")

    tk.Label(root, text="Photo cleanup strength:", anchor="w").grid(row=6, column=0, padx=12, pady=(12, 6), sticky="w")
    strength_frame = tk.Frame(root)
    strength_frame.grid(row=7, column=0, padx=12, pady=4, sticky="w")
    tk.Label(strength_frame, text="Low").pack(side="left", padx=(0, 6))
    tk.Scale(
        strength_frame,
        from_=1,
        to=3,
        orient="horizontal",
        variable=photo_strength_var,
        showvalue=True,
        length=180,
    ).pack(side="left")
    tk.Label(strength_frame, text="High").pack(side="left", padx=(6, 0))

    tk.Label(root, text="Video ratio:", anchor="w").grid(row=8, column=0, padx=12, pady=(12, 6), sticky="w")
    ratio_combo = ttk.Combobox(
        root,
        textvariable=ratio_var,
        values=["Original (No crop)", "4:5", "1:1", "16:9", "9:16", "Custom"],
        state="readonly",
        width=24,
    )
    ratio_combo.grid(row=9, column=0, padx=12, pady=4, sticky="w")
    ratio_combo.bind("<<ComboboxSelected>>", update_custom_ratio_state)

    custom_ratio_frame = tk.Frame(root)
    custom_ratio_frame.grid(row=9, column=1, padx=6, pady=4, sticky="w")
    tk.Label(custom_ratio_frame, text="Custom W:H").grid(row=0, column=0, padx=(0, 6), sticky="w")
    custom_w_entry = tk.Entry(custom_ratio_frame, textvariable=custom_w_var, width=5)
    custom_w_entry.grid(row=0, column=1, padx=(0, 3), sticky="w")
    tk.Label(custom_ratio_frame, text=":").grid(row=0, column=2, sticky="w")
    custom_h_entry = tk.Entry(custom_ratio_frame, textvariable=custom_h_var, width=5)
    custom_h_entry.grid(row=0, column=3, padx=(3, 0), sticky="w")

    tk.Label(
        root,
        text="Note: Ratio only affects videos. Images are processed at original size.",
        fg="#555555",
        anchor="w",
    ).grid(row=10, column=0, padx=12, pady=(12, 8), sticky="w")

    button_frame = tk.Frame(root)
    button_frame.grid(row=11, column=0, columnspan=2, pady=(8, 14))
    tk.Button(button_frame, text="Process", command=process_and_close, width=14).pack(side="left", padx=8)
    tk.Button(button_frame, text="Cancel", command=cancel_and_close, width=14).pack(side="left", padx=8)

    update_custom_ratio_state()
    root.mainloop()

    if result.get("cancelled"):
        return None
    return result if result else None

def main():
    settings = launch_simple_gui()
    if settings is None:
        print("Operation cancelled.")
        return

    path = settings["path"]
    custom_name = settings["custom_name"]
    ext = os.path.splitext(path)[1].lower()

    if ext in IMAGE_EXTENSIONS:
        remove_watermark_image(
            path,
            custom_name,
            removal_mode=settings.get("photo_removal_mode", "manual_refined"),
            removal_strength=settings.get("photo_removal_strength", 2),
        )
    elif ext in VIDEO_EXTENSIONS:
        remove_watermark_video(
            path,
            custom_name,
            aspect_ratio=settings["aspect_ratio"],
            crop_enabled=settings["crop_enabled"],
        )
    else:
        print("Unsupported file type.")

if __name__ == "__main__":
    main()

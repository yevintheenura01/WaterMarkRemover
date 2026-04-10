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
        if current_rect[2] > current_rect[0] and current_rect[3] > current_rect[1]:
            # Add rectangle to watermark areas
            x1, y1, x2, y2 = current_rect
            watermark_areas.append((min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1)))
            print(f"Added watermark area: {watermark_areas[-1]}")

def select_watermark_areas(image):
    """Allow user to manually select watermark areas"""
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

def create_watermark_mask(image, watermark_areas):
    """Create mask from selected watermark areas"""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    for (x, y, w, h) in watermark_areas:
        # Add some padding around the selected area
        padding = 5
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        mask[y1:y2, x1:x2] = 255
    
    return mask

def remove_watermark_image(image_path, custom_name=""):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image:", image_path)
        return

    print("Select watermark areas to remove:")
    watermark_areas = select_watermark_areas(img)
    
    if not watermark_areas:
        print("No watermark areas selected. Using automatic detection...")
        # Fallback to automatic detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    else:
        print(f"Using {len(watermark_areas)} manually selected watermark areas")
        mask = create_watermark_mask(img, watermark_areas)

    # High-quality inpainting to remove watermark
    # Use INPAINT_NS for better quality than INPAINT_TELEA
    result = cv2.inpaint(img, mask, 5, cv2.INPAINT_NS)

    # Save result with custom name or default
    if custom_name:
        # If user provided custom name, use it
        if not custom_name.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
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
        print("No watermark areas selected. Using automatic detection...")
        # Improved automatic detection
        gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        # Use adaptive threshold for better detection
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # Invert mask to get bright areas (watermarks)
        mask = cv2.bitwise_not(mask)
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    else:
        print(f"Using {len(watermark_areas)} manually selected watermark areas")
        mask = create_watermark_mask(first_frame, watermark_areas)
    
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
            # Apply the same watermark areas to each frame with high quality
            result = cv2.inpaint(frame, mask, 5, cv2.INPAINT_NS)
        else:
            # For automatic detection, recalculate mask for each frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            frame_mask = cv2.bitwise_not(frame_mask)
            kernel = np.ones((3,3), np.uint8)
            frame_mask = cv2.morphologyEx(frame_mask, cv2.MORPH_CLOSE, kernel)
            frame_mask = cv2.morphologyEx(frame_mask, cv2.MORPH_OPEN, kernel)
            result = cv2.inpaint(frame, frame_mask, 5, cv2.INPAINT_NS)
        
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
    root.geometry("600x320")
    root.resizable(False, False)

    file_path_var = tk.StringVar()
    output_name_var = tk.StringVar()
    ratio_var = tk.StringVar(value="Original (No crop)")
    custom_w_var = tk.StringVar(value="4")
    custom_h_var = tk.StringVar(value="5")

    def browse_file():
        selected = filedialog.askopenfilename(
            title="Choose Image or Video",
            filetypes=[
                ("Media Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.mp4 *.avi *.mov *.mkv *.wmv"),
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
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']

        if ext not in image_exts + video_exts:
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

    tk.Label(root, text="Video ratio:", anchor="w").grid(row=4, column=0, padx=12, pady=(12, 6), sticky="w")
    ratio_combo = ttk.Combobox(
        root,
        textvariable=ratio_var,
        values=["Original (No crop)", "4:5", "1:1", "16:9", "9:16", "Custom"],
        state="readonly",
        width=24,
    )
    ratio_combo.grid(row=5, column=0, padx=12, pady=4, sticky="w")
    ratio_combo.bind("<<ComboboxSelected>>", update_custom_ratio_state)

    custom_ratio_frame = tk.Frame(root)
    custom_ratio_frame.grid(row=5, column=1, padx=6, pady=4, sticky="w")
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
    ).grid(row=6, column=0, padx=12, pady=(12, 8), sticky="w")

    button_frame = tk.Frame(root)
    button_frame.grid(row=7, column=0, columnspan=2, pady=(8, 14))
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

    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
        remove_watermark_image(path, custom_name)
    elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
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

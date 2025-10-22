import cv2
import numpy as np
import os
import subprocess
import tempfile
import shutil

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

    # Inpainting to remove watermark
    result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    # Save result with custom name or default
    if custom_name:
        # If user provided custom name, use it
        if not custom_name.endswith(('.png', '.jpg', '.jpeg')):
            custom_name += '.png'  # Default to PNG if no extension
        out_path = os.path.join(os.path.dirname(image_path), custom_name)
    else:
        # Default naming
        out_path = os.path.splitext(image_path)[0] + "_nowm.png"
    
    cv2.imwrite(out_path, result)
    print("Processed image saved to", out_path)

def remove_watermark_video(video_path, custom_name=""):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video:", video_path)
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Save result with custom name or default
    if custom_name:
        # If user provided custom name, use it
        if not custom_name.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            custom_name += '.mp4'  # Default to MP4 if no extension
        out_path = os.path.join(os.path.dirname(video_path), custom_name)
    else:
        # Default naming
        out_path = os.path.splitext(video_path)[0] + "_nowm.mp4"
    
    # Create temporary video file without audio first
    temp_video = os.path.join(tempfile.gettempdir(), "temp_video_no_audio.mp4")
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

    print("Processing video frames...")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        result = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
        out.write(result)
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
            # Merge video with original audio
            cmd = [
                'ffmpeg', '-y',  # -y to overwrite output file
                '-i', temp_video,  # Input video (no audio)
                '-i', video_path,  # Original video (for audio)
                '-c:v', 'libx264',  # Re-encode video for compatibility
                '-c:a', 'aac',     # Re-encode audio as AAC
                '-map', '0:v:0',   # Use video from first input
                '-map', '1:a:0',   # Use audio from second input
                '-shortest',       # End when shortest stream ends
                '-preset', 'fast', # Faster encoding
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

def main():
    path = input("Enter image or video file path: ").strip('"\'')
    ext = os.path.splitext(path)[1].lower()
    
    # Ask user for custom output name
    custom_name = input("Enter custom output filename (or press Enter for default): ").strip()
    
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
        remove_watermark_image(path, custom_name)
    elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
        remove_watermark_video(path, custom_name)
    else:
        print("Unsupported file type.")

if __name__ == "__main__":
    main()

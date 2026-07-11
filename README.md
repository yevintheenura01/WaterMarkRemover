# WaterMarkRemover

Remove watermarks from images and videos with high quality, using manual area selection or automatic detection fallback. Supports a wide range of file formats and works with both images and videos, including merging audio with processed videos (using ffmpeg).

## Features
- **Manual Watermark Area Selection:** Select watermarks on images or the first frame of videos using an interactive OpenCV window.
- **Photo Watermark Modes:** For photos, choose a surrounding-pixels brush repair for the least distortion, a painted brush mask, full selected-area removal, a refined mask, stronger stubborn-watermark cleanup, or automatic light/dark/translucent watermark detection.
- **Automatic Detection Fallback:** If you don't select any regions, the script tries to detect low-saturation watermark pixels using brightness and local-contrast cues.
- **High-Quality Inpainting:** Uses OpenCV inpainting with feathered blending to reduce hard edges and visible smearing.
- **Video Support:** Processes every frame of a video, removes watermarks, optionally crops to a selected aspect ratio, and merges original audio back using ffmpeg when available.
- **Smart Output Naming:** Automatically names the output to avoid overwriting originals, or lets you specify a custom filename.
- **Cross-Platform:** Works on Windows, macOS, and Linux (as long as Python and dependencies are installed).

## Requirements
- Python 3.6+
- [OpenCV (cv2)](https://pypi.org/project/opencv-python/)
- [NumPy](https://pypi.org/project/numpy/)
- [ffmpeg](https://ffmpeg.org/download.html) (for merging audio with processed video; required for best results, invoked via command-line)

Install required Python packages:
```bash
pip install opencv-python numpy
```

For full video processing (with audio merging), install ffmpeg and ensure it is in your system PATH.

## Installation
1. Clone or download this repository.
2. Install Python requirements as described above.
3. (Optional, but highly recommended) Install ffmpeg for video audio merging.

## Usage
Run the main script from the command line:

```bash
python waterMarkRemover.py
```

You will be prompted for the file path (image/video) and an optional output filename.

### Image Example
```
Enter image or video file path: myphoto_with_watermark.jpg
Enter custom output filename (or press Enter for default): cleaned_photo.png
```
- An OpenCV window will appear.
- In the GUI, choose a photo watermark mode:
    - **Manual brush surrounding pixels:** Paint only the watermark pixels. Each painted area is repaired from its immediate surrounding pixels to reduce distortion.
    - **Manual brush mask (least distortion):** Paint only the watermark pixels. This is best when a watermark overlaps furniture, faces, fabric, plants, rugs, or other detailed areas.
    - **Manual refined mask (best for photos):** Select the watermark area; the script tries to remove only logo/text-like pixels inside that selection.
    - **Manual area (best for solid logos):** Select the exact watermark region; the full region is inpainted.
    - **Manual strong cleanup (stubborn photos):** Select the watermark area and use a more aggressive mask plus two-pass inpainting.
    - **Auto light watermark / Auto dark watermark:** Skip manual selection and let the script look for simple light or dark watermark pixels.
    - **Auto translucent watermark:** Skip manual selection and let the script look for low-saturation overlays that differ subtly from the surrounding image.
- Use the photo cleanup strength slider:
    - **1:** Softer cleanup for delicate textures.
    - **2:** Balanced default.
    - **3:** Stronger cleanup for visible leftover watermark traces.
- **Instructions:**
    - Click and drag to select one or more watermark areas.
    - Press `r` to reset all selections.
    - Press `d` to delete last selection.
    - Press `s` to save your selection and process the image.
    - Press `q` to quit without saving.
- The script will process and output a high-quality image, e.g., `cleaned_photo.png` or `myphoto_with_watermark_nowm.jpg`.

### Video Example
```
Enter image or video file path: video_with_watermark.mp4
Enter custom output filename (or press Enter for default):
```
- Select watermark areas on the first frame as with images.
- If you skip selection, the script uses automatic translucent watermark detection on each frame.
- The script processes each frame, removes the watermark, and crops only when a video ratio is selected.
- If ffmpeg is installed, the final video merges the original audio track.

### Automatic Detection Fallback
- If you don't select any region(s), the script tries to automatically find and remove translucent watermark-like areas.
- For best quality, manual selection is recommended.

## Notes
- The script does not permanently overwrite your original files.
- Output will be saved either to the directory of the original, with a `_nowm` (no watermark) suffix, or under a name you choose.
- For best results on videos, ensure ffmpeg is installed and available in your PATH.
- Handles most common formats: .jpg, .jpeg, .png, .bmp, .tiff, .mp4, .avi, .mov, .mkv



## Contributing
Pull requests and suggestions are welcome!

---

**Developed by yevintheenura01**

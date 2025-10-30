# WaterMarkRemover

Remove watermarks from images and videos with high quality, using manual area selection or automatic detection fallback. Supports a wide range of file formats and works with both images and videos, including merging audio with processed videos (using ffmpeg).

## Features
- **Manual Watermark Area Selection:** Select watermarks on images or the first frame of videos using an interactive OpenCV window.
- **Automatic Detection Fallback:** If you don't select any regions, the script tries to detect bright watermark areas automatically.
- **High-Quality Inpainting:** Uses advanced inpainting algorithms (OpenCV INPAINT_NS) for best visual results.
- **Video Support:** Processes every frame of a video, removes watermarks, center-crops to 4:5 aspect ratio for compatibility, and (optionally) merges original audio back using ffmpeg.
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
- The script processes each frame, removes the watermark, and center-crops the video to 4:5 ratio.
- If ffmpeg is installed, the final video merges the original audio track.

### Automatic Detection Fallback
- If you don't select any region(s), the script tries to automatically find and remove bright watermark areas.
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

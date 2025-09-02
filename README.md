# FaceCrop-Pro

This script processes images in a directory, detects faces, and saves cropped face images with various processing options.

## Features
- Face detection using Haar Cascade classifier
- Blurry image detection (using Laplacian variance)
- Optional duplicate face filtering (using histogram comparison)
- Configurable padding around detected faces
- Grayscale conversion option
- Fixed output size (224x224 by default)

## Usage
1. Place input images in `pre_dataset/` directory (supports JPG format)
2. Configure settings in the script:
   - Input/output directories
   - Grayscale conversion
   - Padding size
   - Duplicate checking
   - Blur threshold
   - Output face size
3. Run the script: `python script_name.py`

Output images will be saved in `new_dataset/` directory with filenames in format: `originalname_faceX.jpg`

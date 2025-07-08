# CT Reconstruction Pipeline Documentation

## Overview
This pipeline performs 3D fan-beam CT reconstruction from 2D projection data. It handles:
- Projection loading and preprocessing
- Sinogram alignment and normalization
- GPU-accelerated reconstruction
- Visualization and result saving

---

## 1. Projection Data Loading (`load_multiple_projection_files`)

### Purpose
Loads multiple text files containing projection data into a 3D numpy array.

### Key Components:
```python
def load_multiple_projection_files(file_pattern="./Phantom Dataset/Al phantom/1.txt", num_files=360):
    # File handling
    file_list = sorted(glob.glob(file_pattern), key=extract_number)  # Numeric sorting
    
    # Dimension detection
    first_file_data = np.loadtxt(file_list[0])  # Gets array shape
    
    # Initialize 3D array (slices × angles × detector_width)
    projections = np.zeros((num_rows, num_angles, num_cols), dtype=np.float32)
    
    # Progressive loading with error handling
    for i, filename in enumerate(file_list):
        # Handles dimension mismatches
        if data.shape != (num_rows, num_cols):
            # Padding/cropping logic

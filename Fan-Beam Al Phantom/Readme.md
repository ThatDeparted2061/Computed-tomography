# CT Reconstruction Pipeline Documentation

## Overview
This pipeline performs 3D fan-beam CT reconstruction from 2D projection data. It handles:
- Projection loading and preprocessing
- Sinogram alignment and normalization
- GPU-accelerated reconstruction
- Visualization and result saving

---

## Code Structure
### 1. Data Loading (load_multiple_projection_files)
python
- Loads series of projection files
- Handles different file formats and dimensions
- Returns 3D numpy array (slices × angles × detector_width)
Example Usage:
```python
projections = load_multiple_projection_files("./data/scan_*.txt")
````
### 2. Preprocessing (preprocess_projection_data)

- 1. Corrects detector misalignment
- 2. Normalizes intensity values
- 3. Reduces noise
- Returns ready-to-reconstruct sinograms
### 3. Reconstruction (reconstruct_3d_fan_beam)

- Configures fan-beam geometry
- Uses ASTRA toolbox for reconstruction
- Auto-selects GPU (FBP) or CPU (SIRT) algorithm
- Returns 3D volume as a numpy array
### 4. Visualisation (visualize_3d_results)

 **Shows**:
- Three axial slices
- Orthogonal cross-sections
- Maximum intensity projection

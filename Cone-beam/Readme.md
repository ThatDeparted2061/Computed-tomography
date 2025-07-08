# Cone Beam CT Reconstruction with ASTRA Toolbox

This project implements a complete pipeline for 3D cone beam computed tomography (CBCT) reconstruction from 2D projection images using the ASTRA toolbox.

## Project Roadmap

1. **Data Loading**
   - Load multiple projection files from a directory
   - Sort files numerically based on filename
   - Combine into a 3D numpy array (detector_rows × angles × detector_columns)

2. **Preprocessing**
   - Sinogram straightening (center-of-mass alignment)
   - Intensity normalization
   - Noise reduction with Gaussian filtering

3. **Reconstruction**
   - Set up cone beam geometry
   - Perform FDK reconstruction (GPU-accelerated)
   - Fallback to SIRT reconstruction (CPU) if GPU unavailable

4. **Visualization**
   - Display projection images
   - Show orthogonal slices (axial, sagittal, coronal)
   - Plot intensity profiles
   - Display statistics

5. **Output**
   - Save reconstructed volume as numpy array
   - Export slices as PNG images
   - Save reconstruction metadata

## Algorithm Details

### 1. Sinogram Straightening Algorithm
For each angle:
For each detector row:
1. Calculate weights as (air_value - projection)
2. Compute weighted center of mass
3. Store center position

Apply Gaussian smoothing to center positions
Calculate reference center (median of all centers)

For each angle:
For each detector row:
1. Calculate required shift (reference - current center)
2. Apply sub-pixel shift with linear interpolation

text

### 2. Cone Beam Reconstruction
**Setup geometry**:
- Detector pixel size
- Source-to-detector distance
- Source-to-object distance
- Rotation angles
- Create ASTRA data structures:
- Projection data (3D)
- Volume data (3D)
- Run reconstruction:
- GPU: FDK algorithm (Filtered Backprojection)
- CPU: SIRT algorithm (100 iterations)
- Retrieve reconstructed volume

### 3. Visualization Pipeline
- Normalize reconstruction to [0,1] range
**Display**:

- Sample projection image

- Orthogonal slices (axial, sagittal, coronal)

- Intensity profiles

- Maximum intensity projection

- Reconstruction statistics

import numpy as np
import astra
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
from scipy import ndimage
from scipy.optimize import minimize_scalar
import time
import re

def load_multiple_projection_files(file_pattern="./Phantom Dataset/Al phantom/1.txt", num_files=360):
    try:
        # Find all matching files
        def extract_number(filename):
            match = re.search(r'(\d+)', os.path.basename(filename))
            return int(match.group(1)) if match else -1

        file_list = sorted(glob.glob(file_pattern), key=extract_number)
        
        if len(file_list) == 0:
            raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
        
        print(f"Found {len(file_list)} files")
        
        # Load first file to get dimensions
        first_file_data = np.loadtxt(file_list[0])
        if first_file_data.ndim == 1:
            first_file_data = first_file_data.reshape(1, -1)
        
        num_rows, num_cols = first_file_data.shape
        num_angles = len(file_list)
        
        # Initialize projection array: (height, angles, detector_width)
        projections = np.zeros((num_rows, num_angles, num_cols), dtype=np.float32)
        
        # Load all files with progress bar
        print("Loading projection files:")
        for i, filename in enumerate(file_list):
            if i % 10 == 0 or i == len(file_list)-1:
                print(f"\rProgress: {i+1}/{len(file_list)} files loaded", end="", flush=True)
            
            data = np.loadtxt(filename)
            
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            # Check dimensions match
            if data.shape != (num_rows, num_cols):
                print(f"\nWarning: File {filename} has different dimensions: {data.shape}")
                # Try to handle mismatched dimensions by resizing or padding
                if data.shape[1] == num_cols:
                    # Only height differs - take first num_rows or pad with zeros
                    if data.shape[0] > num_rows:
                        data = data[:num_rows, :]
                    else:
                        padded = np.zeros((num_rows, num_cols))
                        padded[:data.shape[0], :] = data
                        data = padded
                else:
                    print("Skipping file due to incompatible dimensions")
                    continue
                    
            projections[:, i, :] = data
        
        print(f"\nFinal projection data shape: {projections.shape}")
        print(f"Data range: min={projections.min():.2f}, max={projections.max():.2f}, mean={projections.mean():.2f}")
        
        return projections
    
    except Exception as e:
        print(f"\nError loading projection files: {e}")
        import traceback
        traceback.print_exc()
        return None

def straighten_sinogram_custom(projections, air_value=65535):
    """
    Enhanced sinogram straightening algorithm with optimizations.
    """
    print("Applying enhanced sinogram straightening algorithm...")
    
    height, num_angles, detector_width = projections.shape
    aligned_projections = np.zeros_like(projections)
    
    # Store centers for each angle and slice
    centers = np.zeros((height, num_angles))
    
    print("Finding attenuation centers for each projection...")
    
    # Precompute weights and indices for efficiency
    indices = np.arange(detector_width)
    
    for angle in range(num_angles):
        if angle % 20 == 0 or angle == num_angles-1:
            print(f"\rAnalyzing angle {angle+1}/{num_angles}", end="", flush=True)
            
        for slice_idx in range(height):
            projection = projections[slice_idx, angle, :]
            weights = air_value - projection
            weights = np.maximum(weights, 0)
            
            if np.sum(weights) > 0:
                centers[slice_idx, angle] = np.sum(indices * weights) / np.sum(weights)
            else:
                centers[slice_idx, angle] = detector_width / 2.0
    
    print("\nApplying smoothing to center positions...")
    
    # Apply smoothing to center positions across angles
    centers = ndimage.gaussian_filter(centers, sigma=(0, 3))  # Only smooth along angle axis
    
    # Detrend using vectorized operations
    x = np.arange(num_angles)
    for slice_idx in range(height):
        y = centers[slice_idx, :]
        coeffs = np.polyfit(x, y, deg=1)
        trend = np.polyval(coeffs, x)
        centers[slice_idx, :] -= trend - np.median(centers[slice_idx, :])
    
    reference_center = np.median(centers)
    print(f"Reference center position: {reference_center:.2f} pixels")
    
    print("Aligning projections to reference center with sub-pixel precision...")
    
    # Vectorized sub-pixel shifting
    for angle in range(num_angles):
        if angle % 20 == 0 or angle == num_angles-1:
            print(f"\rAligning angle {angle+1}/{num_angles}", end="", flush=True)
            
        shifts = reference_center - centers[:, angle]
        for slice_idx in range(height):
            aligned_projections[slice_idx, angle, :] = ndimage.shift(
                projections[slice_idx, angle, :],
                shift=shifts[slice_idx],
                mode='constant',
                cval=air_value,
                order=1
            )
    
    print("\nEnhanced sinogram straightening complete!")
    return aligned_projections

def preprocess_projection_data(projections, apply_alignment=True):
    """Optimized preprocessing pipeline."""
    print("\nPreprocessing projection data...")
    
    projections = projections.astype(np.float32)
    
    # Apply custom sinogram straightening
    if apply_alignment:
        projections = straighten_sinogram_custom(projections, air_value=65535)
    
    # Normalize data
    if projections.max() > 60000:  # Likely 16-bit data
        projections = projections / 65535.0
    
    # Apply Gaussian filtering using vectorized operations
    projections = ndimage.gaussian_filter(projections, sigma=(0, 0, 0.8))
    
    # Enhanced normalization - per slice
    min_vals = projections.min(axis=(1, 2), keepdims=True)
    max_vals = projections.max(axis=(1, 2), keepdims=True)
    projections = (projections - min_vals) / (max_vals - min_vals + 1e-6)
    
    return projections

def reconstruct_3d_fan_beam(projections, angles=None, source_distance=500, detector_distance=100):
    """
    Corrected 3D fan beam reconstruction with proper fan beam algorithms.
    """
    height, num_angles, detector_width = projections.shape
    
    if angles is None:
        angles = np.linspace(0, 2*np.pi, num_angles, False)
    
    print(f"\nStarting 3D fan beam reconstruction...")
    print(f"Volume dimensions: {detector_width}x{detector_width}x{height}")
    
    # Check for GPU availability
    use_gpu = astra.astra.use_cuda()
    print(f"Using {'GPU' if use_gpu else 'CPU'} acceleration")
    
    # Create geometry
    vol_geom = astra.create_vol_geom(detector_width, detector_width)
    detector_pixel_size = 1.0
    
    # Create fan beam geometry
    proj_geom = astra.create_proj_geom(
        'fanflat', 
        detector_pixel_size, 
        detector_width, 
        angles, 
        source_distance, 
        detector_distance
    )
    
    # Initialize reconstruction volume
    reconstruction = np.zeros((height, detector_width, detector_width), dtype=np.float32)
    
    print(f"Reconstructing {height} slices:")
    
    # Create projector
    projector_id = astra.create_projector('cuda' if use_gpu else 'line_fanflat', proj_geom, vol_geom)
    
    # Use appropriate algorithm based on availability
    if use_gpu:
        # Use GPU-accelerated fan beam FBP
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ProjectionDataId'] = None  # Will be set per slice
        cfg['ReconstructionDataId'] = None  # Will be set per slice
        cfg['ProjectorId'] = projector_id
        cfg['FilterType'] = 'hann'
    else:
        # Use CPU-based fan beam SIRT (FBP not available for fan beam on CPU)
        cfg = astra.astra_dict('SIRT')
        cfg['ProjectionDataId'] = None
        cfg['ReconstructionDataId'] = None
        cfg['ProjectorId'] = projector_id
        cfg['option'] = {'MinConstraint': 0.0, 'MaxConstraint': 1.0}
    
    for slice_idx in range(height):
        if slice_idx % 5 == 0 or slice_idx == height-1:
            print(f"\rProgress: {slice_idx+1}/{height} slices", end="", flush=True)
        
        sinogram = projections[slice_idx, :, :]
        
        # Create data structures
        sino_id = astra.data2d.create('-sino', proj_geom, sinogram)
        rec_id = astra.data2d.create('-vol', vol_geom)
        
        # Configure algorithm
        cfg['ProjectionDataId'] = sino_id
        cfg['ReconstructionDataId'] = rec_id
        
        # Create and run algorithm
        alg_id = astra.algorithm.create(cfg)
        
        if use_gpu:
            # FBP only needs 1 iteration
            astra.algorithm.run(alg_id, 1)
        else:
            # SIRT needs more iterations
            astra.algorithm.run(alg_id, 50)  # Reduced from 100 for speed
        
        reconstruction[slice_idx] = astra.data2d.get(rec_id)
        
        # Clean up
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(sino_id)
        astra.data2d.delete(rec_id)
    
    # Clean up projector
    astra.projector.delete(projector_id)
    
    print("\nReconstruction complete!")
    return reconstruction

def visualize_3d_results(projections, reconstruction):
    """Optimized visualization."""
    height = reconstruction.shape[0]
    
    plt.figure(figsize=(15, 10))
    plt.suptitle("3D Reconstruction Results", fontsize=14)
    
    # Show middle slices
    slices_to_show = [height//4, height//2, 3*height//4]
    for i, pos in enumerate(slices_to_show):
        plt.subplot(2, 3, i+1)
        plt.imshow(reconstruction[pos], cmap='gray')
        plt.title(f'Slice {pos}')
        plt.axis('off')
    
    # Show orthogonal views
    plt.subplot(2, 3, 4)
    plt.imshow(reconstruction[:, height//2, :], cmap='gray', aspect='auto')
    plt.title('Coronal View')
    
    plt.subplot(2, 3, 5)
    plt.imshow(reconstruction[:, :, height//2], cmap='gray', aspect='auto')
    plt.title('Sagittal View')
    
    plt.subplot(2, 3, 6)
    plt.imshow(np.max(reconstruction, axis=0), cmap='hot')
    plt.title('Maximum Intensity Projection')
    
    plt.tight_layout()
    plt.show()

def save_reconstruction(reconstruction, output_dir='reconstruction_output'):
    """Optimized saving function."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as compressed numpy file
    np.savez_compressed(os.path.join(output_dir, 'reconstruction.npz'), data=reconstruction)
    
    # Save middle slice as PNG
    mid_slice = reconstruction[reconstruction.shape[0]//2]
    plt.imsave(os.path.join(output_dir, 'middle_slice.png'), mid_slice, cmap='gray')
    
    print(f"Results saved to {output_dir}")

def main():
    """Optimized main workflow."""
    print("=" * 60)
    print("3D Fan Beam CT Reconstruction Pipeline")
    print("=" * 60)
    
    params = {
        'file_pattern': "./Phantom Dataset/Al phantom/*.txt",
        'output_dir': 'reconstruction_results',
        'source_distance': 500,
        'detector_distance': 100
    }
    
    try:
        start_time = time.time()
        
        # Load and preprocess
        print("\nLoading projection data...")
        projections = load_multiple_projection_files(params['file_pattern'])
        if projections is None:
            return
            
        print("\nPreprocessing data...")
        projections = preprocess_projection_data(projections)
        
        # Reconstruct
        print("\nStarting reconstruction...")
        reconstruction = reconstruct_3d_fan_beam(
            projections,
            source_distance=params['source_distance'],
            detector_distance=params['detector_distance']
        )
        
        # Visualize and save
        print("\nVisualizing results...")
        visualize_3d_results(projections, reconstruction)
        
        print("\nSaving results...")
        save_reconstruction(reconstruction, params['output_dir'])
        
        duration = time.time() - start_time
        print(f"\nProcessing completed in {duration:.2f} seconds")
        print(f"Final volume shape: {reconstruction.shape}")
        
        return reconstruction
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    volume = main()

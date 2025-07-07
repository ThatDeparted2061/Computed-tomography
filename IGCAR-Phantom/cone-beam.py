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
    Enhanced sinogram straightening algorithm:
    - Uses weighted center-of-mass for more robust center detection
    - Applies smoothing to center positions across angles
    - Uses sub-pixel shifting for better alignment
    """
    print("Applying enhanced sinogram straightening algorithm...")
    
    height, num_angles, detector_width = projections.shape
    aligned_projections = np.zeros_like(projections)
    
    # Store centers for each angle and slice
    centers = np.zeros((height, num_angles))
    
    print("Finding attenuation centers for each projection...")
    
    for angle in range(num_angles):
        if angle % 20 == 0 or angle == num_angles-1:
            print(f"\rAnalyzing angle {angle+1}/{num_angles}", end="", flush=True)
            
        for slice_idx in range(height):
            projection = projections[slice_idx, angle, :]
            
            # Calculate weights as inverse of intensity (more weight to darker pixels)
            weights = air_value - projection
            weights = np.maximum(weights, 0)  # Ensure non-negative
            
            # Calculate weighted center of mass
            if np.sum(weights) > 0:
                indices = np.arange(detector_width)
                center = np.sum(indices * weights) / np.sum(weights)
            else:
                center = detector_width / 2.0
                
            centers[slice_idx, angle] = center
    
    print("\nApplying smoothing to center positions...")
    
    # Apply smoothing to center positions across angles
    for slice_idx in range(height):
        centers[slice_idx, :] = ndimage.gaussian_filter1d(centers[slice_idx, :], sigma=3)
    
    # Calculate the reference center (median of all centers for stability)
    reference_center = np.median(centers)
    print(f"Reference center position: {reference_center:.2f} pixels")
    
    # Show center statistics
    print(f"Center range: {centers.min():.2f} to {centers.max():.2f} pixels")
    print(f"Center std deviation: {centers.std():.2f} pixels")
    
    print("Aligning projections to reference center with sub-pixel precision...")
    
    # Apply alignment shifts with sub-pixel precision
    for angle in range(num_angles):
        if angle % 20 == 0 or angle == num_angles-1:
            print(f"\rAligning angle {angle+1}/{num_angles}", end="", flush=True)
            
        for slice_idx in range(height):
            projection = projections[slice_idx, angle, :]
            current_center = centers[slice_idx, angle]
            
            # Calculate sub-pixel shift needed
            shift = reference_center - current_center
            
            # Apply the shift with sub-pixel precision
            # Note: For 1D array, shift should be a single value, not a tuple
            aligned_projections[slice_idx, angle, :] = ndimage.shift(
                projection, 
                shift=shift,  # Single value for 1D array
                mode='constant',
                cval=air_value,
                order=1  # Linear interpolation for sub-pixel shifts
            )
    
    print("\nEnhanced sinogram straightening complete!")
    
    # Show alignment improvement
    center_std_before = centers.std()
    
    # Recalculate centers after alignment to verify
    centers_after = np.zeros_like(centers)
    for angle in range(num_angles):
        for slice_idx in range(height):
            projection = aligned_projections[slice_idx, angle, :]
            weights = air_value - projection
            weights = np.maximum(weights, 0)
            
            if np.sum(weights) > 0:
                indices = np.arange(detector_width)
                centers_after[slice_idx, angle] = np.sum(indices * weights) / np.sum(weights)
            else:
                centers_after[slice_idx, angle] = detector_width / 2.0
    
    center_std_after = centers_after.std()
    print(f"Center alignment improvement:")
    print(f"  Before: std = {center_std_before:.3f} pixels")
    print(f"  After:  std = {center_std_after:.3f} pixels")
    print(f"  Improvement: {((center_std_before - center_std_after) / center_std_before * 100):.1f}%")
    
    return aligned_projections

def preprocess_projection_data(projections, apply_alignment=True):
    """Enhanced preprocessing with custom sinogram straightening."""
    print("\nPreprocessing projection data with custom straightening...")
    
    projections = projections.astype(np.float32)
    
    # Apply custom sinogram straightening
    if apply_alignment:
        projections = straighten_sinogram_custom(projections, air_value=65535)
    
    # Normalize data after alignment
    data_max = projections.max()
    if data_max > 60000:  # Likely 16-bit data
        print("Normalizing 16-bit data...")
        projections = projections / 65535.0
    
    # Apply mild Gaussian filtering to reduce noise
    print("Applying noise reduction...")
    for i in range(projections.shape[0]):
        projections[i] = ndimage.gaussian_filter(projections[i], sigma=0.8)
    
    # Enhanced normalization - per slice to account for varying intensity
    print("Applying enhanced normalization...")
    for i in range(projections.shape[0]):
        slice_data = projections[i]
        if slice_data.max() > slice_data.min():
            projections[i] = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
    
    return projections

def setup_cone_geometry(projections, angles=None, source_to_detector=1000, source_to_object=500):
    """
    Setup cone beam geometry for ASTRA.
    
    Parameters:
    - projections: 3D numpy array (detector_rows, num_angles, detector_cols)
    - angles: array of projection angles in radians (if None, creates equally spaced angles)
    - source_to_detector: distance from source to detector (in pixels)
    - source_to_object: distance from source to object (in pixels)
    
    Returns:
    - proj_geom: ASTRA projection geometry
    - vol_geom: ASTRA volume geometry
    """
    detector_rows, num_angles, detector_cols = projections.shape
    
    if angles is None:
        angles = np.linspace(0, 2*np.pi, num_angles, False)
    
    # Calculate magnification and detector pixel size
    magnification = source_to_detector / source_to_object
    det_pixel_size = 1.0  # Assuming unit pixel size
    
    # Create cone beam geometry
    proj_geom = astra.create_proj_geom(
        'cone', 
        det_pixel_size, det_pixel_size,  # Detector pixel size (width, height)
        detector_rows, detector_cols,    # Detector size (rows, columns)
        angles,                          # Projection angles
        source_to_object,                # Distance source to origin
        source_to_detector - source_to_object  # Distance origin to detector
    )
    
    # Create volume geometry (cube with same width as detector columns)
    vol_geom = astra.create_vol_geom(
        detector_cols, detector_cols, detector_cols  # Cubic volume
    )
    
    return proj_geom, vol_geom

def reconstruct_3d_cone_beam(projections, angles=None, use_gpu=True):
    """
    Reconstruct 3D volume from cone beam projection data using ASTRA.
    
    Parameters:
    - projections: 3D numpy array (detector_rows, num_angles, detector_cols)
    - angles: array of projection angles in radians (if None, creates equally spaced angles)
    - use_gpu: whether to use GPU acceleration (recommended)
    
    Returns:
    - reconstruction: 3D numpy array
    """
    detector_rows, num_angles, detector_cols = projections.shape
    
    if angles is None:
        angles = np.linspace(0, 2*np.pi, num_angles, False)
    
    print(f"\nStarting 3D cone beam reconstruction with {num_angles} angles...")
    print(f"Projection data shape: {projections.shape}")
    print(f"Angle range: {np.degrees(angles[0]):.1f}° to {np.degrees(angles[-1]):.1f}°")
    
    # Setup cone beam geometry
    proj_geom, vol_geom = setup_cone_geometry(projections, angles)
    
    # Create projection data in ASTRA
    proj_id = astra.data3d.create('-proj3d', proj_geom, projections)
    
    # Create reconstruction volume
    rec_id = astra.data3d.create('-vol', vol_geom)
    
    # Configure reconstruction algorithm
    if use_gpu:
        # Use GPU accelerated FDK algorithm (cone beam FBP)
        cfg = astra.astra_dict('FDK_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        alg_id = astra.algorithm.create(cfg)
        
        print("Running FDK reconstruction on GPU...")
        astra.algorithm.run(alg_id)
        
        # Get reconstruction result
        reconstruction = astra.data3d.get(rec_id)
        
        # Clean up
        astra.algorithm.delete(alg_id)
    else:
        # Use CPU version (slower)
        print("Warning: CPU cone beam reconstruction is much slower than GPU")
        
        # Create projector
        projector_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
        
        # Use SIRT algorithm (more stable on CPU for cone beam)
        cfg = astra.astra_dict('SIRT3D_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        cfg['ProjectorId'] = projector_id
        alg_id = astra.algorithm.create(cfg)
        
        print("Running SIRT reconstruction on CPU (100 iterations)...")
        astra.algorithm.run(alg_id, 100)
        
        # Get reconstruction result
        reconstruction = astra.data3d.get(rec_id)
        
        # Clean up
        astra.algorithm.delete(alg_id)
        astra.projector.delete(projector_id)
    
    # Clean up remaining data
    astra.data3d.delete(proj_id)
    astra.data3d.delete(rec_id)
    
    return reconstruction

def visualize_3d_results(projections, reconstruction):
    """Enhanced visualization showing before/after correction."""
    detector_rows, num_angles, detector_cols = projections.shape
    
    # Normalize reconstruction for display
    reconstruction = (reconstruction - reconstruction.min()) / (reconstruction.max() - reconstruction.min())
    
    plt.figure(figsize=(20, 16))
    plt.suptitle("3D Cone Beam Reconstruction Results", fontsize=16, y=0.98)
    
    # Sample projection
    plt.subplot(3, 4, 1)
    mid_angle = num_angles // 2
    plt.imshow(projections[:, mid_angle, :], cmap='gray', aspect='auto')
    plt.title(f'Projection at Angle {mid_angle}')
    plt.xlabel('Detector Position')
    plt.ylabel('Detector Row')
    plt.colorbar(label='Intensity')
    
    # Reconstructed slices at different heights
    slice_positions = [detector_cols//4, detector_cols//2, 3*detector_cols//4]
    for i, pos in enumerate(slice_positions):
        plt.subplot(3, 4, i+2)
        plt.imshow(reconstruction[pos, :, :], cmap='gray', vmin=0, vmax=1)
        plt.title(f'Axial Slice Z={pos}')
        plt.axis('off')
        plt.colorbar(label='Density')
    
    # Cross-sectional views
    plt.subplot(3, 4, 5)
    sagittal = reconstruction[:, detector_cols//2, :]
    plt.imshow(sagittal, cmap='gray', aspect='auto', 
               extent=[0, detector_cols, detector_cols, 0])
    plt.title('Sagittal View (X-Z)')
    plt.xlabel('X Position')
    plt.ylabel('Z Position')
    plt.colorbar(label='Density')
    
    plt.subplot(3, 4, 6)  
    coronal = reconstruction[:, :, detector_cols//2]
    plt.imshow(coronal, cmap='gray', aspect='auto',
               extent=[0, detector_cols, detector_cols, 0])
    plt.title('Coronal View (Y-Z)')
    plt.xlabel('Y Position')
    plt.ylabel('Z Position')
    plt.colorbar(label='Density')
    
    # Orthogonal views
    plt.subplot(3, 4, 7)
    plt.imshow(reconstruction[detector_cols//2, :, :], cmap='gray')
    plt.title('Axial View (Central Slice)')
    plt.axis('off')
    plt.colorbar(label='Density')
    
    plt.subplot(3, 4, 8)
    plt.imshow(reconstruction[:, detector_cols//2, :], cmap='gray')
    plt.title('Sagittal View (Central Slice)')
    plt.axis('off')
    plt.colorbar(label='Density')
    
    # Profile plots
    plt.subplot(3, 4, 9)
    center_slice = reconstruction[detector_cols//2, :, :]
    center_profile = center_slice[detector_cols//2, :]
    plt.plot(center_profile)
    plt.title('Horizontal Profile (Center)')
    plt.xlabel('X Position')
    plt.ylabel('Density')
    plt.grid(True)
    
    plt.subplot(3, 4, 10)
    vertical_profile = center_slice[:, detector_cols//2]
    plt.plot(vertical_profile)
    plt.title('Vertical Profile (Center)')
    plt.xlabel('Y Position')
    plt.ylabel('Density')
    plt.grid(True)
    
    # 3D visualization preview
    plt.subplot(3, 4, 11)
    # Create a maximum intensity projection
    mip = np.max(reconstruction, axis=0)
    plt.imshow(mip, cmap='hot')
    plt.title('Maximum Intensity Projection')
    plt.axis('off')
    plt.colorbar(label='Max Density')
    
    # Statistics
    plt.subplot(3, 4, 12)
    plt.text(0.1, 0.9, f'Volume Shape: {reconstruction.shape}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.8, f'Data Range: {reconstruction.min():.4f} - {reconstruction.max():.4f}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.7, f'Mean: {reconstruction.mean():.4f}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f'Std: {reconstruction.std():.4f}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.5, f'Non-zero voxels: {np.count_nonzero(reconstruction)}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f'Total voxels: {reconstruction.size}', transform=plt.gca().transAxes)
    plt.title('Reconstruction Statistics')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def save_reconstruction(reconstruction, output_dir='reconstruction_output'):
    """Enhanced saving with metadata and multiple formats."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full volume
    np.save(os.path.join(output_dir, 'reconstruction_3d.npy'), reconstruction)
    
    # Save metadata
    with open(os.path.join(output_dir, 'metadata.txt'), 'w') as f:
        f.write(f"Reconstruction dimensions: {reconstruction.shape}\n")
        f.write(f"Data range: min={reconstruction.min():.6f}, max={reconstruction.max():.6f}\n")
        f.write(f"Mean: {reconstruction.mean():.6f}, Std: {reconstruction.std():.6f}\n")
        f.write(f"Data type: {reconstruction.dtype}\n")
        f.write(f"Non-zero voxels: {np.count_nonzero(reconstruction)}\n")
        f.write(f"Total voxels: {reconstruction.size}\n")
    
    # Save individual slices
    slice_dir = os.path.join(output_dir, 'slices')
    os.makedirs(slice_dir, exist_ok=True)
    print(f"\nSaving slices to {slice_dir}:")
    
    for i in range(reconstruction.shape[0]):
        if i % 10 == 0 or i == reconstruction.shape[0]-1:
            print(f"\rProgress: {i+1}/{reconstruction.shape[0]} slices", end="", flush=True)
        
        slice_img = reconstruction[i, :, :]
        # Normalize each slice individually for better contrast
        if slice_img.max() > slice_img.min():
            slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min())
        slice_img = (255 * slice_img).astype(np.uint8)
        
        plt.imsave(os.path.join(slice_dir, f'slice_{i:03d}.png'), slice_img, cmap='gray')
    
    print("\nSaving complete.")

def main():
    """Main workflow for cone beam reconstruction."""
    print("=" * 70)
    print("3D Cone Beam CT Reconstruction with ASTRA Toolbox")
    print("=" * 70)
    
    params = {
        'file_pattern': "./Phantom Dataset/Al phantom/*.txt",
        'expected_projections': 360,
        'output_dir': 'cone_beam_reconstruction_results',
        'use_gpu': True,
        'source_to_detector': 1000,  # Distance in pixels
        'source_to_object': 500      # Distance in pixels
    }
    
    try:
        # Load projection data
        projections = load_multiple_projection_files(
            file_pattern=params['file_pattern'],
            num_files=params['expected_projections'])
        
        if projections is None:
            return None
        
        # Preprocess with custom sinogram straightening
        projections = preprocess_projection_data(projections, apply_alignment=True)
        
        # Reconstruct using cone beam geometry
        reconstruction = reconstruct_3d_cone_beam(
            projections,
            use_gpu=params['use_gpu']
        )
        
        # Visualize results
        visualize_3d_results(projections, reconstruction)
        
        # Save results
        save_reconstruction(reconstruction, params['output_dir'])
        
        print("\nCone beam reconstruction completed successfully!")
        print(f"Final volume shape: {reconstruction.shape}")
        print(f"Data range: {reconstruction.min():.6f} to {reconstruction.max():.6f}")
        
        return reconstruction
        
    except Exception as e:
        print(f"\nError during reconstruction: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    start_time = time.time()
    volume = main()
    duration = time.time() - start_time
    print(f"\nTotal execution time: {duration:.2f} seconds")

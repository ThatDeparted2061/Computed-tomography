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

def load_multiple_projection_files(file_pattern="./Phantom Dataset/Al phantom/*.txt", num_files=399):
    try:
        # Find all matching files
        def extract_number(filename):
            match = re.search(r'(\d+)', os.path.basename(filename))
            return int(match.group(1)) if match else -1

        file_list = sorted(glob.glob(file_pattern), key=extract_number)
        
        # Filter files to only include 2.txt to 400.txt
        file_list = [f for f in file_list if 2 <= extract_number(f) <= 400]
        
        if len(file_list) == 0:
            raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
        
        print(f"Found {len(file_list)} files (from 2.txt to 400.txt)")
        
        # First pass: check all file dimensions
        print("Verifying file dimensions...")
        dimensions = []
        for i, filename in enumerate(file_list):
            try:
                data = np.loadtxt(filename)
                dimensions.append(data.shape)
                if i % 50 == 0 or i == len(file_list)-1:
                    print(f"\rProgress: {i+1}/{len(file_list)} files checked", end="", flush=True)
            except Exception as e:
                print(f"\nError loading file {filename}: {e}")
                continue
        
        # Find most common dimensions
        unique_dims, counts = np.unique(dimensions, axis=0, return_counts=True)
        most_common_dim = unique_dims[np.argmax(counts)]
        print(f"\nMost common dimensions: {most_common_dim}")
        
        # Check if we should proceed with these dimensions
        if len(unique_dims) > 1:
            print("Warning: Multiple dimensions found in files:")
            for dim, count in zip(unique_dims, counts):
                print(f"  {dim}: {count} files")
            
            # Ask user whether to proceed with most common dimensions
            response = input(f"Proceed with dimensions {most_common_dim}? (y/n): ")
            if response.lower() != 'y':
                return None
        
        # Set expected dimensions based on most common
        expected_rows, expected_cols = most_common_dim
        print(f"Using dimensions: {expected_rows} rows x {expected_cols} columns")
        
        # Initialize projection array
        projections = np.zeros((expected_rows, len(file_list), expected_cols), dtype=np.float32)
        
        # Load all files with progress bar
        print("\nLoading projection files:")
        for i, filename in enumerate(file_list):
            if i % 10 == 0 or i == len(file_list)-1:
                print(f"\rProgress: {i+1}/{len(file_list)} files loaded", end="", flush=True)
            
            try:
                data = np.loadtxt(filename)
                
                # Handle dimension mismatches
                if data.shape != (expected_rows, expected_cols):
                    if data.size == expected_rows * expected_cols:
                        # Reshape if total elements match
                        data = data.reshape(expected_rows, expected_cols)
                    else:
                        # Pad or crop if needed
                        new_data = np.zeros((expected_rows, expected_cols))
                        rows = min(expected_rows, data.shape[0])
                        cols = min(expected_cols, data.shape[1])
                        new_data[:rows, :cols] = data[:rows, :cols]
                        data = new_data
                
                projections[:, i, :] = data
            except Exception as e:
                print(f"\nError processing file {filename}: {e}")
                continue
        
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
    Enhanced sinogram straightening algorithm for 399 projection files:
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

    # Detrend: fit a linear regression to each slice's centers and remove the trend
    for slice_idx in range(height):
        x = np.arange(num_angles)
        y = centers[slice_idx, :]
        # Fit linear model: y = mx + b
        coeffs = np.polyfit(x, y, deg=1)
        trend = np.polyval(coeffs, x)
        centers[slice_idx, :] -= trend - np.median(centers[slice_idx, :])  # align to median

    
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
            aligned_projections[slice_idx, angle, :] = ndimage.shift(
                projection, 
                shift=shift,
                mode='constant',
                cval=air_value,
                order=1  # Linear interpolation for sub-pixel shifts
            )
    
    print("\nEnhanced sinogram straightening complete!")
    
    # Show alignment improvement
    center_std_before = centers.std()
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

def reconstruct_3d_parallel_beam(projections, angles=None):
    """
    Reconstruct 3D volume from projection data with proper dimension handling.
    CPU-only: no GPU code.
    """
    height, num_angles, detector_width = projections.shape
    
    if angles is None:
        angles = np.linspace(0, np.pi, num_angles, False)  # Full 180 degrees
    
    print(f"\nStarting 3D reconstruction with {num_angles} angles...")
    print(f"Volume dimensions: {detector_width}x{detector_width}x{height}")
    print(f"Projection data shape: {projections.shape}")
    print(f"Angle range: {np.degrees(angles[0]):.1f}° to {np.degrees(angles[-1]):.1f}°")
    
    # Use CPU slice-by-slice reconstruction
    return reconstruct_slice_by_slice(projections, angles)

def reconstruct_slice_by_slice(projections, angles):
    """Enhanced slice-by-slice reconstruction with Shepp-Logan filter."""
    height, num_angles, detector_width = projections.shape
    
    reconstruction = np.zeros((height, detector_width, detector_width), dtype=np.float32)
    
    # Create 2D geometry with proper scaling
    vol_geom = astra.create_vol_geom(detector_width, detector_width)
    proj_geom = astra.create_proj_geom('parallel', 1.5, detector_width, angles)
    
    print(f"Reconstructing {height} slices:")
    print(f"Expected sinogram shape per slice: ({num_angles}, {detector_width})")
    
    # Try different CPU-only reconstruction methods in order of preference
    methods_to_try = [
        ('FBP', 'line'),           # CPU FBP with line projector
        ('FBP', 'linear'),         # CPU FBP with linear projector (alternative)
        ('SIRT', 'line'),          # CPU SIRT with projector
        ('ART', 'line')            # CPU ART with projector (fallback)
    ]
    
    successful_method = None
    projector_id = None
    
    # Test which method works
    for method_name, projector_type in methods_to_try:
        try:
            print(f"\nTrying reconstruction method: {method_name}")
            
            # Create test data for first slice
            test_sinogram = projections[0, :, :]
            proj_id = astra.data2d.create('-sino', proj_geom, test_sinogram)
            rec_id = astra.data2d.create('-vol', vol_geom)
            
            if 'CUDA' in method_name:
                cfg = astra.astra_dict(method_name)
                cfg['ReconstructionDataId'] = rec_id
                cfg['ProjectionDataId'] = proj_id
                cfg['FilterType'] = 'Ram-Lak'
            else:
                if projector_id is None:
                    projector_id = astra.create_projector(projector_type, proj_geom, vol_geom)
                
                cfg = astra.astra_dict(method_name)
                cfg['ReconstructionDataId'] = rec_id
                cfg['ProjectionDataId'] = proj_id
                cfg['ProjectorId'] = projector_id
                cfg['FilterType'] = 'Ram-Lak'
            
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id, 1)
            
            successful_method = (method_name, projector_type)
            print(f"Successfully using method: {method_name}")
            
            astra.algorithm.delete(alg_id)
            astra.data2d.delete(rec_id)
            astra.data2d.delete(proj_id)
            break
            
        except Exception as e:
            print(f"Method {method_name} failed: {str(e)}")
            try:
                astra.algorithm.delete(alg_id)
            except:
                pass
            try:
                astra.data2d.delete(rec_id)
            except:
                pass
            try:
                astra.data2d.delete(proj_id)
            except:
                pass
            continue
    
    if successful_method is None:
        raise RuntimeError("No reconstruction method worked. Check ASTRA installation.")
    
    method_name, projector_type = successful_method
    
    # Now reconstruct all slices using the successful method
    for slice_idx in range(height):
        if slice_idx % 5 == 0 or slice_idx == height-1:
            print(f"\rProgress: {slice_idx+1}/{height} slices", end="", flush=True)
        
        sinogram = projections[slice_idx, :, :]
        
        if sinogram.shape != (num_angles, detector_width):
            print(f"\nError: Sinogram dimensions {sinogram.shape} do not match expected ({num_angles}, {detector_width})")
            continue
        
        proj_id = astra.data2d.create('-sino', proj_geom, sinogram)
        rec_id = astra.data2d.create('-vol', vol_geom)
        
        if 'CUDA' in method_name:
            cfg = astra.astra_dict(method_name)
            cfg['ReconstructionDataId'] = rec_id
            cfg['ProjectionDataId'] = proj_id
            cfg['FilterType'] = 'Ram-Lak'
            iterations = 1 if 'FBP' in method_name else 150
        else:
            cfg = astra.astra_dict(method_name)
            cfg['ReconstructionDataId'] = rec_id
            cfg['ProjectionDataId'] = proj_id
            cfg['ProjectorId'] = projector_id
            cfg['FilterType'] = 'Ram-Lak'
            iterations = 1 if method_name == 'FBP' else 100
        
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, iterations)
        
        reconstruction[slice_idx] = astra.data2d.get(rec_id)
        
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(proj_id)
    
    if projector_id is not None:
        astra.projector.delete(projector_id)
    
    print()
    return reconstruction

def visualize_3d_results(projections, reconstruction):
    """Enhanced visualization showing before/after correction."""
    height, num_angles, detector_width = projections.shape
    
    reconstruction = (reconstruction - reconstruction.min()) / (reconstruction.max() - reconstruction.min())
    
    plt.figure(figsize=(20, 16))
    plt.suptitle("3D Reconstruction Results from 399 Projections (81x271)", fontsize=16, y=0.98)
    
    # Sample sinogram (straightened)
    plt.subplot(3, 4, 1)
    mid_slice = height // 2
    sinogram = projections[mid_slice, :, :]
    sinogram = (sinogram - sinogram.min()) / (sinogram.max() - sinogram.min())  # Normalize for visibility
    plt.imshow(sinogram, cmap='gray', aspect='auto')
    plt.title(f'Straightened Sinogram (Slice {mid_slice})')
    plt.xlabel('Detector Position')
    plt.ylabel('Projection Angle Index')
    plt.colorbar(label='Intensity')
    
    # Reconstructed slices at different heights
    slice_positions = [height//4, height//2, 3*height//4]
    for i, pos in enumerate(slice_positions):
        plt.subplot(3, 4, i+2)
        plt.imshow(reconstruction[pos, :, :], cmap='gray', vmin=0, vmax=1)
        plt.title(f'Slice {pos} (Z={pos})')
        plt.axis('off')
        plt.colorbar(label='Density')
    
    # Cross-sectional views
    plt.subplot(3, 4, 5)
    sagittal = reconstruction[:, detector_width//2, :]
    plt.imshow(sagittal, cmap='gray', aspect='auto', 
               extent=[0, detector_width, height, 0])
    plt.title('Sagittal View (X-Z)')
    plt.xlabel('X Position')
    plt.ylabel('Z (Height)')
    plt.colorbar(label='Density')
    
    plt.subplot(3, 4, 6)  
    coronal = reconstruction[:, :, detector_width//2]
    plt.imshow(coronal, cmap='gray', aspect='auto',
               extent=[0, detector_width, height, 0])
    plt.title('Coronal View (Y-Z)')
    plt.xlabel('Y Position')
    plt.ylabel('Z (Height)')
    plt.colorbar(label='Density')
    
    # Axial view at different Z positions
    plt.subplot(3, 4, 7)
    plt.imshow(reconstruction[height//4, :, :], cmap='gray')
    plt.title(f'Axial View (Z={height//4})')
    plt.axis('off')
    plt.colorbar(label='Density')
    
    plt.subplot(3, 4, 8)
    plt.imshow(reconstruction[3*height//4, :, :], cmap='gray')
    plt.title(f'Axial View (Z={3*height//4})')
    plt.axis('off')
    plt.colorbar(label='Density')
    
    # Profile plots
    plt.subplot(3, 4, 9)
    center_slice = reconstruction[height//2, :, :]
    center_profile = center_slice[detector_width//2, :]
    plt.plot(center_profile)
    plt.title('Horizontal Profile (Center)')
    plt.xlabel('X Position')
    plt.ylabel('Density')
    plt.grid(True)
    
    plt.subplot(3, 4, 10)
    vertical_profile = center_slice[:, detector_width//2]
    plt.plot(vertical_profile)
    plt.title('Vertical Profile (Center)')
    plt.xlabel('Y Position')
    plt.ylabel('Density')
    plt.grid(True)
    
    # 3D visualization preview
    plt.subplot(3, 4, 11)
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

def save_reconstruction(reconstruction, output_dir='reconstruction_output_399'):
    """Enhanced saving with metadata and multiple formats."""
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'reconstruction_3d.npy'), reconstruction)
    
    with open(os.path.join(output_dir, 'metadata.txt'), 'w') as f:
        f.write(f"Reconstruction dimensions: {reconstruction.shape}\n")
        f.write(f"Data range: min={reconstruction.min():.6f}, max={reconstruction.max():.6f}\n")
        f.write(f"Mean: {reconstruction.mean():.6f}, Std: {reconstruction.std():.6f}\n")
        f.write(f"Data type: {reconstruction.dtype}\n")
        f.write(f"Non-zero voxels: {np.count_nonzero(reconstruction)}\n")
        f.write(f"Total voxels: {reconstruction.size}\n")
    
    slice_dir = os.path.join(output_dir, 'slices')
    os.makedirs(slice_dir, exist_ok=True)
    print(f"\nSaving slices to {slice_dir}:")
    
    for i in range(reconstruction.shape[0]):
        if i % 10 == 0 or i == reconstruction.shape[0]-1:
            print(f"\rProgress: {i+1}/{reconstruction.shape[0]} slices", end="", flush=True)
        
        slice_img = reconstruction[i, :, :]
        if slice_img.max() > slice_img.min():
            slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min())
        slice_img = (255 * slice_img).astype(np.uint8)
        
        plt.imsave(os.path.join(slice_dir, f'slice_{i:03d}.png'), slice_img, cmap='gray')
    
    print("\nSaving complete.")

def main():
    """Main workflow for 399 projection files (81x271 each)."""
    print("=" * 70)
    print("3D CT Reconstruction from 399 Projections (2.txt to 400.txt)")
    print("=" * 70)
    
    params = {
        'file_pattern': "./Phantom/*.txt",  # Adjusted to match 2.txt to 400.txt
        'expected_projections': 399,
        'output_dir': 'reconstruction_results_399'
    }
    
    try:
        # Load projection data (2.txt to 400.txt)
        projections = load_multiple_projection_files(
            file_pattern=params['file_pattern'],
            num_files=params['expected_projections'])
        
        if projections is None:
            return None
        
        # Preprocess with custom sinogram straightening
        projections = preprocess_projection_data(projections, apply_alignment=True)
        
        # Reconstruct
        reconstruction = reconstruct_3d_parallel_beam(projections)
        
        # Visualize results
        visualize_3d_results(projections, reconstruction)
        
        # Save results
        save_reconstruction(reconstruction, params['output_dir'])
        
        print("\nReconstruction completed successfully!")
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

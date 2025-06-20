import numpy as np
import astra
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
from scipy import ndimage
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

def reconstruct_3d_parallel_beam(projections, angles=None):
    """
    Reconstruct 3D volume from projection data with proper dimension handling.
    CPU-only: no GPU code.
    """
    height, num_angles, detector_width = projections.shape
    
    if angles is None:
        angles = np.linspace(0, np.pi, num_angles, False)
    
    print(f"\nStarting 3D reconstruction with {num_angles} angles...")
    print(f"Volume dimensions: {detector_width}x{detector_width}x{height}")
    print(f"Projection data shape: {projections.shape}")
    
    # Use CPU slice-by-slice reconstruction
    return reconstruct_slice_by_slice(projections, angles)

def reconstruct_slice_by_slice(projections, angles):
    """Slice-by-slice reconstruction with CPU-only algorithms and correct projector."""
    height, num_angles, detector_width = projections.shape
    
    reconstruction = np.zeros((height, detector_width, detector_width), dtype=np.float32)
    
    # Create 2D geometry
    vol_geom = astra.create_vol_geom(detector_width, detector_width)
    proj_geom = astra.create_proj_geom('parallel', 1.0, detector_width, angles)
    
    # Create projector - CPU only
    projector_id = astra.create_projector('line', proj_geom, vol_geom)
    
    print(f"Reconstructing {height} slices:")
    print(f"Expected sinogram shape per slice: ({num_angles}, {detector_width})")
    print(f"Created projector with ID: {projector_id}")
    
    for slice_idx in range(height):
        if slice_idx % 10 == 0 or slice_idx == height-1:
            print(f"\rProgress: {slice_idx+1}/{height} slices", end="", flush=True)
        
        sinogram = projections[slice_idx, :, :]  # Shape: (num_angles, detector_width)
        
        if sinogram.shape != (num_angles, detector_width):
            print(f"\nError: Sinogram dimensions {sinogram.shape} do not match expected ({num_angles}, {detector_width})")
            continue
        
        proj_id = astra.data2d.create('-sino', proj_geom, sinogram)
        rec_id = astra.data2d.create('-vol', vol_geom)
        
        # CPU-only: use FBP with projector, fallback to SIRT if needed
        try:
            cfg = astra.astra_dict('FBP')
            cfg['ReconstructionDataId'] = rec_id
            cfg['ProjectionDataId'] = proj_id
            cfg['ProjectorId'] = projector_id
            cfg['FilterType'] = 'Ram-Lak'
        except:
            cfg = astra.astra_dict('SIRT')
            cfg['ReconstructionDataId'] = rec_id
            cfg['ProjectionDataId'] = proj_id
            cfg['ProjectorId'] = projector_id
        
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        
        reconstruction[slice_idx] = astra.data2d.get(rec_id)
        
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(proj_id)
    
    astra.projector.delete(projector_id)
    print()  # New line after progress
    return reconstruction

def preprocess_projection_data(projections):
    """Preprocess projection data with global normalization and mild filtering."""
    print("\nPreprocessing projection data...")
    
    projections = projections.astype(np.float32)
    data_max = projections.max()
    if data_max > 60000:  # Likely 16-bit data
        print("Normalizing 16-bit data...")
        projections = projections / 65535.0
    elif data_max > 1.0:
        print("Normalizing data to [0,1] range...")
        projections = projections / data_max
    
    # Apply mild Gaussian filtering to reduce noise
    print("Applying mild noise reduction...")
    for i in range(projections.shape[0]):
        projections[i] = ndimage.gaussian_filter(projections[i], sigma=0.5)
    
    # Global normalization
    projections = (projections - projections.min()) / (projections.max() - projections.min())
    
    return projections


def visualize_3d_results(projections, reconstruction):
    """Enhanced visualization with more informative plots."""
    height, num_angles, detector_width = projections.shape
    
    reconstruction = (reconstruction - reconstruction.min()) / (reconstruction.max() - reconstruction.min())
    
    plt.figure(figsize=(18, 12))
    plt.suptitle("3D Reconstruction Results", fontsize=16, y=1.02)
    
    # Sample sinogram
    plt.subplot(2, 3, 1)
    mid_slice = height // 2
    plt.imshow(projections[mid_slice, :, :], cmap='gray', aspect='auto')
    plt.title(f'Sample Sinogram (Slice {mid_slice})')
    plt.xlabel('Detector Position')
    plt.ylabel('Projection Angle Index')
    plt.colorbar(label='Intensity')
    
    # Reconstructed slices
    slice_positions = [height//4, height//2, 3*height//4]
    for i, pos in enumerate(slice_positions):
        plt.subplot(2, 3, i+2)
        plt.imshow(reconstruction[pos, :, :], cmap='gray', vmin=0, vmax=1)
        plt.title(f'Reconstructed Slice {pos}')
        plt.axis('off')
        plt.colorbar(label='Density')
    
    # Sagittal view (side view)
    plt.subplot(2, 3, 5)
    sagittal = reconstruction[:, detector_width//2, :]
    plt.imshow(sagittal, cmap='gray', aspect='auto', 
               extent=[0, detector_width, height, 0])
    plt.title('Sagittal View (X-Z Plane)')
    plt.xlabel('X Position')
    plt.ylabel('Z (Height)')
    plt.colorbar(label='Density')
    
    # Coronal view (front view)
    plt.subplot(2, 3, 6)
    coronal = reconstruction[:, :, detector_width//2]
    plt.imshow(coronal, cmap='gray', aspect='auto',
               extent=[0, detector_width, height, 0])
    plt.title('Coronal View (Y-Z Plane)')
    plt.xlabel('Y Position')
    plt.ylabel('Z (Height)')
    plt.colorbar(label='Density')
    
    plt.tight_layout()
    plt.show()

def save_reconstruction(reconstruction, output_dir='reconstruction_output'):
    """Enhanced saving with metadata and multiple formats."""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'reconstruction_3d.npy'), reconstruction)
    with open(os.path.join(output_dir, 'metadata.txt'), 'w') as f:
        f.write(f"Reconstruction dimensions: {reconstruction.shape}\n")
        f.write(f"Data range: min={reconstruction.min():.4f}, max={reconstruction.max():.4f}\n")
        f.write(f"Data type: {reconstruction.dtype}\n")
    slice_dir = os.path.join(output_dir, 'slices')
    os.makedirs(slice_dir, exist_ok=True)
    print(f"\nSaving slices to {slice_dir}:")
    for i in range(reconstruction.shape[0]):
        if i % 10 == 0 or i == reconstruction.shape[0]-1:
            print(f"\rProgress: {i+1}/{reconstruction.shape[0]} slices", end="", flush=True)
        slice_img = reconstruction[i, :, :]
        slice_img = (255 * (slice_img - slice_img.min()) / 
                   (slice_img.max() - slice_img.min() + 1e-10)).astype(np.uint8)
        plt.imsave(os.path.join(slice_dir, f'slice_{i:03d}.png'), slice_img, cmap='gray')
    print("\nSaving complete.")

def main():
    """Enhanced main workflow with better parameter handling."""
    print("=" * 60)
    print("Enhanced 3D CT Reconstruction using ASTRA Toolbox")
    print("=" * 60)
    params = {
        'file_pattern': "./Phantom Dataset/Al phantom/*.txt",
        'expected_projections': 360,  # Adjust based on your data
        'output_dir': 'reconstruction_results'
    }
    try:
        projections = load_multiple_projection_files(
            file_pattern=params['file_pattern'],
            num_files=params['expected_projections'])
        if projections is None:
            return None
        projections = preprocess_projection_data(projections)
        reconstruction = reconstruct_3d_parallel_beam(
            projections)
        visualize_3d_results(projections, reconstruction)
        save_reconstruction(reconstruction, params['output_dir'])
        print("\nReconstruction completed successfully!")
        print(f"Final volume shape: {reconstruction.shape}")
        print(f"Data range: {reconstruction.min():.4f} to {reconstruction.max():.4f}")
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

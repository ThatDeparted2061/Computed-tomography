import numpy as np
import astra
import matplotlib.pyplot as plt
import os
import glob
from scipy import ndimage
import time
import re

def load_specific_projection_files(file_pattern="./Phantom/*.txt", 
                                 min_file_num=2, max_file_num=399,
                                 start_angle=0, end_angle=180, num_slices=271):
    """
    Enhanced version with better slice selection visualization
    """
    try:
        # Find and sort files
        def extract_number(filename):
            match = re.search(r'(\d+)', os.path.basename(filename))
            return int(match.group(1)) if match else -1

        all_files = sorted(glob.glob(file_pattern), key=extract_number)
        file_list = [f for f in all_files if min_file_num <= extract_number(f) <= max_file_num]

        if not file_list:
            raise FileNotFoundError(f"No files found in range {min_file_num}.txt to {max_file_num}.txt")

        # Load first file to get dimensions
        first_data = np.loadtxt(file_list[0])
        if first_data.ndim == 1:
            first_data = first_data.reshape(1, -1)

        total_slices_available = first_data.shape[0]
        detector_width = first_data.shape[1]
        
        # Calculate slice selection
        if num_slices is None or num_slices > total_slices_available:
            num_slices = total_slices_available
            slice_indices = range(total_slices_available)
        else:
            slice_step = max(1, total_slices_available // num_slices)
            slice_indices = range(0, total_slices_available, slice_step)
            num_slices = len(slice_indices)

        print(f"\nFile and Slice Information:")
        print(f"Found {len(file_list)} files in range {min_file_num}.txt to {max_file_num}.txt")
        print(f"Total available slices per file: {total_slices_available}")
        print(f"Selected {num_slices} slices with step {slice_step if 'slice_step' in locals() else 1}")
        print(f"Slice indices: {slice_indices[0]} to {slice_indices[-1]} (step {slice_indices[1]-slice_indices[0] if len(slice_indices)>1 else 1})")
        print(f"Detector width: {detector_width} pixels")
        print(f"Angle range: {start_angle}° to {end_angle}°")

        # Initialize projection array
        projections = np.zeros((num_slices, len(file_list), detector_width), dtype=np.float32)
        angles = np.linspace(start_angle, end_angle, len(file_list))

        # Load data with progress and slice info
        print("\nLoading projection files:")
        for i, filename in enumerate(file_list):
            data = np.loadtxt(filename)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            if i == 0:  # Show slice example for first file
                plt.figure(figsize=(12, 6))
                plt.suptitle(f"Slice Selection Example (File: {os.path.basename(filename)})", y=1.02)
                
                plt.subplot(1, 2, 1)
                plt.imshow(data, cmap='gray', aspect='auto')
                plt.title(f'All {data.shape[0]} Slices')
                plt.xlabel('Detector Position')
                plt.ylabel('Original Slice Index')
                plt.colorbar(label='Intensity')
                
                plt.subplot(1, 2, 2)
                selected_data = data[slice_indices, :]
                plt.imshow(selected_data, cmap='gray', aspect='auto')
                plt.title(f'Selected {len(slice_indices)} Slices')
                plt.xlabel('Detector Position')
                plt.ylabel('Selected Slice Index')
                plt.colorbar(label='Intensity')
                
                plt.tight_layout()
                plt.show()

            projections[:, i, :] = data[slice_indices, :]

            if i % 10 == 0 or i == len(file_list)-1:
                print(f"\rProgress: {i+1}/{len(file_list)} files loaded", end="", flush=True)

        print(f"\nFinal projection data shape: (slices, angles, detector) = {projections.shape}")
        print(f"Data range: min={projections.min():.2f}, max={projections.max():.2f}")

        return projections, np.radians(angles)

    except Exception as e:
        print(f"\nError loading files: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def visualize_slice_selection(projections):
    """Visualize how slices are distributed in the volume"""
    num_slices, num_angles, detector_width = projections.shape
    
    plt.figure(figsize=(15, 8))
    plt.suptitle("Slice Distribution in Reconstruction Volume", y=1.02)
    
    # Show slice positions
    plt.subplot(2, 2, 1)
    plt.plot(range(num_slices), np.zeros(num_slices), '|', markersize=20)
    plt.title('Slice Positions in Z-direction')
    plt.xlabel('Slice Index')
    plt.yticks([])
    plt.grid(True)
    
    # Show middle slices - fixed to show 2D data
    for i, pos in enumerate([0, num_slices//4, num_slices//2, 3*num_slices//4, num_slices-1]):
        plt.subplot(2, 5, 6 + i)
        # Ensure we're displaying a 2D slice (angle vs detector position)
        slice_data = projections[pos, :, :].T  # Transpose to get correct orientation
        plt.imshow(slice_data, cmap='gray', aspect='auto')
        plt.title(f'Slice {pos}\nAngle {num_angles//2}')
        plt.xlabel('Angle Index')
        plt.ylabel('Detector Position')
    
    plt.tight_layout()
    plt.show()

def reconstruct_slice_by_slice(projections, angles):
    """Reconstruct each slice separately using FBP"""
    num_slices, num_angles, det_width = projections.shape
    
    # ASTRA geometry setup
    vol_geom = astra.create_vol_geom(det_width, det_width)
    proj_geom = astra.create_proj_geom('parallel', 1.0, det_width, angles)
    
    # Create ASTRA projector
    projector_id = astra.create_projector('strip', proj_geom, vol_geom)
    
    # Initialize reconstruction volume
    reconstruction = np.zeros((num_slices, det_width, det_width), dtype=np.float32)
    
    print("\nReconstructing slices:")
    for i in range(num_slices):
        # Create sinogram for this slice
        sinogram_id = astra.data2d.create('-sino', proj_geom, projections[i])
        
        # Create reconstruction data
        rec_id = astra.data2d.create('-vol', vol_geom)
        
        # Configure and run FBP
        cfg = astra.astra_dict('FBP')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sinogram_id
        cfg['ProjectorId'] = projector_id
        
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        
        # Get reconstruction
        reconstruction[i] = astra.data2d.get(rec_id)
        
        # Clean up
        astra.data2d.delete(sinogram_id)
        astra.data2d.delete(rec_id)
        astra.algorithm.delete(alg_id)
        
        if i % 10 == 0 or i == num_slices-1:
            print(f"\rProgress: {i+1}/{num_slices} slices reconstructed", end="", flush=True)
    
    # Clean up projector
    astra.projector.delete(projector_id)
    
    return reconstruction

def preprocess_projection_data(projections, apply_alignment=True):
    """Preprocess projection data with optional alignment"""
    print("\nPreprocessing projections...")
    
    # Basic preprocessing
    projections = np.maximum(projections, 0)  # Remove negative values
    
    if apply_alignment:
        print("Applying alignment correction...")
        # Simple vertical alignment (example - adjust as needed)
        for i in range(projections.shape[0]):
            projections[i] = ndimage.shift(projections[i], [0, 0], mode='nearest')
    
    # Normalize each slice
    for i in range(projections.shape[0]):
        projections[i] /= np.max(projections[i]) if np.max(projections[i]) > 0 else 1
    
    print("Preprocessing complete.")
    return projections

def save_reconstruction(reconstruction, output_dir):
    """Save reconstruction results"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save as numpy array
    np.save(os.path.join(output_dir, 'reconstruction.npy'), reconstruction)
    
    # Save middle slices as images
    num_slices = reconstruction.shape[0]
    for i in [0, num_slices//4, num_slices//2, 3*num_slices//4, num_slices-1]:
        plt.imsave(os.path.join(output_dir, f'slice_{i:04d}.png'), reconstruction[i], cmap='gray')
    
    print(f"\nResults saved to {output_dir}")

def reconstruct_and_visualize(projections, angles):
    """Enhanced reconstruction with better slice visualization"""
    print("\nStarting reconstruction...")
    height, num_angles, detector_width = projections.shape
    
    # Show reconstruction parameters
    print(f"Reconstruction Parameters:")
    print(f"- Volume dimensions: {detector_width}x{detector_width}x{height}")
    print(f"- Number of angles: {num_angles}")
    print(f"- Angle range: {np.degrees(angles[0]):.1f}° to {np.degrees(angles[-1]):.1f}°")
    
    # Visualize slice selection before reconstruction
    visualize_slice_selection(projections)
    
    # Reconstruction
    reconstruction = reconstruct_slice_by_slice(projections, angles)
    
    # Enhanced visualization
    plt.figure(figsize=(18, 12))
    plt.suptitle("Reconstruction Results with Slice Information", y=1.02)
    
    # Show slice positions in reconstruction
    z_positions = [0, height//4, height//2, 3*height//4, height-1]
    
    for i, z in enumerate(z_positions):
        plt.subplot(2, 5, i+1)
        plt.imshow(reconstruction[z], cmap='gray')
        plt.title(f'Slice {z} (Z={z})')
        plt.axis('off')
        
        # Add profile plots
        plt.subplot(2, 5, 6 + i)
        plt.plot(reconstruction[z, detector_width//2, :], label='Horizontal')
        plt.plot(reconstruction[z, :, detector_width//2], label='Vertical')
        plt.title(f'Profiles at Slice {z}')
        plt.grid(True)
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Additional visualization of every 30th slice
    slice_step = 30
    num_additional_slices = height // slice_step
    if num_additional_slices > 0:
        cols = min(5, num_additional_slices)
        rows = (num_additional_slices + cols - 1) // cols
        
        plt.figure(figsize=(15, 3*rows))
        plt.suptitle(f"Every {slice_step}th Reconstructed Slice (Total: {num_additional_slices})", y=1.02)
        
        for i, z in enumerate(range(0, height, slice_step)):
            plt.subplot(rows, cols, i+1)
            plt.imshow(reconstruction[z], cmap='gray')
            plt.title(f'Slice {z}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return reconstruction

def main():
    print("=" * 70)
    print("ENHANCED 3D CT RECONSTRUCTION WITH CLEAR SLICE VISUALIZATION")
    print("=" * 70)
    
    # Separate loading parameters from output parameters
    load_params = {
        'file_pattern': "./Phantom/*.txt",
        'min_file_num': 2,
        'max_file_num': 399,
        'start_angle': 0,
        'end_angle': 180,
        'num_slices': 271
    }
    
    output_dir = 'enhanced_reconstruction_output'
    
    try:
        # Load data with enhanced visualization
        projections, angles = load_specific_projection_files(**load_params)
        if projections is None:
            return None
        
        # Preprocess
        projections = preprocess_projection_data(projections, apply_alignment=True)
        
        # Reconstruct with enhanced visualization
        reconstruction = reconstruct_and_visualize(projections, angles)
        
        # Save results
        save_reconstruction(reconstruction, output_dir)
        
        print("\nReconstruction completed successfully!")
        print(f"Final volume shape (Z,Y,X): {reconstruction.shape}")
        
        return reconstruction
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    start_time = time.time()
    volume = main()
    duration = time.time() - start_time
    print(f"\nTotal execution time: {duration:.2f} seconds")

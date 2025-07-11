import numpy as np
import astra
import matplotlib.pyplot as plt
import os
import glob
from scipy import ndimage
import time
import re
import warnings

# Suppress ASTRA warnings if they are too verbose
# warnings.filterwarnings("ignore", category=astra.AstraWarning) 

def load_specific_projection_files(file_pattern, min_file_num, max_file_num, 
                                     start_angle, end_angle, num_slices=None):
    """
    Loads projection data from a sequence of text files.
    Assumes files are named numerically (e.g., 2.txt, 3.txt, ..., 400.txt).
    Each file is expected to contain a 2D array where rows are slices (Z-axis)
    and columns are detector pixels (X-axis).
    
    Args:
        file_pattern (str): Glob pattern for projection files (e.g., "./Phantom/*.txt").
        min_file_num (int): The starting number of the projection files (e.g., 2).
        max_file_num (int): The ending number of the projection files (e.g., 400).
        start_angle (float): Starting angle of the scan in degrees.
        end_angle (float): Ending angle of the scan in degrees.
        num_slices (int, optional): Number of slices to select for reconstruction.
                                    If None, all available slices are used.
                                    If specified, slices are sampled uniformly.
                                    Defaults to None.

    Returns:
        tuple: (projections (np.ndarray), angles (np.ndarray))
               projections: (selected_num_slices, num_angles, detector_width)
               angles: Radians array of projection angles.
               Returns (None, None) if an error occurs.
    """
    try:
        def extract_number(filename):
            match = re.search(r'(\d+)', os.path.basename(filename))
            return int(match.group(1)) if match else -1

        all_files = sorted(glob.glob(file_pattern), key=extract_number)
        file_list = [f for f in all_files if min_file_num <= extract_number(f) <= max_file_num]

        expected_count = max_file_num - min_file_num + 1
        if len(file_list) != expected_count:
            missing = sorted(set(range(min_file_num, max_file_num + 1)) -
                             {extract_number(f) for f in file_list})
            print(f"\n[ERROR] Expected {expected_count} files ({min_file_num}.txt to {max_file_num}.txt), "
                  f"found {len(file_list)}. Missing files: {missing}")
            return None, None
        
        if not file_list:
            print(f"[ERROR] No files found matching pattern '{file_pattern}' in range {min_file_num}-{max_file_num}.")
            return None, None

        # Load first file to get dimensions
        first_data = np.loadtxt(file_list[0])
        if first_data.ndim == 1:
            first_data = first_data.reshape(1, -1)

        total_slices_available = first_data.shape[0]
        detector_width = first_data.shape[1]
        
        # Calculate slice selection
        if num_slices is None or num_slices > total_slices_available:
            num_slices_to_use = total_slices_available
            slice_indices = np.arange(total_slices_available)
        else:
            slice_indices = np.linspace(0, total_slices_available - 1, num_slices, dtype=int)
            num_slices_to_use = len(slice_indices)

        print(f"\n--- File and Slice Information ---")
        print(f"Found {len(file_list)} projection files ({min_file_num}.txt to {max_file_num}.txt)")
        print(f"Total available slices per file: {total_slices_available}")
        print(f"Selected {num_slices_to_use} slices for reconstruction.")
        print(f"Slice indices being used: {slice_indices[0]} to {slice_indices[-1]} (step approx. {slice_indices[1]-slice_indices[0] if len(slice_indices)>1 else 1})")
        print(f"Detector width (pixels): {detector_width}")
        print(f"Angle range: {start_angle}° to {end_angle}° ({len(file_list)} angles)")

        # Initialize projection array: (slices, angles, detector_width)
        projections = np.zeros((num_slices_to_use, len(file_list), detector_width), dtype=np.float32)
        angles = np.linspace(start_angle, end_angle, len(file_list), endpoint=False) # Use endpoint=False for 0-180 if 180 is not a duplicate of 0

        # Load data with progress and slice info
        print("\nLoading projection files:")
        for i, filename in enumerate(file_list):
            data = np.loadtxt(filename)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            if data.shape[0] != total_slices_available or data.shape[1] != detector_width:
                warnings.warn(f"File {os.path.basename(filename)} has inconsistent dimensions. Expected "
                              f"({total_slices_available}, {detector_width}), got {data.shape}. Skipping.")
                continue

            projections[:, i, :] = data[slice_indices, :]

            if i % (len(file_list) // 10 + 1) == 0 or i == len(file_list) - 1: # Update progress every 10%
                print(f"\rProgress: {i+1}/{len(file_list)} files loaded", end="", flush=True)

        print(f"\nFinal projection data shape: (slices, angles, detector) = {projections.shape}")
        print(f"Raw data range: min={projections.min():.2f}, max={projections.max():.2f}")

        # Optional: Display slice selection for the first loaded file
        if True: # Set to False to disable this plot
            plt.figure(figsize=(12, 6))
            plt.suptitle(f"Slice Selection Example (File: {os.path.basename(file_list[0])})", y=1.02)
            
            plt.subplot(1, 2, 1)
            plt.imshow(first_data, cmap='gray', aspect='auto')
            plt.title(f'All {first_data.shape[0]} Slices in One File')
            plt.xlabel('Detector Position (pixels)')
            plt.ylabel('Original Slice Index (pixels)')
            plt.colorbar(label='Raw Intensity')
            
            plt.subplot(1, 2, 2)
            selected_data = first_data[slice_indices, :]
            plt.imshow(selected_data, cmap='gray', aspect='auto')
            plt.title(f'Selected {len(slice_indices)} Slices for Reco')
            plt.xlabel('Detector Position (pixels)')
            plt.ylabel('Selected Slice Index (pixels)')
            plt.colorbar(label='Raw Intensity')
            
            plt.tight_layout()
            plt.show()

        return projections, np.radians(angles)

    except Exception as e:
        print(f"\n[ERROR] Error loading files: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def preprocess_projection_data(projections, apply_log_transform=True, use_flat_field_correction=True):
    """
    Improved preprocessing for cone beam CT.

    Args:
        projections (np.ndarray): Raw projection data (slices, angles, detector_width).
        apply_log_transform (bool): Whether to apply -log(I/I0) transform. 
                                    Set to False if data is already attenuation-corrected.
        use_flat_field_correction (bool): Whether to apply a simple row-wise flat field correction.
                                          Consider more robust methods if needed.

    Returns:
        np.ndarray: Preprocessed projection data.
    """
    print("\n--- Preprocessing Projections ---")
    
    # 1. Convert to attenuation values: I = I0 * exp(-mu*L) => -log(I/I0) = mu*L
    # Assuming projections are raw intensity measurements (I), and I0 is max possible intensity (65535 for 16-bit)
    if apply_log_transform:
        print("Applying -log(I/I0) transform...")
        # Add a small epsilon to avoid log(0) if data contains zeros
        # Also, clip data to avoid values > I0, which would result in negative attenuation
        projections = np.maximum(projections, 1) 
        # For I0, a common choice is the maximum value in the empty beam or detector's max range.
        # If 65535 is a fixed detector saturation value, keep it. Otherwise, use projections.max()
        # or load a separate 'bright field' image.
        I0 = 65535.0 # Or consider np.max(projections) or a dedicated bright field
        projections = -np.log(projections / I0)
        print(f"Post-log transform data range: min={projections.min():.4f}, max={projections.max():.4f}")
    else:
        print("Skipping -log(I/I0) transform (assuming data is already attenuation-corrected).")

    # 2. Apply simple row-wise flat field correction
    if use_flat_field_correction:
        print("Applying simple row-wise flat field correction...")
        # This correction assumes that the mean intensity across angles for each detector row
        # represents a baseline, and deviations from this baseline are due to detector non-uniformity.
        # More robust methods use dedicated flat-field (bright field) and dark-field images.
        for i in range(projections.shape[0]): # Iterate through slices (detector rows in Z)
            # Calculate the mean intensity for each column (detector pixel) across all angles for this slice
            col_means_across_angles = np.mean(projections[i], axis=0)
            # Avoid division by zero
            col_means_across_angles[col_means_across_angles == 0] = 1e-6 
            projections[i] /= col_means_across_angles # Normalize each pixel by its average over angles

        # A more standard flat/dark field correction would be:
        # projections = -np.log((raw_data - dark_field) / (bright_field - dark_field + 1e-9))
        # This requires `dark_field` and `bright_field` arrays.
        print(f"Post flat-field data range: min={projections.min():.4f}, max={projections.max():.4f}")
    else:
        print("Skipping flat field correction.")
    
    # 3. Apply mild denoising
    print("Applying mild Gaussian denoising (sigma=0.8) to each 2D projection...")
    for i in range(projections.shape[0]): # For each slice
        # Apply 2D Gaussian filter to each sinogram slice
        projections[i, :, :] = ndimage.gaussian_filter(projections[i, :, :], sigma=0.8)
    print(f"Post denoising data range: min={projections.min():.4f}, max={projections.max():.4f}")

    # Optional: Final normalization (usually for visualization, not strictly for ASTRA)
    # ASTRA expects relative attenuation values. Normalizing to 0-1 for reconstruction can
    # sometimes cause issues with quantitative accuracy. It's often better to let ASTRA handle
    # the internal scaling, or perform this *after* reconstruction for display purposes.
    # If your data is very noisy or has extreme outliers, this can help.
    # print("Normalizing each 2D projection to 0-1 range (for display/stability)...")
    # for i in range(projections.shape[0]): # For each slice
    #     min_val = np.percentile(projections[i], 1)
    #     max_val = np.percentile(projections[i], 99)
    #     projections[i] = (projections[i] - min_val) / (max_val - min_val + 1e-6)
    #     projections[i] = np.clip(projections[i], 0, 1) # Ensure clipping after normalization

    print("Preprocessing complete.")
    return projections

def setup_cone_geometry(projections, angles, 
                        source_to_detector, source_to_object, 
                        detector_pixel_size, reconstructed_voxel_size=None):
    """
    Setup cone beam geometry for ASTRA.

    Args:
        projections (np.ndarray): Preprocessed projection data (slices, angles, detector_width).
        angles (np.ndarray): Projection angles in radians.
        source_to_detector (float): Distance from X-ray source to detector in mm.
        source_to_object (float): Distance from X-ray source to the center of rotation (object origin) in mm.
        detector_pixel_size (float): Physical size of a single detector pixel in mm.
        reconstructed_voxel_size (float, optional): Desired voxel size of the reconstructed volume in mm.
                                                    If None, it's set equal to detector_pixel_size 
                                                    scaled by the magnification at the object plane.

    Returns:
        tuple: (proj_geom (dict), vol_geom (dict))
               proj_geom: ASTRA projection geometry.
               vol_geom: ASTRA volume geometry.
    """
    detector_rows, num_angles, detector_cols = projections.shape
    
    # Calculate derived geometric parameters
    origin_to_detector = source_to_detector - source_to_object
    
    # Magnification at the object plane (center of rotation)
    magnification = source_to_detector / source_to_object
    
    # Determine the effective voxel size and volume size
    if reconstructed_voxel_size is None:
        # Default to a voxel size that scales with the detector pixel size at the object plane
        reconstructed_voxel_size = detector_pixel_size / magnification
        print(f"Auto-determining voxel size. Set to {reconstructed_voxel_size:.4f} mm (detector_pixel_size / magnification)")
    
    # Calculate volume dimensions based on the field of view covered by the detector
    # and the desired voxel size.
    
    # The ASTRA volume geometry is centered at the origin.
    # We need to ensure the volume covers the projected area.
    
    # Max lateral extent of the object that can be seen without truncation
    # = (detector_cols * detector_pixel_size) * (source_to_object / source_to_detector)
    # This is approx (detector_cols * detector_pixel_size) / magnification
    
    # Convert this physical extent to voxels
    vol_size_xy_mm = detector_cols * detector_pixel_size * (source_to_object / source_to_detector)
    vol_size_z_mm = detector_rows * detector_pixel_size * (source_to_object / source_to_detector) # Approx. for Z-direction
    
    # Ensure vol_size is an integer number of voxels, making it slightly larger if necessary
    vol_size_x = int(np.ceil(vol_size_xy_mm / reconstructed_voxel_size))
    vol_size_y = vol_size_x # Assuming square pixels and reconstruction volume
    vol_size_z = int(np.ceil(vol_size_z_mm / reconstructed_voxel_size))
    
    # It's often good to ensure vol_size is even for FDK algorithms, or just round up for safety
    if vol_size_x % 2 != 0: vol_size_x += 1
    if vol_size_y % 2 != 0: vol_size_y += 1
    if vol_size_z % 2 != 0: vol_size_z += 1
    
    # Create cone beam geometry
    proj_geom = astra.create_proj_geom(
        'cone',
        detector_pixel_size,  # Detector pixel size Y (vertical, rows)
        detector_pixel_size,  # Detector pixel size X (horizontal, columns)
        detector_rows,        # Detector rows (height, Z-dimension of projection image)
        detector_cols,        # Detector columns (width, X-dimension of projection image)
        angles,               # Projection angles
        source_to_object,     # Distance source to origin (SOD)
        origin_to_detector    # Distance origin to detector (ODD)
    )
    
    # Create volume geometry
    vol_geom = astra.create_vol_geom(vol_size_x, vol_size_y, vol_size_z, 
                                     -vol_size_x/2 * reconstructed_voxel_size, 
                                     vol_size_x/2 * reconstructed_voxel_size, 
                                     -vol_size_y/2 * reconstructed_voxel_size, 
                                     vol_size_y/2 * reconstructed_voxel_size, 
                                     -vol_size_z/2 * reconstructed_voxel_size, 
                                     vol_size_z/2 * reconstructed_voxel_size)
    
    print("\n--- Geometry Parameters ---")
    print(f"Source-to-detector distance (SD): {source_to_detector:.2f} mm")
    print(f"Source-to-object distance (SO): {source_to_object:.2f} mm")
    print(f"Origin-to-detector distance (OD): {origin_to_detector:.2f} mm")
    print(f"Calculated Magnification (SD/SO): {magnification:.3f}")
    print(f"Detector pixel size: {detector_pixel_size} mm")
    print(f"Reconstructed Voxel Size: {reconstructed_voxel_size:.4f} mm")
    print(f"Reconstructed Volume Size (voxels): {vol_size_x}x{vol_size_y}x{vol_size_z}")
    print(f"Reconstructed Volume Extent (mm): {vol_size_x * reconstructed_voxel_size:.2f} x {vol_size_y * reconstructed_voxel_size:.2f} x {vol_size_z * reconstructed_voxel_size:.2f}")
    
    return proj_geom, vol_geom, reconstructed_voxel_size # Return voxel size for later use

def reconstruct_cone_beam(projections, angles, proj_geom, vol_geom, algorithm_type='FDK_CUDA', num_iterations=50):
    """
    Reconstructs a 3D volume from cone beam projections using ASTRA.

    Args:
        projections (np.ndarray): Preprocessed projection data (slices, angles, detector_width).
        angles (np.ndarray): Projection angles in radians.
        proj_geom (dict): ASTRA projection geometry.
        vol_geom (dict): ASTRA volume geometry.
        algorithm_type (str): Reconstruction algorithm ('FDK_CUDA' or 'SIRT3D_CUDA').
                              'FDK_CUDA' is generally faster for GPU.
                              'SIRT3D_CUDA' (or 'SIRT3D_CPU') is iterative and often better for noise/artifacts.
        num_iterations (int): Number of iterations for iterative algorithms like SIRT. Ignored for FDK.

    Returns:
        np.ndarray: Reconstructed 3D volume.
    """
    print(f"\n--- Starting Cone Beam Reconstruction ({algorithm_type}) ---")
    
    # Create ASTRA data objects
    proj_id = astra.data3d.create('-proj3d', proj_geom, projections)
    rec_id = astra.data3d.create('-vol', vol_geom)
    
    # Configure reconstruction algorithm
    if algorithm_type == 'FDK_CUDA':
        
        cfg = astra.astra_dict('FDK_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        cfg['option'] = {
            'ShortScan': False,  # Set to True if your scan range is less than 180+cone_angle_span
            'FilterType': 'Ram-Lak', # Common filters: 'Ram-Lak', 'Shepp-Logan', 'Cosine'
            'WindowType': 'Hamming'  # Common windows: 'Hamming', 'Hann', 'Cylindric'
        }
        alg_id = astra.algorithm.create(cfg)
        
        print("Running FDK reconstruction on GPU...")
        astra.algorithm.run(alg_id)
        
    elif algorithm_type == 'SIRT3D_CUDA':
        
        if algorithm_type == 'SIRT3D_CPU':
             cfg = astra.astra_dict('SIRT3D_CPU')
        else:
             cfg = astra.astra_dict('SIRT3D_CUDA')

        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        # For iterative algorithms, you often need a projector
        projector_id = astra.create_projector('cuda3d' if 'CUDA' in algorithm_type else 'fp3d', proj_geom, vol_geom)
        cfg['ProjectorId'] = projector_id
        
        cfg['option'] = {
            'MinConstraint': 0.0, # Attenuation coefficients are non-negative
            'MaxConstraint': 1.0 # Or based on expected max attenuation of material
        }
        alg_id = astra.algorithm.create(cfg)
        
        print(f"Running {algorithm_type} reconstruction for {num_iterations} iterations...")
        astra.algorithm.run(alg_id, num_iterations)
        
        astra.projector.delete(projector_id) # Clean up projector
        
    else:
        raise ValueError(f"Unsupported algorithm type: {algorithm_type}. Choose 'FDK_CUDA' or 'SIRT3D_CUDA'.")

    # Get reconstruction result
    reconstruction = astra.data3d.get(rec_id)
    
    # Clean up ASTRA objects
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(proj_id)
    astra.data3d.delete(rec_id)
    
    print("Reconstruction complete.")
    return reconstruction

def visualize_reconstruction(reconstruction, projections, angles, voxel_size_mm):
    """
    Improved visualization of reconstructed slices and sinograms.
    
    Args:
        reconstruction (np.ndarray): The 3D reconstructed volume.
        projections (np.ndarray): The original projection data (slices, angles, detector_width).
        angles (np.ndarray): Angles in radians.
        voxel_size_mm (float): The physical size of a single voxel in mm.
    """
    detector_rows, num_angles, detector_cols = projections.shape
    vol_size_z, vol_size_y, vol_size_x = reconstruction.shape # ASTRA returns Z, Y, X order

    print("\n--- Visualizing Reconstruction ---")
    
    # Enhanced normalization for display - focus on middle 98% of intensities
    # This helps in visualizing details by clipping extreme outliers
    vmin_display = np.percentile(reconstruction, 1)
    vmax_display = np.percentile(reconstruction, 99)
    print(f"Display range for reconstruction: [{vmin_display:.4f}, {vmax_display:.4f}]")

    # 1. Show 10 reconstructed axial slices in a grid format
    plt.figure(figsize=(16, 9))
    plt.suptitle("Reconstructed Axial Slices (Z-Planes)", y=1.02, fontsize=16)
    
    # Select slices across the Z-dimension
    slice_indices_z = np.linspace(0, vol_size_z - 1, 10, dtype=int)
    for i, z_idx in enumerate(slice_indices_z):
        plt.subplot(2, 5, i + 1)
        plt.imshow(reconstruction[z_idx, :, :], cmap='gray', 
                   vmin=vmin_display, vmax=vmax_display,
                   extent=[-vol_size_x/2*voxel_size_mm, vol_size_x/2*voxel_size_mm, 
                           -vol_size_y/2*voxel_size_mm, vol_size_y/2*voxel_size_mm])
        plt.title(f'Z-Slice {z_idx}')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.gca().set_aspect('equal', adjustable='box') # Ensure aspect ratio is correct
        plt.colorbar(label='Attenuation (a.u.)', fraction=0.046, pad=0.04) # Add colorbar to each subplot
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()
    
    # 2. Show orthogonal views (axial, sagittal, coronal)
    plt.figure(figsize=(18, 6))
    plt.suptitle("Orthogonal Views of Reconstructed Volume", y=1.02, fontsize=16)

    # Central Axial Slice (XY plane)
    plt.subplot(1, 3, 1)
    plt.imshow(reconstruction[vol_size_z // 2, :, :], cmap='gray', 
               vmin=vmin_display, vmax=vmax_display,
               extent=[-vol_size_x/2*voxel_size_mm, vol_size_x/2*voxel_size_mm, 
                       -vol_size_y/2*voxel_size_mm, vol_size_y/2*voxel_size_mm])
    plt.title(f'Central Axial Slice (Z={vol_size_z//2})')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar(label='Attenuation (a.u.)', fraction=0.046, pad=0.04)
    
    # Central Sagittal Slice (YZ plane)
    plt.subplot(1, 3, 2)
    # ASTRA volume is Z, Y, X. For Sagittal (YZ), we take reconstruction[:, :, center_x]
    # The image will be Z vs Y.
    plt.imshow(reconstruction[:, vol_size_y // 2, :], cmap='gray', 
               vmin=vmin_display, vmax=vmax_display,
               aspect='auto', # Aspect auto as Y and Z dimensions might not be equal
               extent=[-vol_size_x/2*voxel_size_mm, vol_size_x/2*voxel_size_mm, 
                       vol_size_z/2*voxel_size_mm, -vol_size_z/2*voxel_size_mm]) # Invert Y-axis if needed
    plt.title(f'Central Coronal Slice (Y={vol_size_y//2})') # Corrected: Y-slice is Coronal
    plt.xlabel('X (mm)')
    plt.ylabel('Z (mm)')
    plt.colorbar(label='Attenuation (a.u.)', fraction=0.046, pad=0.04)
    
    # Central Coronal Slice (XZ plane)
    plt.subplot(1, 3, 3)
    # ASTRA volume is Z, Y, X. For Coronal (XZ), we take reconstruction[:, center_y, :]
    # The image will be Z vs X.
    plt.imshow(reconstruction[:, :, vol_size_x // 2], cmap='gray', 
               vmin=vmin_display, vmax=vmax_display,
               aspect='auto', # Aspect auto as X and Z dimensions might not be equal
               extent=[-vol_size_y/2*voxel_size_mm, vol_size_y/2*voxel_size_mm, 
                       vol_size_z/2*voxel_size_mm, -vol_size_z/2*voxel_size_mm]) # Invert Y-axis if needed
    plt.title(f'Central Sagittal Slice (X={vol_size_x//2})') # Corrected: X-slice is Sagittal
    plt.xlabel('Y (mm)')
    plt.ylabel('Z (mm)')
    plt.colorbar(label='Attenuation (a.u.)', fraction=0.046, pad=0.04)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    # 3. Improved sinogram display (example at different detector rows/vertical positions)
    plt.figure(figsize=(18, 6))
    plt.suptitle("Example Sinograms at Different Detector Row Z-Positions", y=1.02, fontsize=16)
    
    # Select specific detector rows (Z-positions on the detector)
    det_rows_to_show = [detector_rows // 4, detector_rows // 2, 3 * detector_rows // 4]
    
    for i, det_row_idx in enumerate(det_rows_to_show):
        plt.subplot(1, 3, i + 1)
        # Extract the sinogram for a specific detector row (slice_index in your projections)
        # The sinogram is (angles, detector_width)
        sino = projections[det_row_idx, :, :] 
        
        plt.imshow(sino, cmap='gray', aspect='auto', 
                   extent=[np.degrees(angles[0]), np.degrees(angles[-1]), 0, detector_cols],
                   origin='lower') # Use origin='lower' for correct orientation
        plt.title(f'Sinogram at Detector Row (Z) {det_row_idx}')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Detector Column (X-position on detector)')
        plt.colorbar(label='Attenuation (a.u.)', fraction=0.046, pad=0.04)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def save_reconstruction(reconstruction, output_dir, params):
    """
    Save reconstruction results and metadata.

    Args:
        reconstruction (np.ndarray): The 3D reconstructed volume.
        output_dir (str): Directory to save the results.
        params (dict): Dictionary of parameters used for reconstruction.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save as numpy array
    npy_path = os.path.join(output_dir, 'reconstruction.npy')
    np.save(npy_path, reconstruction)
    print(f"\nReconstruction saved as NumPy array: {npy_path}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, 'metadata.txt')
    with open(metadata_path, 'w') as f:
        f.write(f"--- Reconstruction Metadata ---\n")
        f.write(f"Timestamp: {time.ctime()}\n")
        f.write(f"Reconstruction dimensions (Z, Y, X): {reconstruction.shape}\n")
        f.write(f"Reconstructed Voxel Size (mm): {params.get('reconstructed_voxel_size', 'N/A')}\n")
        f.write(f"Data range: min={reconstruction.min():.6f}, max={reconstruction.max():.6f}\n")
        f.write(f"Mean value: {reconstruction.mean():.6f}\n")
        f.write(f"Standard deviation: {reconstruction.std():.6f}\n\n")
        
        f.write(f"--- Input Parameters ---\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    print(f"Metadata saved to: {metadata_path}")

def save_reconstructed_slices(reconstruction, output_dir, file_prefix='slice_', cmap='gray', vmin=None, vmax=None):
    """
    Saves all axial (XY) slices of the 3D reconstruction as individual image files.

    Args:
        reconstruction (np.ndarray): The 3D reconstructed volume (Z, Y, X).
        output_dir (str): Directory where individual slice images will be saved.
        file_prefix (str): Prefix for the slice filenames (e.g., 'slice_0000.png').
        cmap (str): Colormap to use for saving images (e.g., 'gray').
        vmin (float, optional): Minimum value for colormap scaling. If None, uses automatic scaling.
        vmax (float, optional): Maximum value for colormap scaling. If None, uses automatic scaling.
    """
    slice_output_dir = os.path.join(output_dir, "reconstructed_slices")
    if not os.path.exists(slice_output_dir):
        os.makedirs(slice_output_dir)
        print(f"\nCreated directory for slices: {slice_output_dir}")
    else:
        print(f"\nSaving slices to existing directory: {slice_output_dir}")

    num_slices = reconstruction.shape[0] # Z-dimension is the number of axial slices

    # Determine display range if not provided
    if vmin is None:
        vmin = np.percentile(reconstruction, 1)
    if vmax is None:
        vmax = np.percentile(reconstruction, 99)
    print(f"Saving slices with display range: [{vmin:.4f}, {vmax:.4f}]")

    print(f"Saving {num_slices} individual slices...")
    for i in range(num_slices):
        slice_data = reconstruction[i, :, :]
        filename = os.path.join(slice_output_dir, f"{file_prefix}{i:04d}.png") # 04d for 4-digit padding
        
        plt.imsave(filename, slice_data, cmap=cmap, vmin=vmin, vmax=vmax)
        
        if (i + 1) % (num_slices // 10 + 1) == 0 or i == num_slices - 1: # Update progress every 10%
            print(f"\rProgress: {i+1}/{num_slices} slices saved", end="", flush=True)
    print("\nAll slices saved.")


def main():
    print("=" * 70)
    print("           IMPROVED CONE BEAM CT RECONSTRUCTION            ")
    print("=" * 70)
    
    # --- Configuration Parameters ---
    params = {
        'file_pattern': "./Phantom/*.txt",
        'min_file_num': 2,
        'max_file_num': 400, # Max file number
        'num_slices_to_reconstruct': None, # Set to an int (e.g., 50) to reconstruct fewer slices, or None for all
        'start_angle': 0, # Degrees
        'end_angle': 180, # Degrees (often 180 or 360 for full scan)
        'source_to_detector': 1000.0, # mm
        'source_to_object': 900.0,   # mm (center of rotation)
        'detector_pixel_size': 0.127, # mm (e.g., 127 microns)
        'reconstructed_voxel_size': None, # mm. If None, derived from detector_pixel_size and magnification.
                                         # e.g., 0.08 mm to get a specific resolution.
        'output_dir': 'cone_beam_reconstruction_results',
        'apply_log_transform_on_load': True, # Set to False if your .txt files are already attenuation values
        'apply_flat_field_correction': False, # Set to False if already corrected or using external dark/bright fields
        'reconstruction_algorithm': 'FDK_CUDA', # Options: 'FDK_CUDA', 'SIRT3D_CUDA'
        'sirt_iterations': 50 # Only applies to SIRT
    }
    
    # Validate angles:
    # ASTRA expects the angles array to represent the angle of the source around the object.
    # If using 0 to 180 degrees, set endpoint=False in np.linspace to avoid a duplicate angle
    # if the 180 degree view is identical to the 0 degree view (common for half-scan).
    # If it's a full 360 scan, use 0 to 360 with `endpoint=False` if 360 is same as 0, or `endpoint=True` if you have distinct 360 view.
    
    try:
        # 1. Load Data
        projections, angles = load_specific_projection_files(
            file_pattern=params['file_pattern'],
            min_file_num=params['min_file_num'],
            max_file_num=params['max_file_num'],
            start_angle=params['start_angle'],
            end_angle=params['end_angle'],
            num_slices=params['num_slices_to_reconstruct']
        )
        
        if projections is None:
            print("[CRITICAL] Failed to load projection data. Exiting.")
            return None
        
        # 2. Preprocess Data
        projections = preprocess_projection_data(
            projections,
            apply_log_transform=params['apply_log_transform_on_load'],
            use_flat_field_correction=params['apply_flat_field_correction']
        )
        
        # 3. Setup Geometry
        proj_geom, vol_geom, reconstructed_voxel_size = setup_cone_geometry(
            projections, angles, 
            params['source_to_detector'], params['source_to_object'], 
            params['detector_pixel_size'], params['reconstructed_voxel_size']
        )
        # Store the actual reconstructed voxel size in params for metadata
        params['reconstructed_voxel_size'] = reconstructed_voxel_size

        # 4. Reconstruct Volume
        reconstruction = reconstruct_cone_beam(
            projections, angles, proj_geom, vol_geom, 
            algorithm_type=params['reconstruction_algorithm'],
            num_iterations=params['sirt_iterations']
        )
        
        # 5. Visualize Results
        visualize_reconstruction(reconstruction, projections, angles, reconstructed_voxel_size)
        
        # 6. Save Main Results (NPY and Metadata)
        save_reconstruction(reconstruction, params['output_dir'], params)

        # 7. Save All Individual Slices as Images
        # You can adjust cmap, vmin, vmax for optimal image saving.
        # Using the same vmin/vmax from visualization ensures consistent contrast.
        vmin_save = np.percentile(reconstruction, 1)
        vmax_save = np.percentile(reconstruction, 99)
        save_reconstructed_slices(reconstruction, 
                                  params['output_dir'], 
                                  file_prefix='slice_', 
                                  cmap='gray',
                                  vmin=vmin_save, 
                                  vmax=vmax_save)
        
        print("\nReconstruction process completed successfully!")
        print(f"Final reconstructed volume shape (Z, Y, X): {reconstruction.shape}")
        
        return reconstruction
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] An unhandled error occurred in main: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    start_time = time.time()
    volume = main()
    duration = time.time() - start_time
    print(f"\nTotal execution time: {duration:.2f} seconds")
    # Clean up ASTRA at the very end to free resources
    astra.clear()

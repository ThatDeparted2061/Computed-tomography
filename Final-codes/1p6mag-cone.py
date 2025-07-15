import numpy as np
import astra
import matplotlib.pyplot as plt
import os
import glob
from scipy import ndimage
import time
import re
import warnings
from scipy.signal import medfilt

def load_specific_projection_files(file_pattern, min_file_num, max_file_num,
                                   start_angle, end_angle, num_slices=None):
    """
    Loads projection data from a sequence of text files.
    Assumes files are named numerically (e.g., 2.txt, 3.txt, ..., 400.txt).
    Each file is expected to contain a 2D array where rows are slices (Z-axis)
    and columns are detector pixels (X-axis).

    Args:
        file_pattern (str): Glob pattern for projection files (e.g., "./Phantom Dataset/Al phantom/*.txt").
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
        angles = np.linspace(start_angle, end_angle, len(file_list), endpoint=True)

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

            if i % (len(file_list) // 10 + 1) == 0 or i == len(file_list) - 1:
                print(f"\rProgress: {i+1}/{len(file_list)} files loaded", end="", flush=True)

        print(f"\nFinal projection data shape: (slices, angles, detector) = {projections.shape}")
        print(f"Raw data range: min={projections.min():.2f}, max={projections.max():.2f}")

        # Optional: Display slice selection for the first loaded file
        if True:
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

    if apply_log_transform:
        print("Applying -log(I/I0) transform...")
        projections = np.maximum(projections, 1)
        I0 = 65535.0
        projections = -np.log(projections / I0)
        print(f"Post-log transform data range: min={projections.min():.4f}, max={projections.max():.4f}")
    else:
        print("Skipping -log(I/I0) transform (assuming data is already attenuation-corrected).")

    if use_flat_field_correction:
        print("Applying simple row-wise flat field correction...")
        for i in range(projections.shape[0]):
            col_means_across_angles = np.mean(projections[i], axis=0)
            col_means_across_angles[col_means_across_angles == 0] = 1e-6
            projections[i] /= col_means_across_angles
        print(f"Post flat-field data range: min={projections.min():.4f}, max={projections.max():.4f}")
    else:
        print("Skipping flat field correction.")

    print("Applying mild Gaussian denoising (sigma=0.8) to each 2D projection...")
    for i in range(projections.shape[0]):
        projections[i, :, :] = ndimage.gaussian_filter(projections[i, :, :], sigma=0.8)
    print(f"Post denoising data range: min={projections.min():.4f}, max={projections.max():.4f}")

    print("Preprocessing complete.")
    return projections

def straighten_sinogram(projections, center_finding_method='COM', display_correction=True):
    """
    Applies a straightening algorithm to the sinogram data. This corrects for
    minor tilts due to mechanical misalignment or sample wobble.

    Args:
        projections (np.ndarray): Preprocessed projection data (slices, angles, detector_width).
                                  Expected to be in attenuation values (-log(I/I0)).
        center_finding_method (str): Method to find the horizontal center of a feature.
                                     'COM' (Center of Mass) or 'PEAK' (Peak Intensity).
        display_correction (bool): If True, displays a plot of the calculated shifts and
                                   an example sinogram before and after correction.

    Returns:
        np.ndarray: Straightened projection data.
    """
    print("\n--- Applying Sinogram Straightening ---")
    num_slices, num_angles, detector_width = projections.shape
    
    shifts_per_slice = np.zeros((num_slices, num_angles))
    
    if num_slices > 5:
        slice_start_idx = max(0, num_slices // 2 - 2)
        slice_end_idx = min(num_slices, num_slices // 2 + 3)
        representative_slices = projections[slice_start_idx:slice_end_idx, :, :]
    else:
        representative_slices = projections

    calculated_centers_avg = np.zeros(num_angles)

    for i in range(num_angles):
        avg_projection_profile = np.mean(representative_slices[:, i, :], axis=0)
        
        if center_finding_method == 'COM':
            intensities = avg_projection_profile + 1e-9
            center_pos = np.sum(np.arange(detector_width) * intensities) / np.sum(intensities)
        elif center_finding_method == 'PEAK':
            center_pos = np.argmax(avg_projection_profile)
        else:
            raise ValueError(f"Unsupported center_finding_method: {center_finding_method}")
        
        calculated_centers_avg[i] = center_pos

    filtered_centers = medfilt(calculated_centers_avg, kernel_size=min(num_angles, 21) if num_angles > 1 else 1)
    ideal_center = np.mean(filtered_centers)
    shifts = ideal_center - filtered_centers
    
    print(f"Calculated average ideal center: {ideal_center:.2f} pixels")
    print(f"Minimum detected center: {filtered_centers.min():.2f}, Maximum detected center: {filtered_centers.max():.2f}")
    print(f"Max absolute shift applied: {np.max(np.abs(shifts)):.2f} pixels")

    straightened_projections = np.zeros_like(projections)

    for s_idx in range(num_slices):
        for a_idx in range(num_angles):
            straightened_projections[s_idx, a_idx, :] = ndimage.shift(
                projections[s_idx, a_idx, :], shifts[a_idx], order=1, mode='constant', cval=0
            )

    if display_correction:
        plt.figure(figsize=(15, 10))
        plt.suptitle("Sinogram Straightening Correction", fontsize=16)

        plt.subplot(2, 1, 1)
        plt.plot(np.degrees(np.linspace(0, 2*np.pi, num_angles, endpoint=True)), calculated_centers_avg, 'o', label='Raw Centers')
        plt.plot(np.degrees(np.linspace(0, 2*np.pi, num_angles, endpoint=True)), filtered_centers, '-', label='Filtered Centers (Used for Shift)')
        plt.axhline(y=ideal_center, color='r', linestyle='--', label='Ideal Center (Average)')
        plt.title('Detected Horizontal Center of Object vs. Angle')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Detector Pixel Position')
        plt.grid(True)
        plt.legend()

        example_slice_idx = num_slices // 2
        
        plt.subplot(2, 2, 3)
        plt.imshow(projections[example_slice_idx, :, :], cmap='gray', aspect='auto',
                   extent=[np.degrees(np.linspace(0, 2*np.pi, num_angles, endpoint=True))[0], 
                           np.degrees(np.linspace(0, 2*np.pi, num_angles, endpoint=True))[-1], 
                           0, detector_width], origin='lower')
        plt.title(f'Original Sinogram (Slice {example_slice_idx})')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Detector Column (X-position)')
        
        plt.subplot(2, 2, 4)
        plt.imshow(straightened_projections[example_slice_idx, :, :], cmap='gray', aspect='auto',
                   extent=[np.degrees(np.linspace(0, 2*np.pi, num_angles, endpoint=True))[0], 
                           np.degrees(np.linspace(0, 2*np.pi, num_angles, endpoint=True))[-1], 
                           0, detector_width], origin='lower')
        plt.title(f'Straightened Sinogram (Slice {example_slice_idx})')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Detector Column (X-position)')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    print("Sinogram straightening complete.")
    return straightened_projections

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
        tuple: (proj_geom (dict), vol_geom (dict), reconstructed_voxel_size)
               proj_geom: ASTRA projection geometry.
               vol_geom: ASTRA volume geometry.
               reconstructed_voxel_size: Voxel size used for reconstruction.
    """
    detector_rows, num_angles, detector_cols = projections.shape

    # Calculate derived geometric parameters
    origin_to_detector = source_to_detector - source_to_object

    # Magnification at the object plane (center of rotation)
    magnification = source_to_detector / source_to_object

    # Determine the effective voxel size and volume size
    if reconstructed_voxel_size is None:
        reconstructed_voxel_size = detector_pixel_size / magnification
        print(f"Auto-determining voxel size. Set to {reconstructed_voxel_size:.4f} mm (detector_pixel_size / magnification)")

    # Calculate volume dimensions based on the field of view covered by the detector
    vol_size_xy_mm = detector_cols * detector_pixel_size * (source_to_object / source_to_detector)
    vol_size_z_mm = detector_rows * detector_pixel_size * (source_to_object / source_to_detector)

    vol_size_x = int(np.ceil(vol_size_xy_mm / reconstructed_voxel_size))
    vol_size_y = vol_size_x
    vol_size_z = int(np.ceil(vol_size_z_mm / reconstructed_voxel_size))

    if vol_size_x % 2 != 0: vol_size_x += 1
    if vol_size_y % 2 != 0: vol_size_y += 1
    if vol_size_z % 2 != 0: vol_size_z += 1

    # Create cone beam geometry
    proj_geom = astra.create_proj_geom(
        'cone',
        detector_pixel_size,
        detector_pixel_size,
        detector_rows,
        detector_cols,
        angles,
        source_to_object,
        origin_to_detector
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

    return proj_geom, vol_geom, reconstructed_voxel_size

def reconstruct_cone_beam(projections, angles, proj_geom, vol_geom, algorithm_type='FDK_CUDA', num_iterations=50):
    """
    Reconstructs a 3D volume from cone beam projections using ASTRA.

    Args:
        projections (np.ndarray): Preprocessed projection data (slices, angles, detector_width).
        angles (np.ndarray): Projection angles in radians.
        proj_geom (dict): ASTRA projection geometry.
        vol_geom (dict): ASTRA volume geometry.
        algorithm_type (str): Reconstruction algorithm ('FDK_CUDA' or 'SIRT3D_CUDA').
        num_iterations (int): Number of iterations for iterative algorithms like SIRT.

    Returns:
        np.ndarray: Reconstructed 3D volume.
    """
    print(f"\n--- Starting Cone Beam Reconstruction ({algorithm_type}) ---")

    proj_id = astra.data3d.create('-proj3d', proj_geom, projections)
    rec_id = astra.data3d.create('-vol', vol_geom)

    if algorithm_type == 'FDK_CUDA':
        cfg = astra.astra_dict('FDK_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        cfg['option'] = {
            'ShortScan': False,
            'FilterType': 'Ram-Lak',
            'WindowType': 'Hamming'
        }
        alg_id = astra.algorithm.create(cfg)
        print("Running FDK reconstruction on GPU...")
        astra.algorithm.run(alg_id)

    elif algorithm_type == 'SIRT3D_CUDA':
        cfg = astra.astra_dict('SIRT3D_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        projector_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
        cfg['ProjectorId'] = projector_id
        cfg['option'] = {
            'MinConstraint': 0.0,
            'MaxConstraint': 1.0
        }
        alg_id = astra.algorithm.create(cfg)
        print(f"Running {algorithm_type} reconstruction for {num_iterations} iterations...")
        astra.algorithm.run(alg_id, num_iterations)
        astra.projector.delete(projector_id)

    else:
        raise ValueError(f"Unsupported algorithm type: {algorithm_type}. Choose 'FDK_CUDA' or 'SIRT3D_CUDA'.")

    reconstruction = astra.data3d.get(rec_id)
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(proj_id)
    astra.data3d.delete(rec_id)
    print("Reconstruction complete.")
    return reconstruction

def visualize_reconstruction(reconstruction, projections, angles, voxel_size_mm):
    """
    Visualizes reconstructed slices and sinograms.

    Args:
        reconstruction (np.ndarray): The 3D reconstructed volume.
        projections (np.ndarray): The original projection data.
        angles (np.ndarray): Angles in radians.
        voxel_size_mm (float): The physical size of a single voxel in mm.
    """
    detector_rows, num_angles, detector_cols = projections.shape
    vol_size_z, vol_size_y, vol_size_x = reconstruction.shape

    print("\n--- Visualizing Reconstruction ---")
    vmin_display = np.percentile(reconstruction, 1)
    vmax_display = np.percentile(reconstruction, 99)
    print(f"Display range for reconstruction: [{vmin_display:.4f}, {vmax_display:.4f}]")

    plt.figure(figsize=(16, 9))
    plt.suptitle("Reconstructed Axial Slices (Z-Planes)", y=1.02, fontsize=16)
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
        plt.gca().set_aspect('equal', adjustable='box')
        plt.colorbar(label='Attenuation (a.u.)', fraction=0.046, pad=0.04)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    plt.figure(figsize=(18, 6))
    plt.suptitle("Orthogonal Views of Reconstructed Volume", y=1.02, fontsize=16)
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

    plt.subplot(1, 3, 2)
    plt.imshow(reconstruction[:, vol_size_y // 2, :], cmap='gray',
               vmin=vmin_display, vmax=vmax_display,
               aspect='auto',
               extent=[-vol_size_x/2*voxel_size_mm, vol_size_x/2*voxel_size_mm,
                       vol_size_z/2*voxel_size_mm, -vol_size_z/2*voxel_size_mm])
    plt.title(f'Central Coronal Slice (Y={vol_size_y//2})')
    plt.xlabel('X (mm)')
    plt.ylabel('Z (mm)')
    plt.colorbar(label='Attenuation (a.u.)', fraction=0.046, pad=0.04)

    plt.subplot(1, 3, 3)
    plt.imshow(reconstruction[:, :, vol_size_x // 2], cmap='gray',
               vmin=vmin_display, vmax=vmax_display,
               aspect='auto',
               extent=[-vol_size_y/2*voxel_size_mm, vol_size_y/2*voxel_size_mm,
                       vol_size_z/2*voxel_size_mm, -vol_size_z/2*voxel_size_mm])
    plt.title(f'Central Sagittal Slice (X={vol_size_x//2})')
    plt.xlabel('Y (mm)')
    plt.ylabel('Z (mm)')
    plt.colorbar(label='Attenuation (a.u.)', fraction=0.046, pad=0.04)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    plt.figure(figsize=(18, 6))
    plt.suptitle("Example Sinograms at Different Detector Row Z-Positions", y=1.02, fontsize=16)
    det_rows_to_show = [detector_rows // 4, detector_rows // 2, 3 * detector_rows // 4]
    for i, det_row_idx in enumerate(det_rows_to_show):
        plt.subplot(1, 3, i + 1)
        sino = projections[det_row_idx, :, :]
        plt.imshow(sino, cmap='gray', aspect='auto',
                   extent=[np.degrees(angles[0]), np.degrees(angles[-1]), 0, detector_cols],
                   origin='lower')
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

    npy_path = os.path.join(output_dir, 'reconstruction.npy')
    np.save(npy_path, reconstruction)
    print(f"\nReconstruction saved as NumPy array: {npy_path}")

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
        file_prefix (str): Prefix for the slice filenames.
        cmap (str): Colormap to use for saving images.
        vmin (float, optional): Minimum value for colormap scaling.
        vmax (float, optional): Maximum value for colormap scaling.
    """
    slice_output_dir = os.path.join(output_dir, "reconstructed_slices")
    if not os.path.exists(slice_output_dir):
        os.makedirs(slice_output_dir)
        print(f"\nCreated directory for slices: {slice_output_dir}")
    else:
        print(f"\nSaving slices to existing directory: {slice_output_dir}")

    num_slices = reconstruction.shape[0]
    if vmin is None:
        vmin = np.percentile(reconstruction, 1)
    if vmax is None:
        vmax = np.percentile(reconstruction, 99)
    print(f"Saving slices with display range: [{vmin:.4f}, {vmax:.4f}]")

    print(f"Saving {num_slices} individual slices...")
    for i in range(num_slices):
        slice_data = reconstruction[i, :, :]
        filename = os.path.join(slice_output_dir, f"{file_prefix}{i:04d}.png")
        plt.imsave(filename, slice_data, cmap=cmap, vmin=vmin, vmax=vmax)
        if (i + 1) % (num_slices // 10 + 1) == 0 or i == num_slices - 1:
            print(f"\rProgress: {i+1}/{num_slices} slices saved", end="", flush=True)
    print("\nAll slices saved.")

def main():
    print("=" * 70)
    print("        IMPROVED CONE BEAM CT RECONSTRUCTION        ")
    print("=" * 70)

    # --- Configuration Parameters ---
    params = {
        'file_pattern': "./Phantom Dataset/Al phantom/*.txt",
        'min_file_num': 1,
        'max_file_num': 360,
        'num_slices_to_reconstruct': 181,
        'start_angle': 0,
        'end_angle': 360,
        'source_to_detector': 800.0,
        'source_to_object': 488.0,  # Adjusted to place source below object
        'detector_pixel_size': 0.200,
        'reconstructed_voxel_size': None,
        'output_dir': 'cone_beam_reconstruction_results',
        'apply_log_transform_on_load': True,
        'apply_flat_field_correction': False,
        'apply_sinogram_straightening': True,
        'straightening_method': 'COM',
        'reconstruction_algorithm': 'FDK_CUDA',
        'sirt_iterations': 50
    }

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
            return

        # 2. Preprocess Data
        projections = preprocess_projection_data(
            projections,
            apply_log_transform=params['apply_log_transform_on_load'],
            use_flat_field_correction=params['apply_flat_field_correction']
        )

        # 3. Apply Sinogram Straightening
        if params['apply_sinogram_straightening']:
            projections = straighten_sinogram(projections, center_finding_method=params['straightening_method'])

        # 4. Setup Geometry
        proj_geom, vol_geom, reconstructed_voxel_size = setup_cone_geometry(
            projections, angles,
            params['source_to_detector'], params['source_to_object'],
            params['detector_pixel_size'], params['reconstructed_voxel_size']
        )
        params['reconstructed_voxel_size'] = reconstructed_voxel_size

        # 5. Reconstruct Volume
        reconstruction = reconstruct_cone_beam(
            projections, angles, proj_geom, vol_geom,
            algorithm_type=params['reconstruction_algorithm'],
            num_iterations=params['sirt_iterations']
        )

        # 6. Visualize Reconstruction
        if reconstruction is not None:
            visualize_reconstruction(reconstruction, projections, angles, params['reconstructed_voxel_size'])

            # 7. Save Reconstruction Results
            save_reconstruction(reconstruction, params['output_dir'], params)
            save_reconstructed_slices(reconstruction, params['output_dir'])

    except Exception as e:
        print(f"\n[CRITICAL] An unhandled error occurred in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
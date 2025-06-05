import astra
import numpy as np
import matplotlib.pyplot as plt

# =================================================================
# 1. Create a 3D Phantom (Larger Sphere)
# =================================================================
size = 128  # Increased from 64 for better visibility
phantom = np.zeros((size, size, size), dtype=np.float32)

# Generate coordinates
x = np.linspace(-1, 1, size)
y = np.linspace(-1, 1, size)
z = np.linspace(-1, 1, size)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Larger sphere (radius = 0.6)
phantom[(X**2 + Y**2 + Z**2) < 0.6**2] = 1.0

# =================================================================
# 2. Define Geometry (Adjusted for Scale)
# =================================================================
vol_geom = astra.create_vol_geom(size, size, size)

# Cone-beam projection geometry
num_angles = 180
angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

# Smaller detector (128x128) to match volume scale
proj_geom = astra.create_proj_geom(
    'cone',
    1.0, 1.0,      # Detector spacing (x, y)
    128, 128,      # Detector size (cols, rows)
    angles,
    1000.0,        # Source-to-origin distance
    1200.0         # Source-to-detector distance
)

# =================================================================
# 3. Run Forward Projection (GPU)
# =================================================================
sinogram_id = astra.data3d.create('-sino', proj_geom)
vol_id = astra.data3d.create('-vol', vol_geom, phantom)

cfg = astra.astra_dict('FP3D_CUDA')
cfg['ProjectionDataId'] = sinogram_id
cfg['VolumeDataId'] = vol_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)

# Extract sinogram
sinogram = astra.data3d.get(sinogram_id)

# =================================================================
# 4. Visualize Correct Slice (Angle 90, All Detector Rows)
# =================================================================
plt.imshow(sinogram[:, 90, :], cmap='gray', vmin=0, vmax=np.max(sinogram))
plt.title("Cone-Beam Projection (Angle 90)")
plt.colorbar()
plt.show()

# =================================================================
# 5. Cleanup
# =================================================================
astra.algorithm.delete(alg_id)
astra.data3d.delete(sinogram_id)
astra.data3d.delete(vol_id)

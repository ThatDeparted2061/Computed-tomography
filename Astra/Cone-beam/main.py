import astra
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create a 3D phantom (simple sphere)
size = 64
phantom = np.zeros((size, size, size), dtype=np.float32)
x = np.linspace(-1, 1, size)
y = np.linspace(-1, 1, size)
z = np.linspace(-1, 1, size)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
phantom[(X**2 + Y**2 + Z**2) < 0.4**2] = 1.0  # spherical phantom

# Step 2: Volume geometry
vol_geom = astra.create_vol_geom(size, size, size)

# Step 3: Define cone-beam projection geometry
num_angles = 180
angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

proj_geom = astra.create_proj_geom(
    'cone',
    1.0,           # detector spacing horizontal
    1.0,           # detector spacing vertical
    256,           # number of detector cols
    256,           # number of detector rows
    angles,
    1000.0,        # Source to origin
    1200.0         # Source to detector
)

# Step 4: Create sinogram data object
sinogram_id = astra.data3d.create('-sino', proj_geom)

# Step 5: Create volume data object
vol_id = astra.data3d.create('-vol', vol_geom, phantom)

# Step 6: Set up forward projection (GPU required for cone-beam)
cfg = astra.astra_dict('FP3D')  # CUDA GPU-based 3D forward projector
cfg['ProjectionDataId'] = sinogram_id
cfg['VolumeDataId'] = vol_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)

# Step 7: Get the sinogram (3D array)
sinogram = astra.data3d.get(sinogram_id)

# Cleanup
astra.algorithm.delete(alg_id)
astra.data3d.delete(sinogram_id)
astra.data3d.delete(vol_id)

# Step 8: Show one slice of sinogram (angle 90)
plt.imshow(sinogram[90], cmap='gray')
plt.title("Cone-beam Sinogram Slice (angle 90)")
plt.colorbar()
plt.show()

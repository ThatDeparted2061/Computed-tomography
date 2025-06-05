import astra
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create a circular phantom
phantom = np.zeros((128, 128), dtype=np.float32)
xx, yy = np.meshgrid(np.linspace(-1, 1, 128), np.linspace(-1, 1, 128))
phantom[(xx**2 + yy**2) < 0.5**2] = 1.0

# Step 2: Volume geometry
vol_geom = astra.create_vol_geom(128, 128)

# Step 3: Fan-beam projection geometry
angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
proj_geom = astra.create_proj_geom(
    'fanflat',     # projection type
    1.0,           # detector spacing
    200,           # number of detectors
    angles,
    100.0,         # source-to-object distance
    200.0          # source-to-detector distance
)

# Step 4: Create 2D data object for projection
proj_id = astra.data2d.create('-sino', proj_geom)

# Step 5: Create 2D data object for phantom
vol_id = astra.data2d.create('-vol', vol_geom, phantom)

# Step 6: Set up configuration for CPU forward projection
cfg = astra.astra_dict('FP')
cfg['ProjectionDataId'] = proj_id
cfg['VolumeDataId'] = vol_id
cfg['ProjectorId'] = astra.create_projector('line_fanflat', proj_geom, vol_geom)

# Step 7: Run the algorithm
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)

# Step 8: Retrieve the sinogram
sinogram = astra.data2d.get(proj_id)

# Cleanup
astra.algorithm.delete(alg_id)
astra.data2d.delete(proj_id)
astra.data2d.delete(vol_id)
astra.projector.delete(cfg['ProjectorId'])

# Step 9: Plot phantom and sinogram
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(phantom, cmap='gray')
ax[0].set_title("Phantom")
ax[1].imshow(sinogram, cmap='gray', aspect='auto')
ax[1].set_title("Fan-beam Sinogram")
plt.tight_layout()
plt.show()

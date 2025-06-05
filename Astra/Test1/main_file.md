## A simple demonstration for astra-toolkit
- Creates a 2D phantom (a 128×128 image of ones),
- Computes the sinogram (forward projection),
- Reconstructs the image using FBP (Filtered Back Projection),
- Plots the original, sinogram, and reconstructed image.

  ```py
  import astra
  import numpy as np
  import matplotlib.pyplot as plt

  phantom = np.ones((128, 128), dtype=np.float32)

  vol_geom = astra.create_vol_geom(128, 128)
  angles = np.linspace(0, np.pi, 180, endpoint=False)
  proj_geom = astra.create_proj_geom('parallel', 1.0, 128, angles)

  projector_id = astra.create_projector('linear', proj_geom, vol_geom)

  sino_id, sinogram = astra.create_sino(phantom, projector_id)

  rec_id = astra.data2d.create('-vol', vol_geom)
  cfg = astra.astra_dict('FBP')
  cfg['ReconstructionDataId'] = rec_id
  cfg['ProjectionDataId'] = sino_id
  cfg['ProjectorId'] = projector_id

  alg_id = astra.algorithm.create(cfg)
  astra.algorithm.run(alg_id)

  reconstruction = astra.data2d.get(rec_id)

  astra.algorithm.delete(alg_id)
  astra.data2d.delete(rec_id)
  astra.data2d.delete(sino_id)
  astra.projector.delete(projector_id)

  fig, axes = plt.subplots(1, 3, figsize=(15, 5))
  axes[0].imshow(phantom, cmap='gray')
  axes[0].set_title('Original Phantom')
  axes[1].imshow(sinogram, cmap='gray', aspect='auto')
  axes[1].set_title('Sinogram')
  axes[2].imshow(reconstruction, cmap='gray')
  axes[2].set_title('Reconstructed (FBP)')
  plt.show()
  ```

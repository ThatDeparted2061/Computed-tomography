# DETAILED STEPS PER TYPE
## STEP 1: Define Volume Geometry
```python

vol_geom = astra.create_vol_geom(Nx, Ny)  # for 2D
# or
vol_geom = astra.create_vol_geom(Nx, Ny, Nz)  # for 3D
```
## STEP 2: Define Projection Geometry
### Parallel Beam (2D)
```python
proj_geom = astra.create_proj_geom('parallel', detector_spacing, num_detectors, angles)
```
### Fan Beam (2D)
```python
proj_geom = astra.create_proj_geom('fanflat', detector_spacing, num_detectors, angles, source_origin_dist, origin_det_dist)
```
### Cone Beam (3D)
```python
proj_geom = astra.create_proj_geom('cone', detector_spacing_y, detector_spacing_x,
                                   num_det_rows, num_det_cols, angles,
                                   source_origin_dist, origin_det_dist)
```
Note: for 3D cone beam, use astra.create_proj_geom('cone', ...) and ensure GPU support if needed (cuda backend).

## STEP 3: Create Sinogram (Forward Projection)
```python

sino_id, sinogram = astra.create_sino(volume, proj_geom, proj_type='line')
#For GPU:

sino_id, sinogram = astra.create_sino3d_gpu(volume, proj_geom)
```
## STEP 4: Reconstruction
```python

cfg = astra.astra_dict('FBP')  # or 'SIRT', 'CGLS', etc.
cfg['ProjectionDataId'] = sino_id
cfg['ReconstructionDataId'] = rec_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)
```
## STEP 5: Cleanup & Visualization
```python
astra.data2d.delete([rec_id, sino_id])
astra.algorithm.delete(alg_id)

plt.imshow(reconstruction, cmap='gray')
```

# Example 1: Visualize Registration Results

This demo showcases the comprehensive visualization of registration results comparing different methods.  
Features includes:
1. Deformed images
2. Difference maps (deformed - fixed)
3. Deformed segmentation labels
4. Deformation/Displacement field using 2D warped grids
5. Jacobian Determinants (folding/Jdet<=0 contoured in black)

## Usage
Just run visualize_all_results.ipynb to reproduce the following results.   
Example OASIS data and predicted displacements are stored in ./data


## Results Gallery

Registration results across all three views:

### Sagittal Plane Results (axis_to_slice=0)
![Sagittal Plane Registration Result](results/f0004_m0236_axis0_slc071of160.png)
*Registration result in sagittal plane (slice 71 of 160)*

### Coronal Plane Results  (axis_to_slice=1)
![Coronal Plane Registration Result](results/f0004_m0236_axis1_slc113of224.png)
*Registration result in coronal plane (slice 113 of 224)*

### Axial Plane Results (axis_to_slice=2)
![Axial Plane Registration Result](results/f0004_m0236_axis2_slc097of192.png)
*Registration result in axial plane (slice 97 of 192)*


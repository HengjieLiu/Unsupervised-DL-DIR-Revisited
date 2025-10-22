# Unsupervised-DL-DIR-Revisited
Unsupervised Deformable Image Registration Revisited: Enhancing Performance through Registration-Specific Designs (accepted by MIDL 2025 short paper track)  
[![OpenReview](https://img.shields.io/badge/OpenReview-DGvFGbX0EG-8C1B13?logo=openreview&logoColor=white)](https://openreview.net/forum?id=DGvFGbX0EG#discussion)

Keywords: Deformable Image Registration, Unsupervised Learning, Benchmark, Ablation Study

## TL;DR <img src="https://raw.githubusercontent.com/iampavangandhi/iampavangandhi/master/gifs/Hi.gif" width="30">
We highlight the value of three registration-specific designs — **multi-resolution pyramids, local correlation calculation, and inverse consistency constraints** — in unsupervised mono-modal deformable image registration. **By relying solely on these principles, simple network architectures (i.e. without large convolutional kernels, attention gates, or feature-level self- or cross-attention) can achieve competitive results, and potentially with far less training data**.

## New features and updates <img src="https://raw.githubusercontent.com/iampavangandhi/iampavangandhi/master/gifs/Hi.gif" width="30">
07/08/2025 **Finalized the repository**  
05/05/2025 **We updated the [validation results](#lumir-validation-results) on the 2024 Learn2Reg LUMIR Challenge**  
05/01/2025 **Our paper got accepted by MIDL 2025 short paper track**  
10/21/2025 **Add visualization demo (🔥 SUGGESTIONS ARE WELCOME 🔥)** 

## Progress
- [x] Upload basic code
- [x] Upload config files
- [x] Training scripts
- [x] Dataset split information
- [x] Finalize README.md
- [x] Add visualization demo

## Planned Enhancements
> **Note:** The items below are planned enhancements. They may be delayed or even skipped, depending on available time.
- [ ] Refactor and clean up codebase
- [ ] Optimize DP-ConvIC-C: The current implementation of DP-ConvIC-C calculates correlation both ways separately, which is redundant.

## Acknowledgement
This work is largely inspired by:
- [rethink-reg](https://github.com/BailiangJ/rethink-reg)
- [Magic-or-Mirage](https://github.com/rohitrango/Magic-or-Mirage)
- [VFA](https://github.com/yihao6/vfa/tree/main)
- [SITReg](https://github.com/honkamj/SITReg)

We also thank the following repositories for providing helpful code and data resources:
- [VoxelMorph](https://github.com/voxelmorph/voxelmorph)
- [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration)
- [LUMIR](https://github.com/JHU-MedImage-Reg/LUMIR_L2R)
- [L2R](https://github.com/MDL-UzL/L2R)
- [deformation-inversion-layer](https://github.com/honkamj/deformation-inversion-layer)
- [composable-mapping](https://github.com/honkamj/composable-mapping)

## Contact

Contributions and feedback are welcome! Please open an issue or submit a pull request. For direct inquiries, you can also reach me at <hjliu@g.ucla.edu>.

## Methods
![Figure 1: Network backbone for controlled registration experiments. The dual-stream pyramid (DP) architecture5 is adopted with the same feature encoder and different variations of deformation decoder (DD) block](images/method_figure1_overview.png)
#### Figure 1 – Network backbone for controlled registration experiments. The dual-stream pyramid (DP) architecture5 is adopted with the same feature encoder and different variations of deformation decoder (DD) block.

#### Table 1 – Summary of the proposed variants and comparing methods

![Table 1: Summary of the proposed variants and comparing methods](images/method_table1_summary.png)

## Results
<a name="lumir-validation-results"></a>
**1-1 LUMIR Validation Results**

| Model                     | Dice↑             | TRE↓ (mm) | NDV↓ (%)   | HdDist95↓ |
|---------------------------|-------------------|-----------|------------|-----------|
| **Official baselines**    |
| VoxelMorph<sup>1</sup>    | 0.7186 ± 0.0340   | 3.1545    | 1.1836     | 3.9821    |
| TransMorph<sup>1</sup>    | 0.7594 ± 0.0319   | 2.4225    | 0.3509     | 3.5074    |
| VFA<sup>1</sup>           | 0.7726 ± 0.0286   | 2.4949    | 0.0788     | 3.2127    |
| **Challenge-winning**     |
| SITReg-v1<sup>2</sup>     | 0.7742 ± 0.0291   | 2.3112    | 0.0231     | 3.3039    |
| SITReg-v2<sup>2</sup>     | 0.7805 ± 0.0287   | 2.3005    | 0.0025     | 3.1187    |
| **Our re-trained baselines**<sup>3</sup> |
| VFA (Ours)                | 0.7734 ± 0.0286   | 2.4739    | 0.1051     | 3.2063    |
| SITReg-v1 (Ours)          | 0.7727 ± 0.0284   | 2.3120    | 0.0308     | 3.3319    |
| **Our variants**<sup>3</sup> |
| (a) DP-Conv-MF            | 0.7713 ± 0.0290   | 2.4676    | 0.4158     | 3.3534    |
| (a) DP-Conv-MFC           | 0.7730 ± 0.0291   | 2.4449    | 0.4672     | 3.3566    |
| (a) DP-Conv-C             | 0.7747 ± 0.0295   | 2.4135    | 0.3795     | 3.3666    |
| (b) DP-ConvIC-MF          | 0.7717 ± 0.0288   | 2.3660    | 0.0310     | 3.3489    |
| (b) DP-ConvIC-C           | 0.7724 ± 0.0288   | 2.3357    | 0.0309     | 3.3873    |
| (c) DP-VFA                | 0.7764 ± 0.0284   | 2.4420    | 0.0540     | 3.2157    |

<sup>1</sup> VoxelMorph, TransMorph and VFA are the official baselines. The results are obtained from [Learn2reg 2024 LUMIR](https://github.com/JHU-MedImage-Reg/LUMIR_L2R).  
<sup>2</sup> SITReg is the challenge-winning method. The results are obtained from the authors' challenge presentation.  
&emsp; SITReg-v1: Vanilla version trained with NCC loss + diffusion loss.  
&emsp; SITReg-v2: Final version further trained with group consistency loss + NDV loss (not used by other methods in this table).  
<sup>3</sup> All of “Ours” (re-trained baselines & variants) were run with the same random seed for the dataloader and identical learning-rate scheduling.

**1-2 LUMIR Validation Results using (b) DP-ConvIC-C with different training set sizes**  
Note: In theory, for inter-subject registration with N subjects, N*(N-1) pairs are available for training.

| Training set size | Dice↑               | HD95↓    | TRE (mm)↓   | NDV (%)↓   |
|-------------------|---------------------|----------|-------------|------------|
| 3,384             | 0.7724 ± 0.0288     | 3.3873   | 2.3357      | 0.0309     |
| 1,000             | **0.7726 ± 0.0283** | 3.3772   | **2.3088**  | 0.0332     |
| 300               | 0.7725 ± 0.0287     | 3.3756   | 2.3208      | **0.0303** |
| 100               | 0.7717 ± 0.0280     | 3.3966   | 2.3305      | 0.0319     |
| 50                | 0.7705 ± 0.0286     | 3.4162   | 2.3479      | 0.0304     |
| 30                | 0.7690 ± 0.0287     | 3.4576   | 2.3723      | 0.0328     |
| 20                | 0.7674 ± 0.0288     | 3.4521   | 2.4053      | 0.0358     |
| 10                | 0.7642 ± 0.0308     | 3.5216   | 2.3771      | 0.0361     |

**2 OASIS test set (200 pairs)**  
Metrics in **bold** denote the best-performing methods, while those <ins>underlined</ins> are competitively close.

| Method                  | Dice ↑                             | HD95 ↓                              | SDlogJ (×100) ↓                         | NDV (%) ↓                         |
|-------------------------|------------------------------------|-------------------------------------|-----------------------------------------|-----------------------------------|
| Initial                 | 0.5759 (0.0682)                    | 3.95 (0.95)                         | –                                       | –                                 |
| Greedy                  | 0.8068 (0.0297)                    | 2.02 (0.56)                         | **13.18 (0.95)**                        | **0.0007**                        |
| VoxelMorph              | 0.7647 (0.0392)                    | 2.55 (0.72)                         | 21.96 (2.77)                            | 1.27                              |
| TransMorph              | 0.7934 (0.0276)                    | 2.15 (0.56)                         | 17.00 (1.79)                            | 0.83                              |
| VFA                     | <ins>0.8203 (0.0233)</ins>         | 1.87 (0.45)                         | <ins>14.00 (0.89)</ins>                 | 0.065                             |
| SITReg-v1 (IC)          | <ins>0.8230 (0.0232)</ins>         | <ins>1.81 (0.45)</ins>              | **12.98 (1.00)**                        | <ins>0.027</ins>                  |
| (a) DP-Conv-MF          | <ins>0.8237 (0.0237)</ins>         | <ins>1.82 (0.46)</ins>              | 15.32 (1.02)                            | 0.33                              |
| (a) DP-Conv-MFC         | **0.8281 (0.0227)**                | **1.79 (0.45)**                     | 15.63 (1.12)                            | 0.37                              |
| (a) DP-Conv-C           | **0.8283 (0.0226)**                | **1.79 (0.46)**                     | <ins>14.64 (0.98)</ins>                 | 0.33                              |
| (b) DP-ConvIC-MF        | <ins>0.8223 (0.0244)</ins>         | <ins>1.82 (0.47)</ins>              | **12.98 (0.98)**                        | <ins>0.027</ins>                  |
| (b) DP-ConvIC-C         | <ins>0.8244 (0.0225)</ins>         | <ins>1.80 (0.45)</ins>              | **12.79 (1.01)**                        | <ins>0.028</ins>                  |
| (c) DP-VFA              | <ins>0.8199 (0.0237)</ins>         | 1.87 (0.46)                         | <ins>13.91 (0.94)</ins>                 | <ins>0.031</ins>                  |

### OASIS results visualization
#### Sagittal
![Registration Result - Axis 0](./demo/example1_visualize_registration_results/results/f0004_m0236_axis0_slc071of160.png)

#### Coronal
![Registration Result - Axis 1](./demo/example1_visualize_registration_results/results/f0004_m0236_axis1_slc113of224.png)

#### Axial
![Registration Result - Axis 2](./demo/example1_visualize_registration_results/results/f0004_m0236_axis2_slc097of192.png)
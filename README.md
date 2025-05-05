# Unsupervised-DL-DIR-Revisited
Unsupervised Deformable Image Registration Revisited: Enhancing Performance through Registration-Specific Designs (accepted by MIDL 2025 short paper track)  
[![OpenReview](https://img.shields.io/badge/OpenReview-DGvFGbX0EG-8C1B13?logo=openreview&logoColor=white)](https://openreview.net/forum?id=DGvFGbX0EG#discussion)


## TL;DR <img src="https://raw.githubusercontent.com/iampavangandhi/iampavangandhi/master/gifs/Hi.gif" width="30">
We highlight the value of incorporating registration-specific designs—such as multi-resolution pyramids, local correlation, and inverse consistency constraints—for unsupervised deformable image registration. With these designs, simple network architectures can achieve competitive performance.

## New features and updates <img src="https://raw.githubusercontent.com/iampavangandhi/iampavangandhi/master/gifs/Hi.gif" width="30">
05/05/2025 **We updated the [validation results](#lumir-validation-results) on the 2024 Learn2Reg LUMIR Challenge**  
05/01/2025 **Our paper got accepted by MIDL 2025 short paper track**

## Progress
- [x] Upload basic code (cleanup: todo)
- [x] Upload config files (cleanup: todo)
- [ ] Training and testing scripts
- [ ] Dataset split information
- [ ] Finalize README.md

## To do
- [ ] The current implementation of DP-ConvIC-C calculates correlation both ways separately, which is not optimal



# Acknowledgement
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

# Extended results
<a name="lumir-validation-results"></a>
**LUMIR Validation Results**

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

<sup>1</sup> VoxelMorph, TransMorph and VFA are the official baselines. The results are obtained from [LUMIR](https://github.com/JHU-MedImage-Reg/LUMIR_L2R).  
<sup>2</sup> SITReg is the challenge-winning method. The results are obtained from the authors' challenge presentation.  
&emsp; SITReg-v1: Vanilla version trained with NCC loss + diffusion loss.  
&emsp; SITReg-v2: Final version further trained with group consistency loss + NDV loss (not used by other methods in this table).  
<sup>3</sup> All of “Ours” (re-trained baselines & variants) were run with the same random seed for the dataloader and identical learning-rate scheduling.

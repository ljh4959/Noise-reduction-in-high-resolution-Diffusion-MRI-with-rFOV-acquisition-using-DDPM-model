# Noise Reduction in High-Resolution Diffusion MRI with Reduced FOV Acquisition Using Diffusion Probabilistic Models

## Overview
The Reduced Field-of-View (rFOV) MRI technique enables high-resolution imaging of Diffusion Magnetic Resonance Imaging (dMRI) by limiting signal acquisition to a pre-selected area of interest. In this study, we explore and present a Denoising Diffusion Probabilistic Model
(DDPM)-based noise-reduction approach that can be integrated with the rFOV technique to enable imaging of intricate and small tissue structures. Our proposed approach outperforms vanilla DDPM and previous approaches (MP-PCA, supervised learning, and Patch2Self), achieving state-of-the-art performance. Our results demonstrate improved delineation
of small brain structures and improved ADC maps, which were previously challenging to image. This innovative noise reduction method provides a promising solution for improving the quality of high-resolution dMRI.

## Process
<p align="center">
  <img src="https://i.ibb.co/TM5py0B/Fig1.png" />
</p>

1. We investigate an unconditioned DDPM model that can work for various
noise levels, and achieve superior performance compared to the vanilla
DDPM model.
2. We explore the use of synthetic image training as a means of addressing the
lack of high-quality public datasets.
3. We examine the effectiveness of performing the noise reduction operation in
the complex domain.
4. We validate our approach by applying it to actual images including a subject
acquired with an untrained orientation. Our results demonstrate improved
delineation of small brain structures, as well as improved ADC maps, which
were difficult to image using previous methods.

* Stage 1: Training from synthetically improved image
* Stage 2: Using noise estimator for determining the start point of Marchov Chain
* Stage 3: Operating in the complex domain

## Results
#### Ablation study
To evaluate the effectiveness of the proposed modules (Stage 1-3), we conducted an ablation study. Specifically, we considered the following variants: (**Base**: vanilla DDPM, **S**: synthesizing restored reference images using LRMA, **NE**: noise estimator, **C**: complex diffusion sampling process). Note that the entire pipeline is identical except for the inclusion of the respective stages. For testing, the noise was injected at a level of 5% of the image’s standard deviation.

<p align="center">
  <img src="https://i.ibb.co/7r85BBC/Fig2.png" />
</p>

#### Comparison study
Representative DWI images (b=0 and b=1000 sec/mm<sup>2</sup>) and their corresponding ADC maps are shown (synthesized noise level = 15%). Unlike conventional methods, the proposed approach can not only effectively remove the noise, but also well preserve the tissue structure details (as indicated by the red arrows). ADC maps showed the same trend.

<p align="center">
  <img src="https://github.com/ljh4959/Noise-reduction-in-high-resolution-Diffusion-MRI-with-rFOV-acquisition-using-DDPM-model/assets/59819627/106cd09d-e8d0-48ea-a2e4-60abde40e11d" />
</p>

<p align="center">
  <img src="https://github.com/ljh4959/Noise-reduction-in-high-resolution-Diffusion-MRI-with-rFOV-acquisition-using-DDPM-model/assets/59819627/971bdb39-741c-4543-b640-603f829d2858" />
</p>

#### Untrained dataset
Additionally, the proposed method generalizes well to untrained orientations. The approach can be further extended to other DWI techniques such as DTI and HARDI, where more diffusion directions are typically acquired. Noise-reduced images exhibit mild blurring, likely due to inadequate SNR in the dataset. A potential remedy could be acquiring a high-quality dataset with more signal averaging.

<p align="center">
  <img src="https://github.com/ljh4959/Noise-reduction-in-high-resolution-Diffusion-MRI-with-rFOV-acquisition-using-DDPM-model/assets/59819627/89b737e0-1f45-4062-9f4e-618d85cea1e4" />
</p>

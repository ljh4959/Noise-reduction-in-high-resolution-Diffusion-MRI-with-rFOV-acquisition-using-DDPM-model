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
To evaluate the effectiveness of the proposed modules (Stage 1-3), we conducted an ablation study. Specifically, we considered the following variants: (Base: vanilla DDPM, S: synthesizing restored reference images using LRMA, NE: noise estimator, C: complex diffusion sampling process). Note that the entire pipeline is identical except for the inclusion of the respective stages. For testing, the noise was injected at a level of 5% of the image’s standard deviation.

<p align="center">
  <img src="https://i.ibb.co/7r85BBC/Fig2.png" />
</p>

#### Comparison study
Representative DWI images (b=0 and b=1000 sec/mm<sup>2) and their corresponding ADC maps are shown (synthesized noise level = 15%). Unlike conventional methods, the proposed approach can not only effectively remove the noise, but also well preserve the tissue structure details (as indicated by the red arrows). ADC maps showed the same trend.

<p align="center">
  <img src="https://i.ibb.co/Tcyfx0P/Fig3.png" />
</p>



Training was performed on two datasets:
* [smithsonian-butterflies-subset](https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset) by HuggingFace
* [croupier-mtg-dataset](https://huggingface.co/datasets/alkzar90/croupier-mtg-dataset) by [alcazar90](https://github.com/alcazar90/croupier-mtg-dataset)

### 128×128 resolution
All 128×128 models were trained for 300 epochs with cosine annealing with initial learning rate set to 2e-4, batch size 6 and 1000 diffusion timesteps. 
#### Training on smithsonian-butterflies-subset
300 epochs, 50266 steps
Epoch 4             |  Epoch 99
:-------------------------:|:-------------------------:
![0004](https://user-images.githubusercontent.com/8377365/189268993-97f7b8be-4ab0-4cc9-af46-87582bfad1b4.png)  |  ![0099](https://user-images.githubusercontent.com/8377365/189269009-49ee9a26-7c63-4bdf-b4d1-79d1e034cc12.png)
Epoch 204             |  Epoch 300
![0204](https://user-images.githubusercontent.com/8377365/189269020-c37756b6-0518-4b90-8d28-64d40eaedc0e.png)  |   ![0300](https://user-images.githubusercontent.com/8377365/189269042-19ac4e36-92c7-4141-b43a-d1b405ad108e.png)
Sampling from the epoch=300 | Sampling from the epoch=300
![diffusion1](https://user-images.githubusercontent.com/8377365/189269282-82b13b7b-eb6b-4746-8a63-6c6b06f40ebc.gif)  |   ![diffusion2](https://user-images.githubusercontent.com/8377365/189269404-59f205c5-95ff-4b4e-9447-e68409f61f9e.gif)

#### Training on croupier-mtg-dataset
300 epoch, 72599 steps
Epoch 4             |  Epoch 99
:-------------------------:|:-------------------------:
![0004](https://user-images.githubusercontent.com/8377365/189183793-c3da77ab-f306-4a94-bd5e-df500bfe3465.png)  |  ![0099](https://user-images.githubusercontent.com/8377365/189183825-37028de4-030b-4471-88e8-2d17094cec8a.png)
Epoch 204             |  Epoch 300
![0204](https://user-images.githubusercontent.com/8377365/189183859-d70a572f-1027-4af5-948b-057c042ab508.png)  |  ![0300](https://user-images.githubusercontent.com/8377365/189183877-63a705da-1489-497f-9d8a-c8be9bdf0bdf.png)
Sampling from the epoch=300 | Sampling from the epoch=300
![diffusion1](https://user-images.githubusercontent.com/8377365/189268712-2cb1fd0c-b566-4058-893b-cbba2d949eb2.gif)   |   ![diffusion2](https://user-images.githubusercontent.com/8377365/189268713-da20e5fc-9ce5-45ce-977c-d6e1db35e090.gif)

### 256×256 resolution
All 256×256 models were trained for 300 epochs with cosine annealing with initial learning rate set to 2e-5, batch size 6 and 1000 diffusion timesteps.
#### Training on smithsonian-butterflies-subset
300 epochs, 50266 steps
Epoch 4             |  Epoch 100
:-------------------------:|:-------------------------:
![0004](https://user-images.githubusercontent.com/8377365/189496165-84d677b0-8b13-4eb1-a6d6-09879db11fc1.png)  |  ![0100](https://user-images.githubusercontent.com/8377365/189496166-286aedb5-7b5e-4317-9cab-cd0cf94487b3.png)
Epoch 205           |  Epoch 300
![0205](https://user-images.githubusercontent.com/8377365/189496168-58933a78-f276-4d6b-8ee3-8a5d94db7b9d.png)  |  ![0300](https://user-images.githubusercontent.com/8377365/189496174-2bfc2d75-6e0e-493b-ab8f-215c25a5175a.png)

[TODO description]

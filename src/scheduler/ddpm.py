import torch
import numpy as np

from tqdm import tqdm
import math

from src.utils.common import broadcast, LR_denoising
from src.utils.tranforms import fft2c, ifft2c


class DDPMPipeline:
    def __init__(self, beta_start=1e-4, beta_end=1e-2, num_timesteps=1000):
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas

        # alpha-hat in the paper, precompute them
        self.alphas_hat = torch.cumprod(self.alphas, dim=0)

        self.num_timesteps = num_timesteps

    def forward_diffusion(self, images, timesteps) -> tuple[torch.Tensor, torch.Tensor]:
        gaussian_noise = torch.randn(images.shape).to(images.device)
        alpha_hat = self.alphas_hat[timesteps].to(images.device)
        alpha_hat = broadcast(alpha_hat, images)

        return torch.sqrt(alpha_hat) * images + torch.sqrt(1 - alpha_hat) * gaussian_noise, gaussian_noise

    def reverse_diffusion(self, model, noisy_images, timesteps):
        predicted_noise = model(noisy_images, timesteps)
        return predicted_noise

    @torch.no_grad()
    def Mag_sampling(self, model, initial_noise, device, save_all_steps=False):
        image = initial_noise
        images = []
        for timestep in tqdm(range(self.num_timesteps - 1, -1, -1)):
            # Broadcast timestep for batch_size
            ts = timestep * torch.ones(image.shape[0], dtype=torch.long, device=device)
            predicted_noise = model(image, ts)
            beta_t = self.betas[timestep].to(device)
            alpha_t = self.alphas[timestep].to(device)
            alpha_hat = self.alphas_hat[timestep].to(device)

            # Algorithm 2, step 4: calculate x_{t-1} with alphas and variance.
            # Since paper says we can use fixed variance (section 3.2, in the beginning),
            # we will calculate the one which assumes we have x0 deterministically set to one point.
            alpha_hat_prev = self.alphas_hat[timestep - 1].to(device)
            beta_t_hat = (1 - alpha_hat_prev) / (1 - alpha_hat) * beta_t
            variance = torch.sqrt(beta_t_hat) * torch.randn(image.shape).to(device) if timestep > 0 else 0

            image = torch.pow(alpha_t, -0.5) * (image -
                                                beta_t / torch.sqrt((1 - alpha_hat_prev)) *
                                                predicted_noise) + variance
            if save_all_steps:
                images.append(image.cpu())
        return images if save_all_steps else image

    def Mag_sampling_N_power(self, model, initial_noise, device, save_all_steps=False, N_power=float):
        image = initial_noise
        images = []
        for timestep in tqdm(range(self.num_timesteps - 1, -1, -1)):
            ts = timestep * torch.ones(image.shape[0], dtype=torch.long, device=device)
            predicted_noise = model(image, ts)
            beta_t = self.betas[timestep].to(device)
            alpha_t = self.alphas[timestep].to(device)
            alpha_hat = self.alphas_hat[timestep].to(device)

            alpha_hat_prev = self.alphas_hat[timestep - 1].to(device)
            beta_t_hat = (1 - alpha_hat_prev) / (1 - alpha_hat) * beta_t
            variance = torch.tensor(math.sqrt(N_power)) * torch.sqrt(beta_t_hat) * torch.randn(image.shape).to(device) if timestep > 0 else 0

            image = torch.pow(alpha_t, -0.5) * (image -
                                                beta_t / torch.sqrt((1 - alpha_hat_prev)) *
                                                predicted_noise) + variance
            if save_all_steps:
                images.append(image.cpu())
        return images if save_all_steps else image

    def Mag_sampling_CN_power(self, model, initial_noise, device, save_all_steps=False, N_power=float):
        image = initial_noise
        images = []
        for timestep in tqdm(range(self.num_timesteps - 1, -1, -1)):
            ts = timestep * torch.ones(image.shape[0], dtype=torch.long, device=device)
            predicted_noise = model(image, ts)
            beta_t = self.betas[timestep].to(device)
            alpha_t = self.alphas[timestep].to(device)
            alpha_hat = self.alphas_hat[timestep].to(device)

            alpha_hat_prev = self.alphas_hat[timestep - 1].to(device)
            beta_t_hat = (1 - alpha_hat_prev) / (1 - alpha_hat) * beta_t
            variance_R = torch.tensor(math.sqrt(N_power)) * torch.sqrt(beta_t_hat) * torch.randn(image.shape).to(device) if timestep > 0 else 0
            variance_I = torch.tensor(math.sqrt(N_power)) * torch.sqrt(beta_t_hat) * torch.randn(image.shape).to(device) if timestep > 0 else 0
            variance = torch.sqrt(torch.pow(variance_R, 2) + torch.pow(variance_I, 2)) if timestep > 0 else 0

            image = torch.pow(alpha_t, -0.5) * (image -
                                                beta_t / torch.sqrt((1 - alpha_hat_prev)) *
                                                predicted_noise) + variance
            if save_all_steps:
                images.append(image.cpu())
        return images if save_all_steps else image

    def Complex_sampling(self, args, model, initial_noise, device, save_all_steps=False):
        R_image = initial_noise[:,:,:,:,0]
        I_image = initial_noise[:,:,:,:,1]

        images = []
        for timestep in tqdm(range(self.num_timesteps - 1, -1, -1)):
            ts_R = timestep * torch.ones(R_image.shape[0], dtype=torch.long, device=device)
            ts_I = timestep * torch.ones(I_image.shape[0], dtype=torch.long, device=device)

            predicted_noise_R = model(R_image, ts_R)
            predicted_noise_I = model(I_image, ts_I)

            beta_t = self.betas[timestep].to(device)
            alpha_t = self.alphas[timestep].to(device)
            alpha_hat = self.alphas_hat[timestep].to(device)

            alpha_hat_prev = self.alphas_hat[timestep - 1].to(device)
            beta_t_hat = (1 - alpha_hat_prev) / (1 - alpha_hat) * beta_t
            variance_R = torch.sqrt(beta_t_hat) * torch.randn(R_image.shape).to(device) if timestep > 0 else 0
            variance_I = torch.sqrt(beta_t_hat) * torch.randn(I_image.shape).to(device) if timestep > 0 else 0

            R_image = torch.pow(alpha_t, -0.5) * (R_image -
                                                  beta_t / torch.sqrt((1 - alpha_hat_prev)) *
                                                  predicted_noise_R) + variance_R
            I_image = torch.pow(alpha_t, -0.5) * (I_image -
                                                  beta_t / torch.sqrt((1 - alpha_hat_prev)) *
                                                  predicted_noise_I) + variance_I
            
            R_image = torch.unsqueeze(R_image, dim=4)
            I_image = torch.unsqueeze(I_image, dim=4)
            image = torch.cat((R_image, I_image), dim=4)

            R_image = image[:,:,:,:,0]
            I_image = image[:,:,:,:,1]

            if save_all_steps:
                images.append(image.cpu())

        return images if save_all_steps else image
    
    def Complex_sampling_N_power(self, args, model, initial_noise, device, save_all_steps=False, N_R_power=float, N_I_power=float):
        R_image = initial_noise[:,:,:,:,0]
        I_image = initial_noise[:,:,:,:,1]

        images = []
        for timestep in tqdm(range(self.num_timesteps - 1, -1, -1)):
            ts_R = timestep * torch.ones(R_image.shape[0], dtype=torch.long, device=device)
            ts_I = timestep * torch.ones(I_image.shape[0], dtype=torch.long, device=device)

            predicted_noise_R = model(R_image, ts_R)
            predicted_noise_I = model(I_image, ts_I)

            beta_t = self.betas[timestep].to(device)
            alpha_t = self.alphas[timestep].to(device)
            alpha_hat = self.alphas_hat[timestep].to(device)

            alpha_hat_prev = self.alphas_hat[timestep - 1].to(device)
            beta_t_hat = (1 - alpha_hat_prev) / (1 - alpha_hat) * beta_t
            variance_R = torch.tensor(math.sqrt(N_R_power)) * torch.sqrt(beta_t_hat) * torch.randn(R_image.shape).to(device) if timestep > 0 else 0
            variance_I = torch.tensor(math.sqrt(N_I_power)) * torch.sqrt(beta_t_hat) * torch.randn(I_image.shape).to(device) if timestep > 0 else 0

            R_image = torch.pow(alpha_t, -0.5) * (R_image -
                                                  beta_t / torch.sqrt((1 - alpha_hat_prev)) *
                                                  predicted_noise_R) + variance_R
            I_image = torch.pow(alpha_t, -0.5) * (I_image -
                                                  beta_t / torch.sqrt((1 - alpha_hat_prev)) *
                                                  predicted_noise_I) + variance_I
            
            R_image = torch.unsqueeze(R_image, dim=4)
            I_image = torch.unsqueeze(I_image, dim=4)
            image = torch.cat((R_image, I_image), dim=4)

            R_image = image[:,:,:,:,0]
            I_image = image[:,:,:,:,1]

            if save_all_steps:
                images.append(image.cpu())

        return images if save_all_steps else image




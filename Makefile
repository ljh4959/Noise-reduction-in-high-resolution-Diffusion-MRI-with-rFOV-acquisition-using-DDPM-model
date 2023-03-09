Norm_Score_t1 = Score_t1_x_mag_NE
.PHONY: train_$(Norm_Score_t1)
train_$(Norm_Score_t1):
	python -m src train --train_file /SSD5_8TB/LJH/DWI_data/train_LLR/train_Y_Complex_sorted.npy \
                                  --val_X_file /SSD5_8TB/LJH/DWI_data/val/img_recon_cor_EX5557_p105.mat \
								  --val_Y_file /SSD5_8TB/LJH/DWI_data/val_LLR/LLR_cor_EX5557_p105.mat \
								  --test_X_file /SSD5_8TB/LJH/DWI_data/test/img_recon_cor_EX5559_p106.mat \
								  --test_Y_file /SSD5_8TB/LJH/DWI_data/test_LLR/LLR_cor_EX5559_p106.mat \
                                  --name $(Norm_Score_t1) \
										--batch 60 \
										--num_epochs 200 \
										--learning_rate 2e-4 \
										--diffusion_timesteps 3000

.PHONY: train_$(Norm_Score_t1)_resume
train_$(Norm_Score_t1)_resume:
	python -m src train --train_file /SSD5_8TB/LJH/DWI_data/train_LLR/train_Y_Complex_sorted.npy \
                                  --val_X_file /SSD5_8TB/LJH/DWI_data/val/img_recon_cor_EX5557_p105.mat \
								  --val_Y_file /SSD5_8TB/LJH/DWI_data/val_LLR/LLR_cor_EX5557_p105.mat \
								  --test_X_file /SSD5_8TB/LJH/DWI_data/test/img_recon_cor_EX5559_p106.mat \
								  --test_Y_file /SSD5_8TB/LJH/DWI_data/test_LLR/LLR_cor_EX5559_p106.mat \
                                  --name $(Norm_Score_t1) \
										--batch 60 \
										--num_epochs 200 \
										--learning_rate 2e-4 \
										--diffusion_timesteps 3000 \
                                        --resume $(Norm_Score_t1)_checkpoint.pth.tar

.PHONY: eval_Mag_$(Norm_Score_t1)
eval_Mag_$(Norm_Score_t1):
	python -m src eval_Mag --weight_file $(Norm_Score_t1)_best.pth.tar \
								  --test_X_file /SSD5_8TB/LJH/DWI_data/test/img_recon_cor_EX5588_p111.mat \
								  --test_Y_file /SSD5_8TB/LJH/DWI_data/test_LLR/LLR_cor_EX5588_p111.mat \
                                  --name $(Norm_Score_t1) \
								  		--diffusion_timesteps 3000 \
										--batch 60

.PHONY: eval_Complex_$(Norm_Score_t1)
eval_Complex_$(Norm_Score_t1):
	python -m src eval_Complex --weight_file $(Norm_Score_t1)_best.pth.tar \
								  --test_X_file /SSD5_8TB/LJH/DWI_data/test/img_recon_ax_EX6198_p144.mat \
								  --test_Y_file /SSD5_8TB/LJH/DWI_data/test_LLR/LLR_ax_EX6198_p144.mat \
                                  --name $(Norm_Score_t1) \
								  		--diffusion_timesteps 3000 \
										--batch 60
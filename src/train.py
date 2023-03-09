import torch
import os
import torch.nn as nn

from src.model.unet import UNet

from torch.utils.data import DataLoader
from .utils.common import train_dataset_Mag_x, train_dataset_Mag_y, train_dataset_Mag_z, val_dataset_Mag_X_x, val_dataset_Mag_X_y, val_dataset_Mag_X_z,\
                          val_dataset_Mag_Y_x, val_dataset_Mag_Y_y, val_dataset_Mag_Y_z,\
                          val_X_postprocess, val_res_postprocess, val_Y_postprocess, test_X_postprocess, test_res_postprocess, test_Y_postprocess,\
                          N_estimator
from torch.utils.data.distributed import DistributedSampler
import random
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
from scipy.io import savemat
import shutil
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.nn.parallel import DistributedDataParallel

from src.scheduler.ddpm import DDPMPipeline
from tqdm import tqdm

best_loss = 1e12


def setup(parser):
    parser.add_argument("--name",  type=str, help="Name")
    parser.add_argument("--train_file", type=str, help="Path for input h5 training")
    
    parser.add_argument("--val_X_file", type=str, help="Path for input h5 validation")
    parser.add_argument("--val_Y_file", type=str, help="Path for input h5 validation")
    parser.add_argument("--test_X_file", type=str, help="Path for input h5 testing")
    parser.add_argument("--test_Y_file", type=str, help="Path for input h5 testing")
    
    parser.add_argument("--workers", default=4, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--batch", type=int, default=10, help="Number of Batchsize")
    
    parser.add_argument("--plot_index-val", default=9, type=int, metavar="I", help="plot index I of validation dataset")
    parser.add_argument("--plot_index-test", default=9, type=int, metavar="I", help="plot index I of test dataset")
    parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
    parser.add_argument("--plot_freq", default=1, type=int, metavar="N", help="plot every N epoch")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
    parser.add_argument("--num_epochs", default=10000, type=int, metavar="N", help="total epochs")
    parser.add_argument("--learning_rate", default=2e-4, type=float, help="Learning rate for network")
    parser.add_argument("--diffusion_timesteps", default=1000, type=int, help="diffusion time steps")

    parser.add_argument("--Y_train_Mag_mean", default=0, type=float)
    parser.add_argument("--Y_train_Mag_std", default=0, type=float)
    
    parser.add_argument("--X_val_Mag_mean", default=0, type=float)
    parser.add_argument("--X_val_Mag_std", default=0, type=float)
    parser.add_argument("--Y_val_Mag_mean", default=0, type=float)
    parser.add_argument("--Y_val_Mag_std", default=0, type=float)

    parser.add_argument("--X_test_Mag_mean", default=0, type=float)
    parser.add_argument("--X_test_Mag_std", default=0, type=float)
    parser.add_argument("--Y_test_Mag_mean", default=0, type=float)
    parser.add_argument("--Y_test_Mag_std", default=0, type=float)

    parser.add_argument("--N_power", default=0.0, type=float)
    
    parser.set_defaults(func=main)            

device = torch.device('cuda:0')


def main(args):
    random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    np.random.seed(0)

    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))
    

def main_worker(gpu, ngpus, args):
    global best_loss
    
    print("Use GPU: {} for training".format(gpu))
    rank = gpu
    world_size = ngpus
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:11111",
        world_size=world_size,
        rank=rank,
    )

    #################################### load data ####################################
    Y_train_Mag, Y_train_Mag_mean, Y_train_Mag_std = train_dataset_Mag_x(args.train_file)
    # Y_train_Mag, Y_train_Mag_mean, Y_train_Mag_std = val_dataset_Mag_X_y(args.val_X_file)
    
    X_val_Mag, args.X_val_Mag_mean, args.X_val_Mag_std = val_dataset_Mag_X_x(args.val_X_file)
    Y_val_Mag, args.Y_val_Mag_mean, args.Y_val_Mag_std = val_dataset_Mag_Y_x(args.val_Y_file)
    
    X_test_Mag, args.X_test_Mag_mean, args.X_test_Mag_std = val_dataset_Mag_X_x(args.test_X_file)
    Y_test_Mag, args.Y_test_Mag_mean, args.Y_test_Mag_std = val_dataset_Mag_Y_x(args.test_Y_file)

    train_dataset=torch.utils.data.TensorDataset(Y_train_Mag)
    val_dataset=torch.utils.data.TensorDataset(X_val_Mag, Y_val_Mag)
    test_dataset=torch.utils.data.TensorDataset(X_test_Mag, Y_test_Mag)
    
    args.workers = int((args.workers + ngpus - 1) / ngpus)
    
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    #################################### model setup ####################################
    model = UNet(args)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    model = DistributedDataParallel(model, device_ids=[gpu])
    print("Model size: ", sum([p.numel() for p in model.parameters() if p.requires_grad]))
    
    criterion = nn.MSELoss().cuda(gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    #################################### resume setup ####################################
    if args.resume:
        if os.path.isfile(args.resume):
            loc = "cpu"
            checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            best_loss = checkpoint["best_loss"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])

            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.learning_rate
    
    diffusion_pipeline = DDPMPipeline(beta_start=1e-6, beta_end=1e-4, num_timesteps=args.diffusion_timesteps)
    global_step = args.start_epoch * len(train_dataloader)
    
    is_printer = rank % ngpus == 0
    if is_printer:
        writer = SummaryWriter(f"runs/{args.name}")
        
    #################################### Train loop ####################################
    for epoch in range(args.start_epoch, args.num_epochs):
        losses = []
        
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        mean_loss = 0
        model.train()
        
        for step, original_images in enumerate(train_dataloader):
            original_images = original_images[0]
            original_images = original_images.cuda(gpu, non_blocking=True)
            
            batch_size = original_images.shape[0]
            
            # Sample a random timestep for each image
            timesteps = torch.randint(0, diffusion_pipeline.num_timesteps, (batch_size,), device=gpu).long()
        
            # Apply forward diffusion process at the given timestep
            noisy_images, noise = diffusion_pipeline.forward_diffusion(original_images, timesteps)
            noisy_images = noisy_images.to(gpu)
            
            # Predict the noise residual
            noise_pred = diffusion_pipeline.reverse_diffusion(model=model, noisy_images=noisy_images, timesteps=timesteps)
            loss = criterion(noise_pred, noise)
            
            # Calculate new mean on the run without accumulating all the values
            mean_loss = mean_loss + (loss.detach().item() - mean_loss) / (step + 1)
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            losses += [loss.item()]
            
            progress_bar.update(1)
            logs = {"loss": mean_loss, "step": global_step}
            progress_bar.set_postfix(**logs)
            global_step += 1

        # Evaluation
        val_loss = evaluate(args, val_loader, diffusion_pipeline, model, gpu, criterion)
        
        # remember best loss and save checkpoint
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        
        if is_printer:
            save_checkpoint(
                {
                    "epoch": epoch+1,
                    "name": args.name,
                    "state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                args.name,
            )
            
            writer.add_scalar("Loss/training", mean_loss, epoch)
            writer.add_scalar("Loss/validation", val_loss, epoch)
            
            if epoch % args.plot_freq == 0:
                valimage = show_val(args, val_loader, model, gpu, diffusion_pipeline)
                testimage = show_test(args, test_loader, model, gpu, diffusion_pipeline)
                writer.add_image(
                    f"validation_index{args.plot_index_val}", valimage, epoch
                )
                writer.add_image(f"test_index{args.plot_index_test}", testimage, epoch)
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.detach().clone().cpu(), epoch)
            
            
def show_val(args, val_loader, model, gpu, pipeline):
    model.eval()
    inputs = []
    outputs = []
    gts = []
    
    with torch.no_grad():
        for X_test, Y_test in val_loader:
            X_test = X_test.cuda(gpu, non_blocking=True)
            Y_test = Y_test.cuda(gpu, non_blocking=True)

            # N_power estimator
            args.N_power = N_estimator(X_test)
            # print("show_val N_power:", args.N_power)
            
            # Reverse diffusion
            output = pipeline.Mag_sampling_N_power(model, X_test, device=gpu, N_power=args.N_power)
            
            inputs += [X_test.detach().cpu()]
            outputs += [output.detach().cpu()]
            gts += [Y_test.detach().cpu()]
    
        inputs = torch.cat(inputs)
        outputs = torch.cat(outputs)
        gts = torch.cat(gts)
        
        inputs = val_X_postprocess(args, inputs)
        outputs = val_res_postprocess(args, outputs)
        gts = val_Y_postprocess(args, gts)
        
        X_test = inputs[args.plot_index_val].unsqueeze(0)
        output = outputs[args.plot_index_val].unsqueeze(0)
        Y_test = gts[args.plot_index_val].unsqueeze(0)
        
        abs_tensor1 = X_test.detach().cpu()
        abs_tensor3 = output.detach().cpu()
        abs_tensor4 = Y_test.detach().cpu()
        abs_tensor5 = abs(output.detach().cpu() - Y_test.detach().cpu())*5

        def _make_image(idx):
            return make_grid(
                [
                    abs_tensor1[0, idx].unsqueeze(0),
                    abs_tensor3[0, idx].unsqueeze(0),
                    abs_tensor4[0, idx].unsqueeze(0),
                    abs_tensor5[0, idx].unsqueeze(0),
                ],
                normalize=True,
                range=(0, 2),
                padding=0,
            )

    return make_grid([_make_image(i) for i in range(7)], nrow=1, padding=0)
            

def show_test(args, test_loader, model, gpu, pipeline):
    model.eval()
    inputs = []
    outputs = []
    gts = []
    
    with torch.no_grad():
        for X_test, Y_test in test_loader:
            X_test = X_test.cuda(gpu, non_blocking=True)
            Y_test = Y_test.cuda(gpu, non_blocking=True)

            # N_power estimator
            args.N_power = N_estimator(X_test)
            # print("show_test N_power:", args.N_power)
            
            # Reverse diffusion
            output = pipeline.Mag_sampling_N_power(model, X_test, device=gpu, N_power=args.N_power)
            
            inputs += [X_test.detach().cpu()]
            outputs += [output.detach().cpu()]
            gts += [Y_test.detach().cpu()]
    
        inputs = torch.cat(inputs)
        outputs = torch.cat(outputs)
        gts = torch.cat(gts)
        
        inputs = test_X_postprocess(args, inputs)
        outputs = test_res_postprocess(args, outputs)
        gts = test_Y_postprocess(args, gts)
        
        X_test = inputs[args.plot_index_val].unsqueeze(0)
        output = outputs[args.plot_index_val].unsqueeze(0)
        Y_test = gts[args.plot_index_val].unsqueeze(0)
        
        abs_tensor1 = X_test.detach().cpu()
        abs_tensor3 = output.detach().cpu()
        abs_tensor4 = Y_test.detach().cpu()
        abs_tensor5 = abs(output.detach().cpu() - Y_test.detach().cpu())*5

        def _make_image(idx):
            return make_grid(
                [
                    abs_tensor1[0, idx].unsqueeze(0),
                    abs_tensor3[0, idx].unsqueeze(0),
                    abs_tensor4[0, idx].unsqueeze(0),
                    abs_tensor5[0, idx].unsqueeze(0),
                ],
                normalize=True,
                range=(0, 2),
                padding=0,
            )

    return make_grid([_make_image(i) for i in range(7)], nrow=1, padding=0)

            
def save_checkpoint(state, is_best, name):
    filename = f"{name}_checkpoint.pth.tar"
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f"{name}_best.pth.tar")
        
        
def evaluate(args, val_loader, pipeline, model, gpu, criterion):
    model.eval()
    outputs = []
    gts = []
    
    with torch.no_grad():
        for X_v, Y_v in val_loader:
            Y_v = Y_v.cuda(gpu, non_blocking=True)
            X_v = X_v.cuda(gpu, non_blocking=True)

            # N_power estimator
            args.N_power = N_estimator(X_v)
            # print("evaluate N_power:", args.N_power)
            
            # Reverse diffusion for T timesteps
            output = pipeline.Mag_sampling_N_power(model, X_v, device=gpu, N_power=args.N_power)
                        
            outputs += [output.detach().cpu()]
            gts += [Y_v.detach().cpu()]
            
        outputs = torch.cat(outputs)
        gts = torch.cat(gts)
        
        outputs = val_res_postprocess(args, outputs)
        gts = val_Y_postprocess(args, gts)
            
        loss = criterion(outputs, gts)        
    return loss
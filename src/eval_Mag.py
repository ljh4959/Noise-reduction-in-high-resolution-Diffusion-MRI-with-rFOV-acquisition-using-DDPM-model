import random
import numpy as np
import torch.multiprocessing as mp

import torch
from tqdm import tqdm
from scipy.io import savemat
from src.model.unet import UNet
from src.scheduler.ddpm import DDPMPipeline
from .utils.common import val_dataset_Mag_X_x, val_dataset_Mag_X_y, val_dataset_Mag_X_z,\
                          val_dataset_Mag_Y_x, val_dataset_Mag_Y_y, val_dataset_Mag_Y_z,\
                          val_dataset_Mag_Y_x_noising, val_dataset_Mag_Y_y_noising, val_dataset_Mag_Y_z_noising
from torch.utils.data import DataLoader

def setup(parser):
    parser.add_argument("--name", type=str, help="Name")
    parser.add_argument("--weight_file", type=str, help="Path for model weight")
    
    parser.add_argument("--test_X_file", type=str, help="Path for input h5 testing")
    parser.add_argument("--test_Y_file", type=str, help="Path for input h5 testing")
    
    parser.add_argument("--workers", default=0, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--batch", type=int, default=10, help="Number of Batchsize")
    
    parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
    parser.add_argument("--num_epochs", default=10, type=int, metavar="N", help="total epochs")
    parser.add_argument("--diffusion_timesteps", default=10000, type=int, help="diffusion time steps")

    parser.add_argument("--X_test_Mag_mean", default=0, type=float)
    parser.add_argument("--X_test_Mag_std", default=0, type=float)
    parser.add_argument("--Y_test_Mag_mean", default=0, type=float)
    parser.add_argument("--Y_test_Mag_std", default=0, type=float)

    parser.add_argument("--N_power", default=1, type=float)
    
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
    
    parser.set_defaults(func=main)   
    
    
def main(args):
    random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    np.random.seed(0)

    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))
    
    
def main_worker(gpu, ngpus, args):
    print("Use GPU: {} for evaluating".format(gpu))
    
    model = UNet(args)
    diffusion_pipeline = DDPMPipeline(beta_start=1e-6, beta_end=1e-4, num_timesteps=args.diffusion_timesteps)
    
    X_test, args.X_test_Mag_mean, args.X_test_Mag_std = val_dataset_Mag_X_x(args.test_X_file)
    # X_test, args.X_test_Mag_mean, args.X_test_Mag_std = val_dataset_Mag_Y_x_noising(args.test_X_file, args.N_power)
    Y_test, args.Y_test_Mag_mean, args.Y_test_Mag_std = val_dataset_Mag_Y_x(args.test_Y_file)
    
    test_dataset=torch.utils.data.TensorDataset(X_test, Y_test)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    loc = "cuda:{}".format(args.gpu)
    data = torch.load(args.weight_file, map_location=loc)
    state_dict = {k[len("module.") :]: v for k, v in data["state_dict"].items()}
    model.load_state_dict(state_dict)
    print(f"Loaded weight at epoch {data['epoch']}")
    
    to_save = {}
    output, gt, input = evaluate(args, test_loader, diffusion_pipeline, model, args.gpu)
    to_save["test"] = dict(output=output, gt=gt, input=input,\
            X_Mag_mean=args.X_test_Mag_mean.numpy(), X_Mag_std=args.X_test_Mag_std.numpy(),\
            Y_Mag_mean=args.Y_test_Mag_mean.numpy(), Y_Mag_std=args.Y_test_Mag_std.numpy())

    savemat(f"{args.name}_Base_p111_N5.mat", to_save)
    
    
def evaluate(args, val_loader, pipeline, model, gpu):
    model.eval()
    
    inputs = []
    outputs = []
    gts = []
    
    with torch.no_grad():
        for X_v, Y_v in tqdm(val_loader):
            Y_v = Y_v.cuda(gpu, non_blocking=True)
            X_v = X_v.cuda(gpu, non_blocking=True)
            
            # Reverse diffusion for T timesteps
            output = pipeline.Mag_sampling(model, X_v, device=gpu)
            
            inputs += [X_v.detach().cpu()]
            outputs += [output.detach().cpu()]
            gts += [Y_v.detach().cpu()]
        
        inputs = torch.cat(inputs)
        outputs = torch.cat(outputs)
        gts = torch.cat(gts)
            
    return outputs.numpy(), gts.numpy(), inputs.numpy()
        
        
    
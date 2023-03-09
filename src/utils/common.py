import torch
from scipy import io
import numpy as np
from torchvision import transforms
import math


def broadcast(values, broadcast_to):
    values = values.flatten()

    while len(values.shape) < len(broadcast_to.shape):
        values = values.unsqueeze(-1)

    return values

def postprocess(images):
    images = (images / 2 + 0.5)
    return images

def mean_std_norm(data):
    mean = torch.mean(data)
    std = torch.std(data)
    
    transform = transforms.Normalize(torch.mean(data),torch.std(data))
    data = transform(data)
    return data, mean, std

def mean_std_norm_Complex(data):
    mean = torch.mean(data)
    std = torch.std(data)
    
    transform_real = transforms.Normalize(torch.mean(data),torch.std(data))
    data = transform_real(data)
    return data, mean, std

# train functions
def train_dataset_Mag_x(h5_path):
    Y = torch.from_numpy(np.load(h5_path))
    
    Y_x = torch.cat((torch.unsqueeze(Y[:,0,:,:,:],1),
                     torch.unsqueeze(Y[:,1,:,:,:],1),
                     torch.unsqueeze(Y[:,4,:,:,:],1),
                     torch.unsqueeze(Y[:,7,:,:,:],1),
                     torch.unsqueeze(Y[:,10,:,:,:],1),
                     torch.unsqueeze(Y[:,13,:,:,:],1),
                     torch.unsqueeze(Y[:,16,:,:,:],1)), axis=1)
    
    Y_x = torch.sqrt(torch.pow(Y_x[:,:,:,:,0],2) + torch.pow(Y_x[:,:,:,:,1],2))
        
    res, mean, std = mean_std_norm(Y_x)
    
    print("train data size:", res.shape)
    
    return (
        res.float(), mean, std
    )

def train_dataset_Mag_y(h5_path):
    Y = torch.from_numpy(np.load(h5_path))
    
    Y_x = torch.cat((torch.unsqueeze(Y[:,0,:,:,:],1),
                     torch.unsqueeze(Y[:,2,:,:,:],1),
                     torch.unsqueeze(Y[:,5,:,:,:],1),
                     torch.unsqueeze(Y[:,8,:,:,:],1),
                     torch.unsqueeze(Y[:,11,:,:,:],1),
                     torch.unsqueeze(Y[:,14,:,:,:],1),
                     torch.unsqueeze(Y[:,17,:,:,:],1)), axis=1)
    
    Y_x = torch.sqrt(torch.pow(Y_x[:,:,:,:,0],2) + torch.pow(Y_x[:,:,:,:,1],2))
        
    res, mean, std = mean_std_norm(Y_x)
    
    print("train data size:", res.shape)
    
    return (
        res.float(), mean, std
    )

def train_dataset_Mag_z(h5_path):
    Y = torch.from_numpy(np.load(h5_path))
    
    Y_x = torch.cat((torch.unsqueeze(Y[:,0,:,:,:],1),
                     torch.unsqueeze(Y[:,3,:,:,:],1),
                     torch.unsqueeze(Y[:,6,:,:,:],1),
                     torch.unsqueeze(Y[:,9,:,:,:],1),
                     torch.unsqueeze(Y[:,12,:,:,:],1),
                     torch.unsqueeze(Y[:,15,:,:,:],1),
                     torch.unsqueeze(Y[:,18,:,:,:],1)), axis=1)
    
    Y_x = torch.sqrt(torch.pow(Y_x[:,:,:,:,0],2) + torch.pow(Y_x[:,:,:,:,1],2))
        
    res, mean, std = mean_std_norm(Y_x)
    
    print("train data size:", res.shape)
    
    return (
        res.float(), mean, std
    )

# val functions
def val_dataset_Mag_X_x(h5_path):
    f = io.loadmat(h5_path)
    img_all = None
    
    if img_all is None:
        img_all = f.get('img_all')
    
    X = torch.from_numpy(np.complex64(img_all[:,:,8,:,:]))
    X = X.permute(2,3,0,1)
    
    X_x = torch.cat((torch.unsqueeze(X[:,0,:,:],1),
                     torch.unsqueeze(X[:,1,:,:],1),
                     torch.unsqueeze(X[:,4,:,:],1),
                     torch.unsqueeze(X[:,7,:,:],1),
                     torch.unsqueeze(X[:,10,:,:],1),
                     torch.unsqueeze(X[:,13,:,:],1),
                     torch.unsqueeze(X[:,16,:,:],1)), axis=1)
    
    X_x = torch.sqrt(torch.pow(torch.real(X_x),2) + torch.pow(torch.imag(X_x),2))

    res, mean, std = mean_std_norm(X_x)

    noise = torch.tensor(math.sqrt(0.05) * np.random.randn(res.shape[0],res.shape[1],res.shape[2],res.shape[3]))
    res = res + noise
    
    print("val X data size:", res.shape)
    
    return (
        res.float(), mean, std
    )

def val_dataset_Mag_X_y(h5_path):
    f = io.loadmat(h5_path)
    img_all = None
    
    if img_all is None:
        img_all = f.get('img_all')
    
    X = torch.from_numpy(np.complex64(img_all[:,:,0,:,:]))
    X = X.permute(2,3,0,1)
    
    X_x = torch.cat((torch.unsqueeze(X[:,0,:,:],1),
                     torch.unsqueeze(X[:,2,:,:],1),
                     torch.unsqueeze(X[:,5,:,:],1),
                     torch.unsqueeze(X[:,8,:,:],1),
                     torch.unsqueeze(X[:,11,:,:],1),
                     torch.unsqueeze(X[:,14,:,:],1),
                     torch.unsqueeze(X[:,17,:,:],1)), axis=1)
    
    X_x = torch.sqrt(torch.pow(torch.real(X_x),2) + torch.pow(torch.imag(X_x),2))

    res, mean, std = mean_std_norm(X_x)
    
    print("val X data size:", res.shape)
    
    return (
        res.float(), mean, std
    )

def val_dataset_Mag_X_z(h5_path):
    f = io.loadmat(h5_path)
    img_all = None
    
    if img_all is None:
        img_all = f.get('img_all')
    
    X = torch.from_numpy(np.complex64(img_all[:,:,0,:,:]))
    X = X.permute(2,3,0,1)
    
    X_x = torch.cat((torch.unsqueeze(X[:,0,:,:],1),
                     torch.unsqueeze(X[:,3,:,:],1),
                     torch.unsqueeze(X[:,6,:,:],1),
                     torch.unsqueeze(X[:,9,:,:],1),
                     torch.unsqueeze(X[:,12,:,:],1),
                     torch.unsqueeze(X[:,15,:,:],1),
                     torch.unsqueeze(X[:,18,:,:],1)), axis=1)
    
    X_x = torch.sqrt(torch.pow(torch.real(X_x),2) + torch.pow(torch.imag(X_x),2))

    res, mean, std = mean_std_norm(X_x)
    
    print("val X data size:", res.shape)
    
    return (
        res.float(), mean, std
    )

def val_dataset_Mag_Y_x(h5_path):
    f = io.loadmat(h5_path)
    denoised_img = None
    
    if denoised_img is None:
        denoised_img = f.get('denoised_img')
    
    Y = torch.from_numpy(np.complex64(denoised_img))
    Y = Y.permute(2,3,0,1)
    
    Y_x = torch.cat((torch.unsqueeze(Y[:,0,:,:],1),
                     torch.unsqueeze(Y[:,1,:,:],1),
                     torch.unsqueeze(Y[:,4,:,:],1),
                     torch.unsqueeze(Y[:,7,:,:],1),
                     torch.unsqueeze(Y[:,10,:,:],1),
                     torch.unsqueeze(Y[:,13,:,:],1),
                     torch.unsqueeze(Y[:,16,:,:],1)), axis=1)
    
    Y_x = torch.sqrt(torch.pow(torch.real(Y_x),2) + torch.pow(torch.imag(Y_x),2))
    
    res, mean, std = mean_std_norm(Y_x)
    
    print("val Y data size:", res.shape)
    
    return (
        res.float(), mean, std
    )

def val_dataset_Mag_Y_y(h5_path):
    f = io.loadmat(h5_path)
    denoised_img = None
    
    if denoised_img is None:
        denoised_img = f.get('denoised_img')
    
    Y = torch.from_numpy(np.complex64(denoised_img))
    Y = Y.permute(2,3,0,1)
    
    Y_x = torch.cat((torch.unsqueeze(Y[:,0,:,:],1),
                     torch.unsqueeze(Y[:,2,:,:],1),
                     torch.unsqueeze(Y[:,5,:,:],1),
                     torch.unsqueeze(Y[:,8,:,:],1),
                     torch.unsqueeze(Y[:,11,:,:],1),
                     torch.unsqueeze(Y[:,14,:,:],1),
                     torch.unsqueeze(Y[:,17,:,:],1)), axis=1)
    
    Y_x = torch.sqrt(torch.pow(torch.real(Y_x),2) + torch.pow(torch.imag(Y_x),2))
    
    res, mean, std = mean_std_norm(Y_x)
    
    print("val Y data size:", res.shape)
    
    return (
        res.float(), mean, std
    )

def val_dataset_Mag_Y_z(h5_path):
    f = io.loadmat(h5_path)
    denoised_img = None
    
    if denoised_img is None:
        denoised_img = f.get('denoised_img')
    
    Y = torch.from_numpy(np.complex64(denoised_img))
    Y = Y.permute(2,3,0,1)
    
    Y_x = torch.cat((torch.unsqueeze(Y[:,0,:,:],1),
                     torch.unsqueeze(Y[:,3,:,:],1),
                     torch.unsqueeze(Y[:,6,:,:],1),
                     torch.unsqueeze(Y[:,9,:,:],1),
                     torch.unsqueeze(Y[:,12,:,:],1),
                     torch.unsqueeze(Y[:,15,:,:],1),
                     torch.unsqueeze(Y[:,18,:,:],1)), axis=1)
    
    Y_x = torch.sqrt(torch.pow(torch.real(Y_x),2) + torch.pow(torch.imag(Y_x),2))
    
    res, mean, std = mean_std_norm(Y_x)
    
    print("val Y data size:", res.shape)
    
    return (
        res.float(), mean, std
    )

# postprocess functions
def val_X_postprocess(args, images):    
    res = (images / 2 + 0.5)
    return res
    
def val_Y_postprocess(args, images):
    res = (images / 2 + 0.5)
    return res

def val_res_postprocess(args, images):
    res = (images / 2 + 0.5)
    return res

def test_X_postprocess(args, images):
    res = (images / 2 + 0.5)
    return res

def test_Y_postprocess(args, images):
    res = (images / 2 + 0.5)
    return res

def test_res_postprocess(args, images):
    res = (images / 2 + 0.5)
    return res

# Complex_sampling functions
def val_dataset_Complex_X_x(h5_path):
    f = io.loadmat(h5_path)
    img_all = None
    
    if img_all is None:
        img_all = f.get('img_all')
    
    X = torch.from_numpy(np.complex64(img_all[:,:,0,:,:]))
    X = X.permute(2,3,0,1)
    
    X_x = torch.cat((torch.unsqueeze(X[:,0,:,:],1),
                     torch.unsqueeze(X[:,1,:,:],1),
                     torch.unsqueeze(X[:,4,:,:],1),
                     torch.unsqueeze(X[:,7,:,:],1),
                     torch.unsqueeze(X[:,10,:,:],1),
                     torch.unsqueeze(X[:,13,:,:],1),
                     torch.unsqueeze(X[:,16,:,:],1)), axis=1)
    
    X_x = torch.view_as_real(X_x)

    res, mean, std = mean_std_norm_Complex(X_x)
    
    print("val X data size:", res.shape)
    
    return (
        res.float(), mean, std
    )

def val_dataset_Complex_X_y(h5_path):
    f = io.loadmat(h5_path)
    img_all = None
    
    if img_all is None:
        img_all = f.get('img_all')
    
    X = torch.from_numpy(np.complex64(img_all[:,:,0,:,:]))
    X = X.permute(2,3,0,1)
    
    X_x = torch.cat((torch.unsqueeze(X[:,0,:,:],1),
                     torch.unsqueeze(X[:,2,:,:],1),
                     torch.unsqueeze(X[:,5,:,:],1),
                     torch.unsqueeze(X[:,8,:,:],1),
                     torch.unsqueeze(X[:,11,:,:],1),
                     torch.unsqueeze(X[:,14,:,:],1),
                     torch.unsqueeze(X[:,17,:,:],1)), axis=1)
    
    X_x = torch.view_as_real(X_x)

    res, mean, std = mean_std_norm_Complex(X_x)
    
    print("val X data size:", res.shape)
    
    return (
        res.float(), mean, std
    )

def val_dataset_Complex_X_z(h5_path):
    f = io.loadmat(h5_path)
    img_all = None
    
    if img_all is None:
        img_all = f.get('img_all')
    
    X = torch.from_numpy(np.complex64(img_all[:,:,0,:,:]))
    X = X.permute(2,3,0,1)
    
    X_x = torch.cat((torch.unsqueeze(X[:,0,:,:],1),
                     torch.unsqueeze(X[:,3,:,:],1),
                     torch.unsqueeze(X[:,6,:,:],1),
                     torch.unsqueeze(X[:,9,:,:],1),
                     torch.unsqueeze(X[:,12,:,:],1),
                     torch.unsqueeze(X[:,15,:,:],1),
                     torch.unsqueeze(X[:,18,:,:],1)), axis=1)
    
    X_x = torch.view_as_real(X_x)

    res, mean, std = mean_std_norm_Complex(X_x)
    
    print("val X data size:", res.shape)
    
    return (
        res.float(), mean, std
    )

def val_dataset_Complex_Y_x(h5_path):
    f = io.loadmat(h5_path)
    denoised_img = None
    
    if denoised_img is None:
        denoised_img = f.get('denoised_img')
    
    Y = torch.from_numpy(np.complex64(denoised_img))
    Y = Y.permute(2,3,0,1)
    
    Y_x = torch.cat((torch.unsqueeze(Y[:,0,:,:],1),
                     torch.unsqueeze(Y[:,1,:,:],1),
                     torch.unsqueeze(Y[:,4,:,:],1),
                     torch.unsqueeze(Y[:,7,:,:],1),
                     torch.unsqueeze(Y[:,10,:,:],1),
                     torch.unsqueeze(Y[:,13,:,:],1),
                     torch.unsqueeze(Y[:,16,:,:],1)), axis=1)
    
    Y_x = torch.view_as_real(Y_x)
    
    res, mean, std = mean_std_norm_Complex(Y_x)
    
    print("val Y data size:", res.shape)
    
    return (
        res.float(), mean, std
    )

def val_dataset_Complex_Y_y(h5_path):
    f = io.loadmat(h5_path)
    denoised_img = None
    
    if denoised_img is None:
        denoised_img = f.get('denoised_img')
    
    Y = torch.from_numpy(np.complex64(denoised_img))
    Y = Y.permute(2,3,0,1)
    
    Y_x = torch.cat((torch.unsqueeze(Y[:,0,:,:],1),
                     torch.unsqueeze(Y[:,2,:,:],1),
                     torch.unsqueeze(Y[:,5,:,:],1),
                     torch.unsqueeze(Y[:,8,:,:],1),
                     torch.unsqueeze(Y[:,11,:,:],1),
                     torch.unsqueeze(Y[:,14,:,:],1),
                     torch.unsqueeze(Y[:,17,:,:],1)), axis=1)
    
    Y_x = torch.view_as_real(Y_x)
    
    res, mean, std = mean_std_norm_Complex(Y_x)
    
    print("val Y data size:", res.shape)
    
    return (
        res.float(), mean, std
    )

def val_dataset_Complex_Y_z(h5_path):
    f = io.loadmat(h5_path)
    denoised_img = None
    
    if denoised_img is None:
        denoised_img = f.get('denoised_img')
    
    Y = torch.from_numpy(np.complex64(denoised_img))
    Y = Y.permute(2,3,0,1)
    
    Y_x = torch.cat((torch.unsqueeze(Y[:,0,:,:],1),
                     torch.unsqueeze(Y[:,3,:,:],1),
                     torch.unsqueeze(Y[:,6,:,:],1),
                     torch.unsqueeze(Y[:,9,:,:],1),
                     torch.unsqueeze(Y[:,12,:,:],1),
                     torch.unsqueeze(Y[:,15,:,:],1),
                     torch.unsqueeze(Y[:,18,:,:],1)), axis=1)
    
    Y_x = torch.view_as_real(Y_x)
    
    res, mean, std = mean_std_norm_Complex(Y_x)
    
    print("val Y data size:", res.shape)
    
    return (
        res.float(), mean, std
    )

# Denoising functions
def val_dataset_Mag_Y_x_noising(h5_path, N_power):
    f = io.loadmat(h5_path)
    denoised_img = None
    
    if denoised_img is None:
        denoised_img = f.get('denoised_img')
    
    Y = torch.from_numpy(np.complex64(denoised_img))
    Y = Y.permute(2,3,0,1)
    
    Y_x = torch.cat((torch.unsqueeze(Y[:,0,:,:],1),
                     torch.unsqueeze(Y[:,1,:,:],1),
                     torch.unsqueeze(Y[:,4,:,:],1),
                     torch.unsqueeze(Y[:,7,:,:],1),
                     torch.unsqueeze(Y[:,10,:,:],1),
                     torch.unsqueeze(Y[:,13,:,:],1),
                     torch.unsqueeze(Y[:,16,:,:],1)), axis=1)
    
    Y_x = torch.sqrt(torch.pow(torch.real(Y_x),2) + torch.pow(torch.imag(Y_x),2))
    
    res, mean, std = mean_std_norm(Y_x)

    noise = torch.tensor(math.sqrt(N_power) * np.random.randn(res.shape[0],res.shape[1],res.shape[2],res.shape[3]))
    res = res + noise
    
    print("val Y data size:", res.shape)
    
    return (
        res.float(), mean, std
    )

def val_dataset_Mag_Y_y_noising(h5_path, N_power):
    f = io.loadmat(h5_path)
    denoised_img = None
    
    if denoised_img is None:
        denoised_img = f.get('denoised_img')
    
    Y = torch.from_numpy(np.complex64(denoised_img))
    Y = Y.permute(2,3,0,1)
    
    Y_x = torch.cat((torch.unsqueeze(Y[:,0,:,:],1),
                     torch.unsqueeze(Y[:,2,:,:],1),
                     torch.unsqueeze(Y[:,5,:,:],1),
                     torch.unsqueeze(Y[:,8,:,:],1),
                     torch.unsqueeze(Y[:,11,:,:],1),
                     torch.unsqueeze(Y[:,14,:,:],1),
                     torch.unsqueeze(Y[:,17,:,:],1)), axis=1)
    
    Y_x = torch.sqrt(torch.pow(torch.real(Y_x),2) + torch.pow(torch.imag(Y_x),2))
    
    res, mean, std = mean_std_norm(Y_x)

    noise = torch.tensor(math.sqrt(N_power) * np.random.randn(res.shape[0],res.shape[1],res.shape[2],res.shape[3]))
    res = res + noise
    
    print("val Y data size:", res.shape)
    
    return (
        res.float(), mean, std
    )

def val_dataset_Mag_Y_z_noising(h5_path, N_power):
    f = io.loadmat(h5_path)
    denoised_img = None
    
    if denoised_img is None:
        denoised_img = f.get('denoised_img')
    
    Y = torch.from_numpy(np.complex64(denoised_img))
    Y = Y.permute(2,3,0,1)
    
    Y_x = torch.cat((torch.unsqueeze(Y[:,0,:,:],1),
                     torch.unsqueeze(Y[:,3,:,:],1),
                     torch.unsqueeze(Y[:,6,:,:],1),
                     torch.unsqueeze(Y[:,9,:,:],1),
                     torch.unsqueeze(Y[:,12,:,:],1),
                     torch.unsqueeze(Y[:,15,:,:],1),
                     torch.unsqueeze(Y[:,18,:,:],1)), axis=1)
    
    Y_x = torch.sqrt(torch.pow(torch.real(Y_x),2) + torch.pow(torch.imag(Y_x),2))
    
    res, mean, std = mean_std_norm(Y_x)

    noise = torch.tensor(math.sqrt(N_power) * np.random.randn(res.shape[0],res.shape[1],res.shape[2],res.shape[3]))
    res = res + noise
    
    print("val Y data size:", res.shape)
    
    return (
        res.float(), mean, std
    )

def val_dataset_Complex_Y_x_noising(h5_path, N_power):
    f = io.loadmat(h5_path)
    denoised_img = None
    
    if denoised_img is None:
        denoised_img = f.get('denoised_img')
    
    Y = torch.from_numpy(np.complex64(denoised_img))
    Y = Y.permute(2,3,0,1)
    
    Y_x = torch.cat((torch.unsqueeze(Y[:,0,:,:],1),
                     torch.unsqueeze(Y[:,1,:,:],1),
                     torch.unsqueeze(Y[:,4,:,:],1),
                     torch.unsqueeze(Y[:,7,:,:],1),
                     torch.unsqueeze(Y[:,10,:,:],1),
                     torch.unsqueeze(Y[:,13,:,:],1),
                     torch.unsqueeze(Y[:,16,:,:],1)), axis=1)
    
    Y_x = torch.view_as_real(Y_x)

    res, mean, std = mean_std_norm_Complex(Y_x)

    noise = torch.tensor(math.sqrt(N_power) * np.random.randn(res.shape[0],res.shape[1],res.shape[2],res.shape[3],res.shape[4]))
    res = res + noise
    
    print("val Y data size:", res.shape)
    
    return (
        res.float(), mean, std
    )

def val_dataset_Complex_Y_y_noising(h5_path, N_power):
    f = io.loadmat(h5_path)
    denoised_img = None
    
    if denoised_img is None:
        denoised_img = f.get('denoised_img')
    
    Y = torch.from_numpy(np.complex64(denoised_img))
    Y = Y.permute(2,3,0,1)
    
    Y_x = torch.cat((torch.unsqueeze(Y[:,0,:,:],1),
                     torch.unsqueeze(Y[:,2,:,:],1),
                     torch.unsqueeze(Y[:,5,:,:],1),
                     torch.unsqueeze(Y[:,8,:,:],1),
                     torch.unsqueeze(Y[:,11,:,:],1),
                     torch.unsqueeze(Y[:,14,:,:],1),
                     torch.unsqueeze(Y[:,17,:,:],1)), axis=1)
    
    Y_x = torch.view_as_real(Y_x)

    res, mean, std = mean_std_norm_Complex(Y_x)

    noise = torch.tensor(math.sqrt(N_power) * np.random.randn(res.shape[0],res.shape[1],res.shape[2],res.shape[3],res.shape[4]))
    res = res + noise
    
    print("val Y data size:", res.shape)
    
    return (
        res.float(), mean, std
    )

def val_dataset_Complex_Y_z_noising(h5_path, N_power):
    f = io.loadmat(h5_path)
    denoised_img = None
    
    if denoised_img is None:
        denoised_img = f.get('denoised_img')
    
    Y = torch.from_numpy(np.complex64(denoised_img))
    Y = Y.permute(2,3,0,1)
    
    Y_x = torch.cat((torch.unsqueeze(Y[:,0,:,:],1),
                     torch.unsqueeze(Y[:,3,:,:],1),
                     torch.unsqueeze(Y[:,6,:,:],1),
                     torch.unsqueeze(Y[:,9,:,:],1),
                     torch.unsqueeze(Y[:,12,:,:],1),
                     torch.unsqueeze(Y[:,15,:,:],1),
                     torch.unsqueeze(Y[:,18,:,:],1)), axis=1)
    
    Y_x = torch.view_as_real(Y_x)

    res, mean, std = mean_std_norm_Complex(Y_x)

    noise = torch.tensor(math.sqrt(N_power) * np.random.randn(res.shape[0],res.shape[1],res.shape[2],res.shape[3],res.shape[4]))
    res = res + noise
    
    print("val Y data size:", res.shape)
    
    return (
        res.float(), mean, std
    )

# Noise estimation
def N_estimator(image):
        LR_denoised_image = []
        for i in range(image.shape[0]):
            slice_data = image[i,:,:,:]
            tmp = LR_denoising(slice_data, rank=3, patch=(9,9), step=4)
            LR_denoised_image.append(tmp)
        LR_denoised_image = np.stack(LR_denoised_image, axis=0)
        LR_denoised_image = torch.from_numpy(LR_denoised_image)

        N_power = abs(torch.var(LR_denoised_image) - torch.var(image))

        return N_power

def N_Complex_estimator(image):
        LR_denoised_image = []
        for i in range(image.shape[0]):
            slice_data = image[i,:,:,:,:]
            tmp = LR_denoising(slice_data, rank=3, patch=(9,9), step=4)
            LR_denoised_image.append(tmp)
        LR_denoised_image = np.stack(LR_denoised_image, axis=0)
        LR_denoised_image = torch.from_numpy(LR_denoised_image)

        N_R_power = abs(torch.var(LR_denoised_image[:,:,:,:,0]) - torch.var(image[:,:,:,:,0]))
        N_I_power = abs(torch.var(LR_denoised_image[:,:,:,:,1]) - torch.var(image[:,:,:,:,1]))

        return N_R_power, N_I_power

def LR_denoising(noisy_data, rank=None, mp_var=None, patch=None, step=1):
    noisy_data = (noisy_data.cpu()).numpy()
    truncated_data = np.zeros_like(noisy_data)
    if patch:
        # This performs locally low rank
        if rank is None or isinstance(rank, int):
            def svd_t(x):
                return svd_trunc(x, r=rank)
        elif isinstance(rank, str) and rank.lower() == 'mp'\
                and mp_var is not None:
            mp_shape = (np.prod(tuple(patch) + noisy_data.shape[2:-1]),
                        noisy_data.shape[-1])
            mp_lim = max_sv_from_mp(mp_var, mp_shape)

            def svd_t(x):
                return svd_trunc(x, sv_lim=mp_lim)
        else:
            raise TypeError('rank parameter must be positive integer or "mp".'
                            ' If "mp" then mp_var must be specified.')

        data_shape = noisy_data.shape[:2]
        M = np.full(data_shape, False)
        M[::step, ::step] = True
        ii, jj = np.where(M)

        idx_i = np.arange(0, np.min((patch[0], data_shape[0])))
        idx_j = np.arange(0, np.min((patch[1], data_shape[1])))

        navg = np.zeros((len(ii),) + data_shape)
        covar_out = []
        variance_patches = np.zeros((len(ii),) + noisy_data.shape)
        for count, (idx, jdx) in enumerate(zip(ii, jj)):
            i = np.mod(idx+idx_i-1, data_shape[0])
            j = np.mod(jdx+idx_j-1, data_shape[1])
            denoised, var, cvar = svd_t(noisy_data[np.ix_(i, j)])
            truncated_data[np.ix_(i, j)] += denoised
            variance_patches[np.ix_([count], i, j)] = var
            covar_out.append(cvar)
            navg[np.ix_([count], i, j)] += 1

        navg_all = np.sum(navg, axis=0)
        # Deal with broadcasting comparing along trailing dimension
        truncated_data = (truncated_data.T / navg_all.T).T

    else:
        # Performs global low rank
        if rank is None or isinstance(rank, int):
            truncated_data, variance, covar = svd_trunc(noisy_data, r=rank)
        elif isinstance(rank, str) and rank.lower() == 'mp'\
                and mp_var is not None:
            mp_shape = (np.prod(noisy_data.shape[:-1]), noisy_data.shape[-1])
            mp_lim = max_sv_from_mp(mp_var, mp_shape)
            truncated_data, variance, covar = svd_trunc(noisy_data,
                                                        sv_lim=mp_lim)
        else:
            raise TypeError('rank parameter must be positive integer or "mp".'
                            ' If "mp" then mp_var must be specified.')

    return truncated_data

def lsvd(A, r=None, sv_lim=None):
    # Quick checks
    if r is not None and r < 1:
        raise ValueError('Rank must be > 0')
    if sv_lim is not None and sv_lim < 0.0:
        raise ValueError('sv_lim must be > 0.0')

    if A.shape[1] > A.shape[0]:
        swap = True
        A = A.conj().T
    else:
        swap = False

    D, V = np.linalg.eigh(A.conj().T @ A)
    D[D < 0.0] = 1E-15
    S = D**0.5

    if r is None and sv_lim is None:
        r = A.shape[1]
    elif r is None and sv_lim > 0.0:
        r = len(S) - np.searchsorted(S, sv_lim)
        if r == 0:
            r = 1

    S = S[-r:]
    V = V[:, -r:]
    U = A @ (V @ np.diag(1 / S))
    S = np.diag(S)

    if swap:
        return V, S, U
    else:
        return U, S, V

def svd_trunc(A, r=None, sv_lim=None):
    dims = A.shape
    u, s, v = lsvd(A.reshape((-1, dims[-1])), r=r, sv_lim=sv_lim)

    est_var = calc_var(u, v)
    est_covar = calc_covar(v)

    return (u @ s @ v.conj().T).reshape(dims), est_var.reshape(dims), est_covar

def calc_var(u, v):
    return (np.linalg.norm(u, axis=1)**2)[:, np.newaxis]\
        + np.linalg.norm(v, axis=1)**2

def calc_covar(u):
    return u @ u.conj().T

def max_sv_from_mp(data_var, data_shape):
    if data_shape[1] > data_shape[0]:
        data_shape = (data_shape[1], data_shape[0])

    c = data_shape[1] / data_shape[0]
    sv_lim = data_var * (1 + np.sqrt(c))**2
    return (sv_lim * data_shape[0])**0.5






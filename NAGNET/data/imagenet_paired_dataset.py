import re
from skimage import transform, data
import imageio
from PIL import Image
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import matplotlib.pyplot as plt
from basicsr.data.data_util import paths_from_lmdb, scandir
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import imresize, rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY
import pydicom

import cv2
import numpy as np
import os
import random
import monai
import skimage.transform, skimage.io
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from einops import rearrange
from pathlib import Path
from bm3d import bm3d
from scipy.ndimage import zoom

import sys

from noise_compact import noisecomplication,noisecomplication_iter
# from clip_get_feature import token_get

def imshow(tensor):

    image = tensor.detach().cpu().numpy()

    image = image.transpose(1, 2, 0)

    image = (image * 0.5) + 0.5
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()



def get_all_files_oswalk(root_dir):

    file_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            file_paths.append(full_path)
    return file_paths
from pathlib import Path
from typing import List


def natural_sort_key(s):

    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]


def get_png_files(folder_path: str) -> List[Path]:

    path = Path(folder_path)

    # Check if the path exists and is a directory
    if not path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Specified path is not a folder: {folder_path}")


    png_files = []
    for file in path.iterdir():
        if file.is_file() and file.suffix.lower() == '.png':

            windows_path = file.resolve()
            png_files.append(windows_path)


    png_files.sort(key=lambda x: natural_sort_key(x.name))

    return png_files


def get_grandparent_dir_name(file_path):
    path = Path(file_path).resolve()  # Resolve to absolute path and normalize
    try:
        # Get parent directory twice (go up two levels)
        return path.parent.parent.name
    except AttributeError:
        # Return empty string if the path hierarchy is insufficient (e.g., root directory)
        return ""

# def random_blur(image,min_kernel_size=3,max_kernel_size=5):
#     kernel_size=random.choice(range(min_kernel_size,max_kernel_size+1,2))
#     blurrred_image=cv2.GaussianBlur(image,(kernel_size,kernel_size),0)
#     return  blurrred_image
def downsample_keep_size(arr, factor=0.5):
    """
    Downsample the input NumPy array while keeping the original size.

    Parameters:
    arr (numpy.ndarray): Input NumPy array.
    factor (float): Downsampling factor, default is 0.5 (i.e., downsample to half).

    Returns:
    numpy.ndarray: Downsampled array resized back to the original size.
    """
    # Downsample
    downsampled = zoom(arr, factor)

    # Calculate the scaling factor to restore original size
    resize_factor = tuple(s_orig / s_down for s_orig, s_down in zip(arr.shape, downsampled.shape))

    # Resize back to original size
    resized = zoom(downsampled, resize_factor)

    return resized
def random_blur(image, min_kernel_size=3, max_kernel_size=5, min_sigma=0.5, max_sigma=1.5):
    # Randomly select kernel size
    kernel_size = random.choice(range(min_kernel_size, max_kernel_size + 1, 2))

    # Randomly select standard deviation
    sigma = random.uniform(min_sigma, max_sigma)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    return blurred_image

def normalize(x, x_min, x_max):
    """
    Normalize a number to the range [0, 1].

    :param x: Number to be normalized.
    :param x_min: Minimum value of the data.
    :param x_max: Maximum value of the data.
    :return: Normalized number.
    """
    if x_min == x_max:
        raise ValueError("Minimum and maximum values cannot be equal, otherwise normalization cannot be performed.")

    normalized_x = (x - x_min) / (x_max - x_min)
    return normalized_x

def random_crop_cv2(image, crop_size=(128, 128)):
    """
    Randomly crop a patch of a specified size from an image using OpenCV.

    :param image: Input image as ndarray.
    :param crop_size: Size of the crop patch, default is (128, 128).
    :return: Cropped patch.
    """
    r=image.shape
    height, width = image.shape[:2]
    crop_height, crop_width = crop_size

    # Ensure the crop area is within the image boundaries
    if crop_height > height or crop_width > width:
        raise ValueError("Crop size is larger than the image size.")

    # Randomly select the top-left corner of the crop area
    y = np.random.randint(0, height - crop_height + 1)
    x = np.random.randint(0, width - crop_width + 1)

    # Crop the image
    cropped_image = image[y:y + crop_height, x:x + crop_width]
    return cropped_image

def split_image(image, slice_width, slice_height, overlap):
    """
    Slice the image into overlapping patches.
    """
    slices = []
    original_height, original_width = image.shape

    for y in range(0, original_height - slice_height + 1, slice_height - overlap):
        for x in range(0, original_width - slice_width + 1, slice_width - overlap):
            slice = image[y:y + slice_height, x:x + slice_width]
            slice=torch.from_numpy(slice)
            slice=slice.unsqueeze(0).to(torch.float32)
            slices.append(slice)

    return slices




def add_random_blur(image, min_kernel_size=3, max_kernel_size=5):
    # Randomly select the blur kernel size
    kernel_size = np.random.randint(min_kernel_size, max_kernel_size + 1)
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image

def crop_center(img, crop_size):
    h, w = img.shape[:2]
    start_h = (h - crop_size[0]) // 2
    start_w = (w - crop_size[1]) // 2
    return img[start_h:start_h + crop_size[0], start_w:start_w + crop_size[1]]


@DATASET_REGISTRY.register()
class ImageNetPairedDataset(data.Dataset):

    def __init__(self, opt):
        super(ImageNetPairedDataset, self).__init__()
        self.opt=opt

        self.hr_paths = self.opt['dataroot_gt']
        self.lr_paths = self.opt['dataroot_lq']
        self.phase = self.opt['phase']
        self.input_width, self.input_height = 256,256
        self.scale = 2
        self.repeat = 1
        self.value_range = 255
        limit=True

        self.paths = []
        if self.phase=='train':
            self.data_num = 6000
        else:
            self.data_num=1
        self.hr_list = []
        # for hr_paths in self.hr_paths:
        self.scan_dir(Path(self.hr_paths))
        if limit == True:
            # self.hr_list = self.hr_list[:self.data_num]
            # self.hr_list = self.hr_list[0:self.data_num]
            self.hr_list = random.sample(self.hr_list, self.data_num)

        self.data_len = len(self.hr_list)
        self.lr_list = [None] * len(self.hr_list)
        self.full_len = self.data_len * self.repeat

    def __len__(self):
        return self.full_len

    def __getitem__(self, index):
        idx = index % self.data_len
        url_hr, name = self.hr_list[idx]
        file_name=os.path.basename(url_hr)
        lrfile_path=os.path.join(self.lr_paths,file_name)
        # print(url_hr)
        if url_hr.suffix == ".npz":
            img_hr = np.load(url_hr)['arr_0']
        elif url_hr.suffix == ".png":
            img_hr = imageio.imread(url_hr, mode='L')
            img_hr=cv2.resize(img_hr,(256,256))
            if self.phase=='val':
                img_lr = imageio.imread(lrfile_path, mode='L')
                min_val = np.min(img_lr)
                max_val = np.max(img_lr)
                img_lr1 = (img_lr - min_val) / (max_val - min_val)
        elif url_hr.suffix == ".dcm":
            img_hr = pydicom.dcmread(url_hr)
            dataacuq = img_hr.AcquisitionMatrix[0]
            img_hr = img_hr.pixel_array
            if self.phase=='val':
                img_lr = pydicom.dcmread(url_hr)
                dataacuq = img_lr.AcquisitionMatrix[0]
                img_lr = img_lr.pixel_array
                min_val = np.min(img_lr)
                max_val = np.max(img_lr)
                # Normalize to 0-1
                img_lr1 = (img_lr - min_val) / (max_val - min_val)
                img_lr1

        min_val = np.min(img_hr)
        max_val = np.max(img_hr)

        # Normalize to 0-1
        img_hr = (img_hr - min_val) / (max_val - min_val)




            # img_hr = np.load(url_hr)
        # if self.phase == 'train' and img_hr.shape[0] == 512:
        # img_lr = skimage.transform.resize(img_hr, (256, 256))
        # if self.phase == 'train'or self.phase=='val':
        if self.phase == 'train':
            img_hr = img_hr
            # img_hr = random_crop_cv2(img_hr, (256, 256))
            # img_hr = self.mixup(img_hr)
            img_hr=cv2.resize(img_hr,(256,256))
            # img_hr = cv2.resize(img_hr, (512, 512))
            img_lr ,noise= self.down_sample(img_hr, idx,False)
            noise=torch.tensor(noise)
        elif self.phase=='val':

            # # img_hr=cv2.resize(img_hr,(448,448))
            # img_hr=cv2.resize(img_hr,(512,512))
            # hr_patches = split_image(img_hr, 512, 512, 128)
            # img_lr,noise=self.down_sample(img_hr,idx)
            #
            # img_lr=img_lr1
            # img_lr=cv2.resize(img_lr,(448,448))
            #
            #
            # lr_patches=split_image(img_lr,256,256,64)
            # hrshape=img_hr.shape
            # lrshape=img_lr.shape
            # # noise_patches=split_image(noise,128,128,32)
            # noise_patches=[noise,noise,noise,noise]
            #
            # # img_hr=cv2.resize(img_hr,(448,448))
            img_hr = cv2.resize(img_hr, (512, 512))

            img_lr, noise = self.down_sample(img_hr, idx,False)

            img_lr = img_lr1
            img_lr = cv2.resize(img_lr, (256, 256))
            img_lr=np.asarray(img_lr, dtype=np.float64)
            img_hr = np.asarray(img_hr, dtype=np.float64)


            hrshape = img_hr.shape
            lrshape = img_lr.shape
            # noise_patches=split_image(noise,128,128,32)
            noise_patches = [noise, noise, noise, noise]
            noise_patches=noise
            img_lr = img_lr.astype(np.float32)
            img_hr = img_hr.astype(np.float32)
            img_lr = torch.from_numpy(img_lr).float()
            img_hr = torch.from_numpy(img_hr).float()
            img_lr=img_lr.unsqueeze(0)
            img_hr=img_hr.unsqueeze(0)


            # noise= transform.resize(noise, (256,256), anti_aliasing=True)
        # elif self.phase=='val':
        #     lr_path=r'F:\val\noise'+name
        #     img_lr = np.array(Image.open(lr_path).convert('L'))
        #     img_lr=cv2.resize(img_lr,(224,224))
        #
        #     img_hr=cv2.resize(img_hr,(448,448))




        if self.phase == 'test':
            img_lr = self.down_sample(img_hr, idx,'ncx2')
        # img_hr = skimage.transform.resize(img_hr, (512, 512))

        #
        # if self.phase == 'train' or self.phase == 'val':
        if self.phase == 'train' :
            h, w = img_lr.shape
            s = self.scale
            # random cropping
            if self.input_height < h:
                y = random.randint(0, h - self.input_height)
                x = random.randint(0, w - self.input_width)
                img_lr = img_lr[y: y + self.input_height, x: x + self.input_width]
                img_hr = img_hr[y * s: (y + self.input_height) * s,
                         x * s: (x + self.input_width) * s]

            img_lr = np.ascontiguousarray(img_lr, dtype=np.float32)
            img_hr = np.ascontiguousarray(img_hr, dtype=np.float32)
            img_lr = np.array(img_lr)
            img_hr = np.array(img_hr)
            # horizontal flip


        # BGR to RGB, HWC to CHW, uint8 to float32
            img_lr = rearrange(img_lr, 'h w -> 1 h w').astype(np.float32)
            img_hr = rearrange(img_hr, 'h w -> 1 h w').astype(np.float32)


            # numpy array to tensor, [0, 255] to [0, 1]
            img_lr = torch.from_numpy(img_lr).float()
            img_hr = torch.from_numpy(img_hr).float()
            # noise=torch.from_numpy(noise).float()
            data = {'lq': img_lr, 'gt': img_hr, 'noise': noise.to(torch.float) , 'gt_path': str(url_hr), 'lq_path': str(url_hr)}
        elif self.phase=='val':
            # data = {'lq': lr_patches, 'gt': hr_patches,'noise': noise_patches, 'gt_path': str(index),'lq_path': str(url_hr), 'hrshape': hrshape, 'lrshape': lrshape}
            data = {'lq': img_lr, 'gt': img_hr, 'noise': noise_patches, 'gt_path': str(index),
                    'lq_path': str(url_hr), 'hrshape': hrshape, 'lrshape': lrshape}

        return data

    def get_bm3d(self, lr, sigma=0.1):
        denoise = bm3d(lr, sigma)
        return denoise

    def down_sample(self, hr, idx, laician, factor=0.35):
        # ... the downsampling part before remains the same ...
        lr = hr
        lr = skimage.transform.downscale_local_mean(lr, (2, 2))
        lrs = lr  # preserve the clean downsampled image

        if self.phase == 'train' or self.phase == 'val':
            # ----------------- New branch: non-central chi-square noise (multi-channel RSS) -----------------
            if laician == 'ncx2':  # use string identifier, compatible with the original bool parameter
                # Configurable parameters
                n_channels = 4  # number of simulated receiver channels
                rel_std_range = (0.02, 0.10)  # relative noise standard deviation range (relative to max image magnitude)
                rel_std = np.random.uniform(*rel_std_range)
                # Dynamically compute absolute noise standard deviation based on the image
                signal_max = np.max(lr)  # or use np.mean(lr) etc.
                abs_std = rel_std * signal_max

                # Generate L complex channels, with signal evenly distributed per channel (total power conserved)
                channels = []
                signal_per_channel = lr / np.sqrt(n_channels)  # magnitude allocation, phase set to 0 (real axis)
                for _ in range(n_channels):
                    real = signal_per_channel + np.random.normal(0, abs_std, lr.shape)
                    imag = np.random.normal(0, abs_std, lr.shape)
                    channels.append(real + 1j * imag)

                # root-sum-of-squares combined magnitude map -> follows a non-central chi-squared distribution (degrees of freedom = 2L)
                lr_noisy = np.sqrt(np.sum([np.abs(ch) ** 2 for ch in channels], axis=0))

                # Compute residual (clean image - noisy image), consistent with other branches
                noise = lrs - lr_noisy
                lr = lr_noisy

                # Placeholder noise parameters (may be required by the original interface, but does not affect the return)
                stdre = torch.tensor(abs_std)
                stdim = torch.tensor(0.0)

            # ----------------- Original Rician branch (Boolean True) -----------------
            elif laician is True:
                # ... original Rician noise addition code remains unchanged ...
                std = np.random.uniform(0.05, 0.20)
                noise_transform = monai.transforms.RandRicianNoise(
                    prob=1.0, mean=0.0, std=std, relative=True,
                    channel_wise=True, sample_std=False
                )
                noise_seed = np.random.randint(10000)
                noise_transform.set_random_state(noise_seed)
                lr_noisy = noise_transform(lr)
                noise = lrs - lr_noisy
                lr = lr_noisy
                stdre = torch.tensor(normalize(std, lr.min(), lr.max()))
                stdim = torch.tensor(0.0)

            # ----------------- Original k-space Gaussian noise branch (Boolean False) -----------------
            else:
                # ... original k-space noise addition + center truncation code remains unchanged ...
                kspace = np.fft.fftshift(np.fft.fft2(lr, axes=(0, 1)), axes=(0, 1))
                kspace_real = np.real(kspace)
                kspace_imag = np.imag(kspace)
                realmax = kspace_real.max()
                realmin = kspace_real.min()
                imamax = kspace_imag.max()
                imamin = kspace_imag.min()
                a = random.uniform(0.8,1.25)
                std1 = np.random.uniform(0.001 * realmax, 0.006 * realmax)
                std2 = np.random.uniform(0.001 * imamax, 0.04 * imamax)
                noise_real = np.random.normal(0, std1, kspace_real.shape)
                noise_imag = np.random.normal(0, std2, kspace_imag.shape)
                kspace_real_noisy = kspace_real + noise_real
                kspace_imag_noisy = kspace_imag + noise_imag
                kspace = kspace_real_noisy + 1j * kspace_imag_noisy
                mask = np.zeros_like(kspace)
                h, w = kspace.shape
                h_offset = int(h * factor)
                w_offset = int(w * factor)
                mask[h // 2 - h_offset:h // 2 + h_offset, w // 2 - w_offset:w // 2 + w_offset] = 1
                kspace *= mask
                lr = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace, axes=(0, 1)), axes=(0, 1)))
                noise = lrs - lr
                # stdre, stdim were commented out in the original code, kept as comments

            return lr, noise
    def select_random_subset(lst, percentage):
        # Calculate the number of elements to extract
        subset_size = int(len(lst) * percentage)

        # Randomly sample the specified number of elements
        subset = random.sample(lst, subset_size)

        return subset

    def add_file(self, dir: Path):
        parent_name = dir.parent.stem + "_" + dir.stem
        for file in dir.iterdir():
            self.hr_list.append((file, parent_name + "_" + file.stem))

    def scan_dir(self, dir: Path):
        for p in dir.iterdir():
            if p.is_file():
                self.add_file(p.parent)
                break
            else:
                self.scan_dir(p)

    def mixup(self, img):
        index = random.randint(0, self.data_len - 1)
        idx = index % self.data_len
        url_hr, name = self.hr_list[idx]
        # print(url_hr)
        if url_hr.suffix == ".npz":
            img_hr = np.load(url_hr)['arr_0']
        else:
            img_hr = np.load(url_hr)
        # if self.phase == 'train' and img_hr.shape[0] == 512:
        img_hr = skimage.transform.resize(img_hr, (256, 256))
        alpha = 1
        weight = np.random.beta(alpha, alpha)
        return weight * img + (1 - weight) * img_hr


@DATASET_REGISTRY.register()
class ImageNetPairedDatasetp2p(data.Dataset):

    def __init__(self, opt):
        super(ImageNetPairedDatasetp2p, self).__init__()
        self.opt = opt

        self.hr_paths = self.opt['dataroot_gt']
        self.lr_paths = self.opt['dataroot_lq']
        self.phase = self.opt['phase']
        self.input_width, self.input_height = 256, 256
        self.scale = 2
        self.repeat = 1
        self.value_range = 255
        limit = True

        self.paths = []
        if self.phase == 'train':
            self.data_num = 8000
        else:
            self.data_num = 30
        self.hr_list = []
        # for hr_paths in self.hr_paths:
        self.scan_dir(Path(self.hr_paths))
        if limit == True:
            # self.hr_list = self.hr_list[:self.data_num]
            # self.hr_list = self.hr_list[0:self.data_num]
            self.hr_list = random.sample(self.hr_list, self.data_num)

        self.data_len = len(self.hr_list)
        self.lr_list = [None] * len(self.hr_list)
        self.full_len = self.data_len * self.repeat

    def __len__(self):
        return self.full_len

    def __getitem__(self, index):
        idx = index % self.data_len
        url_hr, name = self.hr_list[idx]
        file_name = os.path.basename(url_hr)
        lrfile_path = os.path.join(self.lr_paths, file_name)
        # print(url_hr)
        if url_hr.suffix == ".npz":
            img_hr = np.load(url_hr)['arr_0']
        elif url_hr.suffix == ".png":
            img_hr = imageio.imread(url_hr, mode='L')
            img_hr = cv2.resize(img_hr, (256, 256))
            img_lr= imageio.imread(lrfile_path, mode='L')
            min_val = np.min(img_lr)
            max_val = np.max(img_lr)
            img_lr = (img_lr - min_val) / (max_val - min_val)
            if self.phase == 'val':
                img_lr = imageio.imread(lrfile_path, mode='L')
                min_val = np.min(img_lr)
                max_val = np.max(img_lr)
                img_lr = (img_lr - min_val) / (max_val - min_val)
        elif url_hr.suffix == ".dcm":
            img_hr = pydicom.dcmread(url_hr)
            dataacuq = img_hr.AcquisitionMatrix[0]
            img_hr = img_hr.pixel_array
            if self.phase == 'val':
                img_lr = pydicom.dcmread(url_hr)
                dataacuq = img_lr.AcquisitionMatrix[0]
                img_lr = img_lr.pixel_array
                min_val = np.min(img_lr)
                max_val = np.max(img_lr)
                # Normalize to 0-1
                img_lr = (img_lr - min_val) / (max_val - min_val)


        min_val = np.min(img_hr)
        max_val = np.max(img_hr)

        # Normalize to 0-1
        img_hr = (img_hr - min_val) / (max_val - min_val)


        if self.phase == 'train':
            img_hr = img_hr
            # img_hr = random_crop_cv2(img_hr, (256, 256))
            # img_hr = self.mixup(img_hr)
            img_hr = cv2.resize(img_hr, (256, 256))
            img_lr = cv2.resize(img_lr, (128, 128))
            # noise=img_hr-img_lr
            # img_hr = cv2.resize(img_hr, (512, 512))
            # img_lr2, noise = self.down_sample(img_hr, idx)
            noise = torch.tensor(img_lr)
        elif self.phase == 'val':


            img_hr = cv2.resize(img_hr, (512, 512))

            # img_lr, noise = self.down_sample(img_hr, idx)

            # img_lr = img_lr1
            img_lr = cv2.resize(img_lr, (256, 256))
            img_lr = np.asarray(img_lr, dtype=np.float64)
            img_hr = np.asarray(img_hr, dtype=np.float64)

            hrshape = img_hr.shape
            lrshape = img_lr.shape
            # noise_patches=split_image(noise,128,128,32)

            img_lr = img_lr.astype(np.float32)
            img_hr = img_hr.astype(np.float32)
            img_lr = torch.from_numpy(img_lr).float()
            img_hr = torch.from_numpy(img_hr).float()
            img_lr = img_lr.unsqueeze(0)
            img_hr = img_hr.unsqueeze(0)



        if self.phase == 'test':
            hrshape = img_hr.shape
            lrshape = img_lr.shape
            # img_lr = self.down_sample(img_hr, idx)
            img_lr=torch.from_numpy(img_lr).float()
            img_hr=torch.from_numpy(img_hr).float()

        # img_hr = skimage.transform.resize(img_hr, (512, 512))

        #
        # if self.phase == 'train' or self.phase == 'val':
        if self.phase == 'train':

            #
            # # numpy array to tensor, [0, 255] to [0, 1]
            img_lr = torch.from_numpy(img_lr).float()
            img_hr = torch.from_numpy(img_hr).float()
            # noise=torch.from_numpy(noise).float()
            data = {'lq': img_lr, 'gt': img_hr, 'noise': noise.to(torch.float), 'gt_path': str(url_hr),
                    'lq_path': str(url_hr)}
        elif self.phase == 'val':
            # data = {'lq': lr_patches, 'gt': hr_patches,'noise': noise_patches, 'gt_path': str(index),'lq_path': str(url_hr), 'hrshape': hrshape, 'lrshape': lrshape}
            data = {'lq': img_lr, 'gt': img_hr, 'noise': img_lr, 'gt_path': str(index),
                    'lq_path': str(url_hr), 'hrshape': hrshape, 'lrshape': lrshape}
        else:
            data = {'lq': img_lr, 'gt': img_hr, 'noise': noise.to(torch.float), 'gt_path': str(url_hr),
                    'lq_path': str(url_hr), 'hrshape': hrshape, 'lrshape': lrshape}

        return data

    def get_bm3d(self, lr, sigma=0.1):
        denoise = bm3d(lr, sigma)
        return denoise

    def down_sample(self, hr, idx, factor=0.35):
        lr = hr
        lr = skimage.transform.downscale_local_mean(lr, (2, 2))
        # lr=downsample_keep_size(lr,factor=0.5)
        lrs = lr

        if self.phase == 'train' or self.phase == 'val':
            kspace = np.fft.fftshift(np.fft.fft2(lr, axes=(0, 1)), axes=(0, 1))

            kspace_real = np.real(kspace)
            kspace_imag = np.imag(kspace)

            realmax = kspace_real.max()
            realmin = kspace_real.min()
            imamax = kspace_imag.max()
            imamin = kspace_imag.min()

            # realhigh=0.0005 * realmax
            # reallow=0.00001 * realmax
            # imahigh=0.01 * imamax
            # imalow=0.0001 * imamax

            # Randomly generate standard deviations
            std1 = np.random.uniform(0.001 * realmax, 0.006 * realmax)
            std2 = np.random.uniform(0.001 * imamax, 0.04 * imamax)
            # #zhong
            # std1 = np.random.uniform(0.0001 * realmax, 0.003 * realmax)
            # std2 = np.random.uniform(0.001 * imamax, 0.02 * imamax)
            # di
            # std1 = np.random.uniform(0.00001 * realmax, 0.005 * realmax)
            # std2 = np.random.uniform(0.0001 * imamax, 0.008 * imamax)

            # Add Gaussian noise to the real and imaginary parts
            noise_real = np.random.normal(0, std1, kspace_real.shape)
            noise_imag = np.random.normal(0, std2, kspace_imag.shape)

            # Add noise to the real and imaginary parts of k-space
            kspace_real_noisy = kspace_real + noise_real
            kspace_imag_noisy = kspace_imag + noise_imag

            # Recombine the real and imaginary parts into complex k-space
            kspace = kspace_real_noisy + 1j * kspace_imag_noisy

            mask = np.zeros_like(kspace)
            h, w = kspace.shape
            h_offset = int(h * factor)
            w_offset = int(w * factor)

            mask[h // 2 - h_offset:h // 2 + h_offset, w // 2 - w_offset:w // 2 + w_offset] = 1
            kspace *= mask
            lr = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace, axes=(0, 1)), axes=(0, 1)))
            stdre = torch.tensor(normalize(std1, realmin, realmax))
            stdim = torch.tensor(normalize(std2, imamin, imamax))
            # noise=[stdre,stdim]
            # lr=random_blur(lr)

            # kspace = np.fft.fftshift(np.fft.fft2(lr, axes=(0, 1)), axes=(0, 1))
            # h, w = kspace.shape
            # mask = np.zeros_like(kspace)
            # h_offset = int(h * factor)
            # w_offset = int(w * factor)
            # mask[h // 2 - h_offset:h // 2 + h_offset, w // 2 - w_offset:w // 2 + w_offset] = 1
            # kspace = kspace * mask
            # lr = np.fft.ifft2(np.fft.ifftshift(kspace, axes=(0, 1)), axes=(0, 1))
            # lr = np.abs(lr)
            # # lr=add_random_blur(lr)
            #
            #
            # # randomly generate standard deviation
            # std = np.random.uniform(0.2, 0.8)
            #
            # noise = monai.transforms.RandRicianNoise(prob=1.0, mean=0, std=std, relative=True, channel_wise=True,
            #                                          sample_std=False)
            # noise_seed = np.random.randint(10000)
            # noise.set_random_state(noise_seed)
            # lr = noise(lr)
            noise = lrs - lr
            # lr=lr.numpy()

            # blur

        return lr, noise

    def select_random_subset(lst, percentage):
        # Calculate the number of elements to extract
        subset_size = int(len(lst) * percentage)

        # Randomly sample the specified number of elements
        subset = random.sample(lst, subset_size)

        return subset

    def add_file(self, dir: Path):
        parent_name = dir.parent.stem + "_" + dir.stem
        for file in dir.iterdir():
            self.hr_list.append((file, parent_name + "_" + file.stem))

    def scan_dir(self, dir: Path):
        for p in dir.iterdir():
            if p.is_file():
                self.add_file(p.parent)
                break
            else:
                self.scan_dir(p)

    def mixup(self, img):
        index = random.randint(0, self.data_len - 1)
        idx = index % self.data_len
        url_hr, name = self.hr_list[idx]
        # print(url_hr)
        if url_hr.suffix == ".npz":
            img_hr = np.load(url_hr)['arr_0']
        else:
            img_hr = np.load(url_hr)
        # if self.phase == 'train' and img_hr.shape[0] == 512:
        img_hr = skimage.transform.resize(img_hr, (256, 256))
        alpha = 1
        weight = np.random.beta(alpha, alpha)
        return weight * img + (1 - weight) * img_hr

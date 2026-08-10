# synthesis/generate_synthetic.py
import os
import argparse
import numpy as np
import glob
from tqdm import tqdm
from simulation import estimate_noise_from_kspace, synthesize_lowfield_from_rawkspace

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic low-field MRI data.")
    parser.add_argument('--high_field_dir', required=True, help='Folder containing high-field (1.5T) k-space .npy files')
    parser.add_argument('--real_ref_dir', required=True, help='Folder containing real low-field (0.1T) k-space .npy files for noise estimation')
    parser.add_argument('--output_dir', required=True, help='Output folder for synthetic images and noise maps')
    parser.add_argument('--num_slices', type=int, default=24000, help='Number of slices to generate')
    parser.add_argument('--scale', type=float, default=0.5, help='Down-sampling scale factor')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Estimate noise from real low-field reference
    ref_files = glob.glob(os.path.join(args.real_ref_dir, '*.npy'))
    if not ref_files:
        raise FileNotFoundError(f"No .npy files found in {args.real_ref_dir}")

    print(f"Estimating noise from {len(ref_files)} real reference slices...")
    sigma_re_list, sigma_im_list = [], []
    for f in ref_files[:100]:  # use first 100 for speed, or remove limit for full
        k = np.load(f)
        s_re, s_im = estimate_noise_from_kspace(k)
        sigma_re_list.append(s_re)
        sigma_im_list.append(s_im)

    sigma_re = np.mean(sigma_re_list)
    sigma_im = np.mean(sigma_im_list)
    print(f"Estimated noise: Re={sigma_re:.4f}, Im={sigma_im:.4f}")

    # 2. Loop over high-field files
    high_files = glob.glob(os.path.join(args.high_field_dir, '*.npy'))
    if not high_files:
        raise FileNotFoundError(f"No .npy files found in {args.high_field_dir}")

    print(f"Generating {min(args.num_slices, len(high_files))} synthetic slices...")
    for i, f in enumerate(tqdm(high_files[:args.num_slices])):
        kspace_high = np.load(f)
        low_img, epsilon_gt = synthesize_lowfield_from_rawkspace(
            kspace_high=kspace_high,
            sigma_re=sigma_re,
            sigma_im=sigma_im,
            beta_range=(0.5, 1.5),
            scale_factor=args.scale,
            random_state=i  # deterministic per slice
        )
        base = os.path.basename(f).replace('.npy', '')
        np.save(os.path.join(args.output_dir, f'{base}_low.npy'), low_img)
        np.save(os.path.join(args.output_dir, f'{base}_noise.npy'), epsilon_gt)

    print("Done!")

if __name__ == '__main__':
    main()

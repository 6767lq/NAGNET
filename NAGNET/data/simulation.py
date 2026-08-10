import numpy as np
from scipy.ndimage import zoom

def estimate_noise_from_kspace(kspace: np.ndarray, ratio: float = 0.7):
    """
    Estimate noise standard deviation from the high‑frequency annular region in k‑space
    as described in Fig. 2(b) of the paper.

    Parameters:
        kspace : complex numpy array (H, W), raw k‑space data.
        ratio : inner radius ratio, default 0.7 (i.e., annulus from 0.7*r0 to r0).
    Returns:
        sigma_re, sigma_im : standard deviations of the real and imaginary parts.
    """
    H, W = kspace.shape
    r0 = min(H, W) // 2          # distance from center to the shorter side
    ri = int(r0 * ratio)

    # Build coordinate grid (center at (0,0))
    y, x = np.ogrid[-H//2:H//2, -W//2:W//2]
    radius = np.sqrt(x**2 + y**2)

    # High‑frequency annular mask (ri <= r <= r0)
    mask = (radius >= ri) & (radius <= r0)

    # Extract real and imaginary values from the annular region
    re_vals = np.real(kspace)[mask]
    im_vals = np.imag(kspace)[mask]

    # Noise in k‑space is typically zero‑mean Gaussian
    sigma_re = np.std(re_vals)
    sigma_im = np.std(im_vals)

    return sigma_re, sigma_im


def synthesize_lowfield_from_rawkspace(
    kspace_high: np.ndarray,          # high‑field raw complex k‑space (H, W)
    sigma_re: float,                  # estimated real‑part noise std from real low‑field data
    sigma_im: float,                  # estimated imaginary‑part noise std from real low‑field data
    beta_range: tuple = (0.5, 1.5),   # random scaling range for beta coefficients
    scale_factor: float = 0.5,        # down‑sampling factor (to match low‑field resolution)
    random_state: int = None
):
    """
    Synthesize low‑field data by adding independent complex Gaussian noise in k‑space
    and then down‑sampling in the image domain, as per Fig. 2(c) of the paper.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # 1. Randomly sample beta coefficients
    beta_r = np.random.uniform(*beta_range)
    beta_i = np.random.uniform(*beta_range)

    # 2. Generate complex Gaussian noise (real and imaginary parts separately)
    noise_re = np.random.normal(0, np.sqrt(beta_r) * sigma_re, size=kspace_high.shape)
    noise_im = np.random.normal(0, np.sqrt(beta_i) * sigma_im, size=kspace_high.shape)
    noise_complex = noise_re + 1j * noise_im

    # 3. Add noise directly in k‑space (Eq. 3)
    kspace_noisy = kspace_high + noise_complex

    # 4. Inverse Fourier transform to obtain complex image, take magnitude (Eq. 4)
    img_noisy_complex = np.fft.ifft2(kspace_noisy)
    img_noisy = np.abs(img_noisy_complex)          # magnitude image

    # 5. Compute full‑size ground‑truth noise map (noisy image minus original high‑field magnitude)
    #    Note: the original high‑field magnitude image is obtained from the clean k‑space
    img_high = np.abs(np.fft.ifft2(kspace_high))
    epsilon_full = img_noisy - img_high

    # 6. Down‑sample in the image domain (to match low‑field physical resolution, Sec III‑B1)
    def resize_2d(arr, scale):
        return zoom(arr, (scale, scale), order=1)

    low_img = resize_2d(img_noisy, scale_factor)
    epsilon_gt = resize_2d(epsilon_full, scale_factor)

    # Clip tiny negative values for numerical stability
    low_img = np.clip(low_img, 0, None)

    return low_img, epsilon_gt

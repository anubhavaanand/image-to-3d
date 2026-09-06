import torch
import lpips
import numpy as np

# Initialize LPIPS model
_lpips_vgg = None

def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes PSNR between two images (assumed to be in range [0, 255] and same shape)."""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def compute_lpips(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Computes LPIPS between two images (assumed to be [-1, 1] tensors in NCHW format)."""
    global _lpips_vgg
    if _lpips_vgg is None:
        _lpips_vgg = lpips.LPIPS(net='vgg').eval()
    
    with torch.no_grad():
        dist = _lpips_vgg(img1, img2)
    return dist.item()

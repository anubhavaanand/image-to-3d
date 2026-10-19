import numpy as np
import torch
import lpips

_lpips_vgg = None


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes PSNR (dB) between two RGB images in [0, 255]."""
    a = np.asarray(img1, dtype=np.float32)[..., :3]
    b = np.asarray(img2, dtype=np.float32)[..., :3]
    mse = float(np.mean((a - b) ** 2))
    if mse < 1e-10:
        return 100.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def _to_lpips_tensor(img) -> torch.Tensor:
    if isinstance(img, torch.Tensor):
        t = img.detach().cpu().float()
        if t.ndim == 3:
            t = t.unsqueeze(0)
        return t
    arr = np.asarray(img, dtype=np.float32)[..., :3]
    if arr.max() > 1.05:
        arr = arr / 255.0
    arr = arr * 2.0 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()


def compute_lpips(img1, img2) -> float:
    """Computes VGG LPIPS perceptual distance between two images (NumPy HWC or Torch NCHW)."""
    global _lpips_vgg
    if _lpips_vgg is None:
        _lpips_vgg = lpips.LPIPS(net="vgg").eval()
    t1 = _to_lpips_tensor(img1)
    t2 = _to_lpips_tensor(img2)
    with torch.no_grad():
        dist = _lpips_vgg(t1, t2)
    return float(dist.item())

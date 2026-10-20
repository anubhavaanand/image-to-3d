from __future__ import annotations

import numpy as np
import rembg
from PIL import Image
from scipy.ndimage import label


class BackgroundRemover:
    def __init__(self, model_name: str = "isnet-general-use"):
        self.model_name = model_name
        self.session = rembg.new_session(self.model_name)

    def remove(self, image: Image.Image) -> Image.Image:
        orig_rgba = image.convert("RGBA")
        orig_arr = np.array(orig_rgba)

        # If input already has a valid transparent background (e.g. from pyrender RGBA), preserve it
        if orig_arr[..., 3].min() < 10 and 0.03 < float((orig_arr[..., 3] > 10).mean()) < 0.85:
            return orig_rgba

        rgba = rembg.remove(orig_rgba, session=self.session).convert("RGBA")
        arr = np.array(rgba)

        # Only strip pure-white (>252) pixels that are connected to the 4 image corners
        # so white shoes / pandas in the center are NEVER erased!
        white_pixels = np.all(orig_arr[..., :3] > 252, axis=-1)
        labeled, _ = label(white_pixels)
        border_labels = set(
            np.concatenate([
                labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]
            ])
        ) - {0}
        border_white_bg = np.isin(labeled, list(border_labels))

        fg_ratio = float((arr[..., 3] > 10).mean())
        if arr[..., 3].max() < 10 or fg_ratio > 0.90:
            fg_mask = ~border_white_bg
            if fg_mask.any():
                orig_arr[..., 3] = fg_mask.astype(np.uint8) * 255
                return Image.fromarray(orig_arr, mode="RGBA")
        else:
            arr[border_white_bg, 3] = 0
            return Image.fromarray(arr, mode="RGBA")
        return rgba

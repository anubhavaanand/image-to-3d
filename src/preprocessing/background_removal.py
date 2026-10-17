from __future__ import annotations

import numpy as np
import rembg
from PIL import Image


class BackgroundRemover:
    def __init__(self, model_name: str = "isnet-general-use"):
        self.model_name = model_name
        self.session = rembg.new_session(self.model_name)

    def remove(self, image: Image.Image) -> Image.Image:
        orig_rgba = image.convert("RGBA")
        orig_arr = np.array(orig_rgba)
        rgba = rembg.remove(orig_rgba, session=self.session).convert("RGBA")
        arr = np.array(rgba)

        white_bg_mask = np.all(orig_arr[..., :3] > 250, axis=-1)
        fg_ratio = float((arr[..., 3] > 10).mean())
        if arr[..., 3].max() < 10 or fg_ratio > 0.85:
            non_white = ~white_bg_mask
            if non_white.any():
                orig_arr[..., 3] = non_white.astype(np.uint8) * 255
                return Image.fromarray(orig_arr, mode="RGBA")
        else:
            arr[white_bg_mask, 3] = 0
            return Image.fromarray(arr, mode="RGBA")
        return rgba

from __future__ import annotations

import numpy as np
import rembg
from PIL import Image


class BackgroundRemover:
    """
    Segments foreground objects into true RGBA images (alpha=0 on background,
    alpha=255 on foreground). Never flatten alpha to 255 here.
    """

    def __init__(self, model_name: str = "isnet-general-use"):
        self.model_name = model_name
        self.session = rembg.new_session(self.model_name)

    def remove(self, image: Image.Image) -> Image.Image:
        rgba = rembg.remove(image.convert("RGBA"), session=self.session).convert("RGBA")
        arr = np.array(rgba)
        if arr[..., 3].max() < 10:
            orig = np.array(image.convert("RGBA"))
            non_white = np.any(orig[..., :3] < 248, axis=-1)
            if non_white.any():
                orig[..., 3] = non_white.astype(np.uint8) * 255
                return Image.fromarray(orig, mode="RGBA")
        return rgba

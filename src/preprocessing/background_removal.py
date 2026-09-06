import rembg
from PIL import Image

class BackgroundRemover:
    def __init__(self, model_name: str = "birefnet-general"):
        self.model_name = model_name
        self.session = rembg.new_session(self.model_name)

    def remove(self, image: Image.Image) -> Image.Image:
        """Removes the background from the image and composites it on a white background."""
        # Convert image to appropriate format
        image_no_bg = rembg.remove(image, session=self.session)
        
        # Composite on white background
        white_bg = Image.new("RGBA", image_no_bg.size, "WHITE")
        white_bg.paste(image_no_bg, (0, 0), image_no_bg)
        
        return white_bg.convert("RGB")

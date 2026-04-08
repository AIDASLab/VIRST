from PIL import Image as PILImage, ImageOps

def read_image_sam2(img_path) -> PILImage.Image:
    with open(img_path , "rb") as f:
        img = PILImage.open(f)
        img = ImageOps.exif_transpose(img)  # Apply EXIF orientation
        if img is None:
            raise ValueError(f"Image is None at {img_path}")
        img = img.convert("RGB")
    return img
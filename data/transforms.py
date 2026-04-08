from typing import List, Union
import torch
import torchvision.transforms.functional as F
import torchvision.transforms.v2.functional as Fv2
from PIL import Image as PILImage 
import numpy as np

def get_size_with_aspect_ratio(image_size, size, max_size=None):
    w, h = image_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = max_size * min_original_size / max_original_size

    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)

    if w < h:
        ow = int(round(size))
        oh = int(round(size * h / w))
    else:
        oh = int(round(size))
        ow = int(round(size * w / h))

    return (oh, ow)

def resize(frames, index, size, max_size=None, square=False, v2=False):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    if square:
        size = size, size
    else:
        cur_size = (
            frames[index].size()[-2:][::-1]
            if v2
            else frames[index].size
        )
        size = get_size(cur_size, size, max_size)

    if v2:
        frames[index] = Fv2.resize(
            frames[index], size, antialias=True
        )
    else:
        frames[index] = F.resize(frames[index], size)

    return frames

class SAM2ResizeAPI:
    def __init__(
        self, size, max_size=None, square=False, v2=False
    ):
        self.size = size
        self.max_size = max_size
        self.square = square
        self.v2 = v2

    def __call__(self, frames, **kwargs):

        for i in range(len(frames)):
            size = self.size
            frames = resize(
                frames, i, size, self.max_size, square=self.square, v2=self.v2
            )
        return frames
    
class SAM2ToTensorAPI:
    def __init__(self, v2=False):
        self.v2 = v2

    def __call__(self, frames, **kwargs):
        imgs = []
        for img in frames:
            if self.v2:
                imgs.append(Fv2.to_image_tensor(img))
            else:
                imgs.append(F.to_tensor(img))
        return imgs

class SAM2NormalizeAPI:
    def __init__(self, mean, std, v2=False):
        self.mean = mean
        self.std = std
        self.v2 = v2

    def __call__(self, frames, **kwargs):
        imgs = []
        for img in frames:
            if self.v2:
                img = Fv2.convert_image_dtype(img, torch.float32)
                img = Fv2.normalize(img, mean=self.mean, std=self.std)
            else:
                img = F.normalize(img, mean=self.mean, std=self.std)
            imgs.append(img)

        return imgs
    
class SAM2Transform:
    def __init__(
        self, 
        size=1024, 
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225],
        square=True,
    ):
        self.transforms = [
            SAM2ResizeAPI(size=size, square=square),
            SAM2ToTensorAPI(),
            SAM2NormalizeAPI(mean=mean,std=std)
        ]
    
    def __call__(self, frames: Union[PILImage.Image,List[PILImage.Image]]) -> torch.Tensor:
        if isinstance(frames, PILImage.Image): # when the input is a single image
            frames = [frames]
        for t in self.transforms:
            frames = t(frames)
        return torch.stack(frames)
    
class SAM2MaskTransform:
    def __init__(self, size= 1024, square=True):
        if square:
            self.size = [size, size]
        else:
            raise NotImplementedError

    def __call__(
        self,
        mask: Union[np.ndarray, List[np.ndarray]]
    ) -> torch.Tensor:
        def process_mask(m: np.ndarray) -> torch.Tensor:
            m = (m>0).astype(np.uint8)
            m_tensor = torch.from_numpy(m).unsqueeze(0) # (1,h,w)
            m_tensor = F.resize(m_tensor, self.size)
            return m_tensor.squeeze(0) # (h,w)
        
        
        if isinstance(mask, np.ndarray):
            return process_mask(mask)
        elif isinstance(mask, list):
            m_list = []
            for m in mask:
                m_list.append(process_mask(m))
            return torch.stack(m_list, dim=0)
        else:
            raise TypeError("Mask should be binary uint8 numpy ndarray.")

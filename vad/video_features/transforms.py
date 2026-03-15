"""
Transforms for video feature extraction
"""

import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image


class ToFloatTensorInZeroOne(torch.nn.Module):
    """Convert to float tensor in [0, 1]"""
    def forward(self, pic):
        if isinstance(pic, np.ndarray):
            pic = torch.from_numpy(pic)
        if isinstance(pic, torch.Tensor):
            return pic.float() / 255.0
        return TF.to_tensor(pic)


class Resize(torch.nn.Module):
    """Resize image so that min(H,W) = size"""
    def __init__(self, size=256):
        super().__init__()
        self.size = size

    def forward(self, pic):
        if isinstance(pic, torch.Tensor):
            h, w = pic.shape[-2:]
        else:
            w, h = pic.size
        
        if min(h, w) == self.size:
            return pic
        
        if h < w:
            new_h, new_w = self.size, int(self.size * w / h)
        else:
            new_h, new_w = int(self.size * h / w), self.size
        
        if isinstance(pic, torch.Tensor):
            return torch.nn.functional.interpolate(
                pic.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False
            ).squeeze(0)
        else:
            return pic.resize((new_w, new_h), Image.BILINEAR)


class CenterCrop(torch.nn.Module):
    """Center crop image to size"""
    def __init__(self, size=224):
        super().__init__()
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def forward(self, pic):
        if isinstance(pic, torch.Tensor):
            _, h, w = pic.shape
        else:
            w, h = pic.size
        
        left = (w - self.size[1]) // 2
        top = (h - self.size[0]) // 2
        right = left + self.size[1]
        bottom = top + self.size[0]
        
        if isinstance(pic, torch.Tensor):
            return pic[:, top:bottom, left:right]
        else:
            return pic.crop((left, top, right, bottom))


class ToUInt8(torch.nn.Module):
    """Convert tensor to uint8"""
    def forward(self, pic):
        if isinstance(pic, torch.Tensor):
            pic = torch.clamp(pic, 0, 255).to(torch.uint8)
        return pic


class ScaleTo1_1(torch.nn.Module):
    """Scale pixel values to [-1, 1]"""
    def forward(self, pic):
        if isinstance(pic, torch.Tensor):
            return (pic.float() - 127.5) / 127.5
        return pic


class PermuteAndUnsqueeze(torch.nn.Module):
    """Permute (T,C,H,W) to (1,C,T,H,W) or add batch dim to (C,H,W)"""
    def forward(self, pic):
        if isinstance(pic, torch.Tensor):
            if pic.dim() == 4:
                # Input: [T, C, H, W] → permute to [C, T, H, W] → unsqueeze to [1, C, T, H, W]
                return pic.permute(1, 0, 2, 3).unsqueeze(0)
            elif pic.dim() == 3:
                # Input: [C, H, W] → add batch dimension [1, C, H, W]
                return pic.unsqueeze(0)
        return pic


class PILToTensor(torch.nn.Module):
    """Convert PIL Image to tensor"""
    def forward(self, pic):
        if isinstance(pic, Image.Image):
            return TF.pil_to_tensor(pic)
        return pic


class TensorCenterCrop(torch.nn.Module):
    """Center crop on tensor in (B,C,H,W) format"""
    def __init__(self, size=224):
        super().__init__()
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def forward(self, pic):
        _, _, h, w = pic.shape
        top = (h - self.size[0]) // 2
        left = (w - self.size[1]) // 2
        return pic[:, :, top:top+self.size[0], left:left+self.size[1]]


class Clamp(torch.nn.Module):
    """Clamp tensor values"""
    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, pic):
        if isinstance(pic, torch.Tensor):
            return torch.clamp(pic, self.min_val, self.max_val)
        return pic


class ResizeImproved(torch.nn.Module):
    """Improved resize for PIL images"""
    def __init__(self, size=256):
        super().__init__()
        self.size = size

    def forward(self, pic):
        if isinstance(pic, Image.Image):
            w, h = pic.size
            if min(h, w) == self.size:
                return pic
            if h < w:
                new_h, new_w = self.size, int(self.size * w / h)
            else:
                new_h, new_w = int(self.size * h / w), self.size
            return pic.resize((new_w, new_h), Image.LANCZOS)
        return pic


class ToFloat(torch.nn.Module):
    """Convert tensor to float"""
    def forward(self, pic):
        if isinstance(pic, torch.Tensor):
            return pic.float()
        return pic


import os
from os.path import join
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import nibabel as nib

class RIGA_labeled_set(Dataset):
    def __init__(self, root, image_list, label_list, target_size=(512, 512)):
        self.root = root
        self.image_list = image_list
        self.label_list = label_list
        self.target_size = target_size

    def __getitem__(self, item):
        # 1. 修复路径拼接问题 (强制转 str)
        img_file = join(self.root, str(self.image_list[item]))
        label_file = join(self.root, str(self.label_list[item]))

        # 2. 读取 Image (支持 NIfTI 和 普通图片)
        if img_file.endswith('.nii.gz') or img_file.endswith('.nii'):
            nii = nib.load(img_file)
            vol = nii.get_fdata()
            # 归一化
            vol = (vol - np.percentile(vol, 1)) / (np.percentile(vol, 99) - np.percentile(vol, 1) + 1e-8)
            vol = np.clip(vol, 0, 1)
            # 维度调整 (H,W,D) -> (D,H,W)
            if vol.ndim == 3:
                vol = vol.transpose(2, 0, 1)
                vol = np.rot90(vol, k=1, axes=(1, 2)) # 旋转以匹配方向
            img = torch.from_numpy(vol.copy()).float()
        else:
            img = Image.open(img_file)

        # 3. 读取 Label (支持 NIfTI 和 普通图片)
        if label_file.endswith('.nii.gz') or label_file.endswith('.nii'):
            mask = nib.load(label_file).get_fdata()
            if mask.ndim == 3:
                mask = mask.transpose(2, 0, 1)
                mask = np.rot90(mask, k=1, axes=(1, 2))
            label = torch.from_numpy(mask.copy()).float()
        else:
            label = Image.open(label_file)

        # 4. Resize Image (修复缩进错误 + Tensor支持)
        if isinstance(img, torch.Tensor):
            # Tensor 使用 interpolate
            if img.ndim == 3: img = img.unsqueeze(1)
            img = F.interpolate(img, size=self.target_size, mode='bilinear', align_corners=False)
            if img.ndim == 4: img = img.squeeze(1)
        else:
            # PIL 使用 resize
            img = img.resize(self.target_size)

        # 5. Resize Label
        if isinstance(label, torch.Tensor):
            if label.ndim == 3: label = label.unsqueeze(1)
            label = F.interpolate(label, size=self.target_size, mode='nearest')
            if label.ndim == 4: label = label.squeeze(1)
        else:
            label = label.resize(self.target_size, Image.NEAREST)

        return img, label, str(self.image_list[item])

    def __len__(self):
        return len(self.image_list)

class RIGA_unlabeled_set(Dataset):
    def __init__(self, root, image_list, target_size=(512, 512)):
        self.root = root
        self.image_list = image_list
        self.target_size = target_size

    def __getitem__(self, item):
        img_file = join(self.root, str(self.image_list[item]))
        if img_file.endswith('.nii.gz') or img_file.endswith('.nii'):
            nii = nib.load(img_file)
            vol = nii.get_fdata()
            vol = (vol - np.percentile(vol, 1)) / (np.percentile(vol, 99) - np.percentile(vol, 1) + 1e-8)
            vol = np.clip(vol, 0, 1)
            if vol.ndim == 3:
                vol = vol.transpose(2, 0, 1)
                vol = np.rot90(vol, k=1, axes=(1, 2))
            img = torch.from_numpy(vol.copy()).float()
        else:
            img = Image.open(img_file)

        if isinstance(img, torch.Tensor):
            if img.ndim == 3: img = img.unsqueeze(1)
            img = F.interpolate(img, size=self.target_size, mode='bilinear', align_corners=False)
            if img.ndim == 4: img = img.squeeze(1)
        else:
            img = img.resize(self.target_size)

        return img, str(self.image_list[item])

    def __len__(self):
        return len(self.image_list)

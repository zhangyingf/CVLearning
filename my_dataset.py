from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy import ndimage
import nibabel as nib  # nii格式一般都会用到这个包
import skimage

class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


def normalize(volume):
    """归一化"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def resize_volume(img, h=128, w=128, d=128):
    """修改图像大小"""
    # # Get current depth
    current_depth = img.shape[-1]
    # current_width = img.shape[0]
    # current_height = img.shape[1]
    # # Compute depth factor
    # depth = current_depth / desired_depth
    # width = current_width / desired_width
    # height = current_height / desired_height
    # depth_factor = 1 / depth
    # width_factor = 1 / width
    # height_factor = 1 / height
    # 旋转
    img = ndimage.rotate(img, 90, reshape=False)
    # 数据调整
    # img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    if current_depth <= d:
        img = skimage.transform.resize(img,[h, w, d],order=1, mode='reflect', cval=0, clip=True, preserve_range=False, anti_aliasing=True, anti_aliasing_sigma=None)
    else:
        idx = np.random.choice(range(img.shape[-1]), d)
        img = img[:, :, idx]
    return img.astype(np.float32)

class BrainPETSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, h=128, d=64, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
        self.h = h
        self.d = d

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        # img = Image.open(self.images_path[item])
        # # RGB为彩色图片，L为灰度图片
        # if img.mode != 'RGB':
        #     raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        nii_volume = nib.load(self.images_path[item])  # 读取nii
        volume = nii_volume.get_fdata().squeeze()
        # 归一化
        # volume = normalize(volume)
        # 调整尺寸 h=128, w=128, d=64
        img = resize_volume(volume, h=self.h, w=self.h, d=self.d).astype(np.float32)

        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
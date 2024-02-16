from torch.utils.data import Dataset
import numpy as np
import cv2
from pathlib import Path
from omegaconf import OmegaConf
from .utils import *
import albumentations as A

conf ={
    "image_dir": '/home/koki/Sparse_Matcher/data/Megadepth/Megadepth/',
    "image_list": '/home/koki/Sparse_Matcher/megadepth_homography_list_no1522.txt',
    "homography": {
        "difficulty": 0.6,
        "translation": 1.0,
        "max_angle": 60,
        "n_angles": 10,
        "min_convexity": 0.05,
    },
}

# python -m superpoint.homography_dataset


albumentations_transform = A.Compose([
    A.RandomRain(p=0.2),
    A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0),
            contrast_limit=(-0.1, 0),
    ),
    A.RandomGamma(gamma_limit=(15, 65)),
])


class HomographyDataset(Dataset):
    def __init__(self,conf):
        conf = OmegaConf.create(conf)

        image_list = Path(conf.image_list)
        images = image_list.read_text().rstrip("\n").split("\n")
        np.random.RandomState(0).shuffle(images)
        images = images[:25000]
        self.image_names = np.array(images)
        self.image_dir = conf.image_dir
        self.conf = conf
        self.photo_augment = albumentations_transform

    def __len__(self):
        return len(self.image_names)
    def __getitem__(self,idx):
        name = self.image_names[idx]
        image = cv2.imread(self.image_dir + name)
        image = image[..., ::-1]

        image = image.astype(np.float32) / 255.0
        size = image.shape[:2][::-1]
        patch_shape = [640, 480]
        data0 = self._read_view(image, self.conf.homography, patch_shape)
        data1 = self._read_view(image, self.conf.homography, patch_shape)

        H = compute_homography(data0["coords"], data1["coords"], [1, 1])
        
        data = {
            "name": name,
            "original_image_size": np.array(size),
            "H_0to1": H.astype(np.float32),
            "idx": idx,
            "view0": data0,
            "view1": data1,
        }
        return data
    
    def _read_view(self, image, conf, patch_shape):
        data = sample_homography(image, conf, patch_shape)

        image = data["image"]
        # transformed = self.photo_augment(image=image)
        # transformed_image = transformed["image"]
        data["image"] = image.transpose((2, 0, 1))  #HWC to CHW
        
        return data

def sample_homography(img, conf: dict, patch_shape: list):
    data = {}
    ori_shape = img.shape[:2][::-1]
    H, coords= sample_homography_corners(ori_shape, patch_shape, **conf)
    
    data["image"] = cv2.warpPerspective(img, H, tuple(patch_shape))
    data["image_size"] = np.array(patch_shape, dtype=np.float32)
    data["H_"]     =      H.astype(np.float32)
    data["coords"] = coords.astype(np.float32)
    
    return data


import matplotlib.pyplot as plt

if __name__=="__main__":
    
    dataset = HomographyDataset(conf)
    data = dataset[100]
    image0 = data['view0']['image']
    image1 = data['view1']['image']
    coord0 = data['view0']['coords']
    coord1 = data['view1']['coords']
    plt.imsave(f'/home/koki/Sparse_Matcher/augment_image0.png',image0)
    plt.imsave(f'/home/koki/Sparse_Matcher/augment_image1.png',image1)
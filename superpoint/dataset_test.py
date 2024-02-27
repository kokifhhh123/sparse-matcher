from superpoint.homography_dataset import HomographyDataset,conf
import numpy as np
from torch.utils.data import DataLoader
from superpoint.superpoint import SuperPoint
import torch
from torch.nn.functional import pad
from superpoint.utils import gt_matches_from_homography
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = HomographyDataset(conf)
dataloader = DataLoader(dataset)
extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)

data = next(iter(dataloader))

image0 = data['view0']['image'][0].to(device)
image1 = data['view1']['image'][0].to(device)
H      = data['H_0to1'].to(device)

feats0 = extractor.extract(image0)
feats1 = extractor.extract(image1)

kpts0 = feats0['keypoints']
kpts1 = feats1['keypoints']
desc0 = feats0['descriptors'][0].cpu()
desc1 = feats1['descriptors'][0].cpu()


all_match = gt_matches_from_homography(kpts0, kpts1, H)
m0 = all_match['matches0'][0].cpu()
m0_vailid = m0>-1
match0      = m0[m0_vailid]

desc_match0 = desc0[m0_vailid]
desc_match1 = desc1[match0]


print(desc_match0.shape)
print(desc_match1.shape)
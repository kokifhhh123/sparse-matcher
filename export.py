from superpoint.homography_dataset import HomographyDataset,conf
import numpy as np
from torch.utils.data import DataLoader
from superpoint.superpoint import SuperPoint
import torch
from torch.nn.functional import pad
from superpoint.utils import gt_matches_from_homography
from tqdm import tqdm
import os


'''
python export_megadepth.py
'''

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = HomographyDataset(conf)
    dataloader = DataLoader(dataset, num_workers=4)
    extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)
    
    for data in tqdm(dataloader):

        name = data['name'][0]
        name = os.path.splitext(os.path.basename(name))[0]

        image0 = data['view0']['image'][0].to(device)
        image1 = data['view1']['image'][0].to(device)
        H = data['H_0to1'].to(device)

        feats0 = extractor.extract(image0)
        feats1 = extractor.extract(image1)


        kpts0 = feats0['keypoints']
        kpts1 = feats1['keypoints']

        desc0 = feats0['descriptors'][0].cpu()
        desc1 = feats1['descriptors'][0].cpu()

        
        all_match = gt_matches_from_homography(kpts0,kpts1,H)
        m0 = all_match['matches0'][0].cpu()

        padding_d0 = (0,0,0,1024-len(desc0))
        padding_d1 = (0,0,0,1024-len(desc1))
        

        desc0 = pad(desc0, padding_d0)
        desc1 = pad(desc1, padding_d1)

        desc_all = [desc0, desc1, m0]


        path = f'/home/koki/Sparse_Matcher/data/Megadepth/Superpoint_features/scene_no_15_22_features2/{name}.pth'
        torch.save(desc_all, path)


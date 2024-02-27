import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
import os
import time
from module import MLP_module_4_short, MLP_module_8_short, MLP_module_16_short
import argparse
import random
from torch.nn.functional import pad

'''
python -m mlp.mlp_train

nohup python3 -u -m mlp.mlp_train2 \
    --type 16_short \
    --folder_path /home/koki/Sparse_Matcher/data/Megadepth/MLP_ckpt/1_short/pair2/dim16/ \
    >/home/koki/Sparse_Matcher/data/Megadepth/MLP_ckpt/1_short/pair2/log/pair_log_16.txt 2>&1 &


nohup python3 -u train.py \
    --type 4_short \
    --folder_path /home/koki/Sparse_Matcher/testrun/ \
    >/home/koki/Sparse_Matcher/testlog.txt 2>&1 &
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


EPOCH = 500
FEATURE_PATH = '/home/koki/Sparse_Matcher/data/Megadepth/features/scene_no_15_22_features_pair/'
BATCH_SIZE = 128

class MLPDataset(Dataset):
    def __init__(self):
        self.feature_path = FEATURE_PATH
        self.files = os.listdir(FEATURE_PATH)

    def __getitem__(self, index):
        feat_path= self.feature_path + self.files[index]
        feats = torch.load(feat_path)
        
        desc0 = feats[0].cuda()
        desc1 = feats[1].cuda()
        m0    = feats[2]

        m0_vailid = m0>-1
        padding_m = (0,1024-len(m0_vailid))

        # print('m0 len: ',len(m0_vailid))
        m0_vailid = pad(m0_vailid, padding_m, "constant" ,value=False)
        m0        = pad(m0,        padding_m, "constant", value=0)

        match0      = m0[m0_vailid]
        desc_match0 = desc0[m0_vailid]
        

        length = len(desc_match0)
        desc_match1 = desc1[match0][:]
        
        padding_m = (0, 0, 0, 1024-length)
        desc_match0 = pad(desc_match0, padding_m).cuda()
        desc_match1 = pad(desc_match1, padding_m).cuda()

        return desc0, desc1, desc_match0, desc_match1

    def __len__(self):
        return len(self.files)


dataset = MLPDataset()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MLP_module_4_short().to(device)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,num_workers=4)


def train_one_epoch():
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(trainloader):
        d0,d1,dm0,dm1 = data
        _ , d0_back = model(d0)
        _ , d1_back = model(d1)
        _ , dm0_back = model(dm0)
        _ , dm1_back = model(dm1)

        optimizer.zero_grad()

        loss0 = loss_fn(d0, d0_back)
        loss1 = loss_fn(d1, d1_back)
        lossm = loss_fn(dm0_back, dm1_back) * 2

        print(loss0.item(),loss1.item(),lossm.item())

        loss = loss0 + loss1 + lossm
        
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 50 == 49:
            last_loss = running_loss / 50 # loss per batch
            print('batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.
        
    return last_loss



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default="4_short", help='mlp dimension')
    parser.add_argument('--folder_path')
    args = parser.parse_args()
    type = args.type
    folder_path = args.folder_path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f'Folder "{folder_path}" created successfully.')
    else:
        print(f'Folder "{folder_path}" already exists.')


    mlp_modules = {
        "4_short": MLP_module_4_short,
        "8_short": MLP_module_8_short,
        "16_short": MLP_module_16_short
    }
    selected_module = mlp_modules[type]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = selected_module()



    # ckpt_path = '/home/koki/Sparse_Matcher/data/Megadepth/MLP_ckpt/1_short/pair/dim8/model_20240219_232809_285'
    # ckpt = torch.load(ckpt_path)
    # model.load_state_dict(ckpt)




    model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f"{folder_path}runs/fashion_trainer_{timestamp}")


    epoch_number = 0
    best_vloss = 1_000_000.
    seed = 100
    torch.manual_seed(seed)
    dataset = MLPDataset()
    trainset, valset = random_split(dataset, [0.8, 0.2])
    trainloader = DataLoader(trainset, batch_size = BATCH_SIZE, shuffle=True)
    valloader   = DataLoader(valset,   batch_size = BATCH_SIZE, shuffle=True)



    for epoch in range(EPOCH):
        print('EPOCH {}:'.format(epoch_number + 1))
        start_time = time.time()
        
        model.train(True)
        avg_train_loss = train_one_epoch()


        # running_vloss = 0.0
        # model.eval()
        # with torch.no_grad():
        #     for i, data in enumerate(valloader):
        #         d0,d1,dm0,dm1 = data
        #         _ , d0_back = model(d0)
        #         _ , d1_back = model(d1)
        #         _ , dm0_back = model(dm0)
        #         _ , dm1_back = model(dm1)

        #         loss0 = loss_fn(d0, d0_back)
        #         loss1 = loss_fn(d1, d1_back)
        #         lossm = loss_fn(dm0_back, dm1_back) * 2
        #         loss = loss0 + loss1 + lossm
        #         running_vloss += loss

        # avg_val_loss = running_vloss / (i + 1)
        # print('LOSS train {} valid {}'.format(avg_train_loss, avg_val_loss))
        # end_time = time.time()

        # epoch_time = end_time - start_time
        # print('Time taken for one EPOCH {}: {:.2f} seconds'.format(epoch + 1, epoch_time))

        # writer.add_scalars('Training vs. Validation',
        #                 { 'Training' : avg_train_loss, 'Validation' : avg_val_loss },
        #                 epoch_number+1)
        # writer.flush()
        # if avg_val_loss < best_vloss:
        #     best_vloss = avg_val_loss
        #     model_path = '{}model_{}_{}'.format(folder_path, timestamp, epoch_number)
        #     torch.save(model.state_dict(), model_path)

        # epoch_number += 1


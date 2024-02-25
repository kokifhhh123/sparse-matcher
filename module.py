import torch
from torch import nn

from torchrl.modules import MLP


class MLP_module_4_long(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.MLP    = MLP(in_features=256, out_features=4   , num_cells=[128, 64, 32, 32, 16, 16, 8, 4])
        self.MLP_de = MLP(in_features=4, out_features=256,    num_cells=[4, 8, 16, 16, 32, 32, 64, 128])

    def forward(self, desc: torch.Tensor):
        desc_mlp = self.MLP(desc)
        desc_back = self.MLP_de(desc_mlp)
        return desc_mlp, desc_back
    

class MLP_module_8_long(nn.Module):
    def __init__(self):
        super().__init__()
        self.MLP    = MLP(in_features=256, out_features=8   , num_cells=[128, 64, 64, 32, 32, 16, 8])
        self.MLP_de = MLP(in_features=8, out_features=256,    num_cells=[8, 16, 32, 32, 64, 64, 128])
    def forward(self, desc: torch.Tensor):
        desc_mlp = self.MLP(desc)
        desc_back = self.MLP_de(desc_mlp)
        return desc_mlp, desc_back
    

class MLP_module_16_long(nn.Module):
    def __init__(self):
        super().__init__()
        self.MLP    = MLP(in_features=256, out_features=16   , num_cells=[256, 128, 128, 64, 64, 32, 32, 16, 16])
        self.MLP_de = MLP(in_features=16, out_features=256,    num_cells=[16, 16, 32, 32, 64, 64, 128, 128, 256])
    def forward(self, desc: torch.Tensor):
        desc_mlp = self.MLP(desc)
        desc_back = self.MLP_de(desc_mlp)
        return desc_mlp, desc_back


class MLP_module_4_short(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.MLP    = MLP(in_features=256, out_features=4   , num_cells=[128, 64, 32, 16, 8])
        self.MLP_de = MLP(in_features=4,   out_features=256,  num_cells=[8, 16, 32, 64, 128])

    def forward(self, desc: torch.Tensor):
        desc_mlp = self.MLP(desc)
        desc_back = self.MLP_de(desc_mlp)
        return desc_mlp, desc_back
    

class MLP_module_8_short(nn.Module):
    def __init__(self):
        super().__init__()


        self.MLP    = MLP(in_features=256, out_features=8   , num_cells=[128, 64, 32, 16, 8])
        self.MLP_de = MLP(in_features=8,   out_features=256,  num_cells=[8, 16, 32, 64, 128])
    
    
    def forward(self, desc: torch.Tensor):
        desc_mlp = self.MLP(desc)
        desc_back = self.MLP_de(desc_mlp)
        return desc_mlp, desc_back

class MLP_module_16_short(nn.Module):
    def __init__(self):
        super().__init__()
        self.MLP    = MLP(in_features=256, out_features=16   , num_cells=[128, 64, 32, 16])
        self.MLP_de = MLP(in_features=16, out_features=256,    num_cells=[16, 32, 64, 128])
    def forward(self, desc: torch.Tensor):
        desc_mlp = self.MLP(desc)
        desc_back = self.MLP_de(desc_mlp)
        return desc_mlp, desc_back
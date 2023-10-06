import torch
from torch import nn
from torch.nn import functional as F
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange, repeat

from nat import NeighborhoodAttention


class Channel_Layernorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features,1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features,1)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


     

class MSNATLayer(nn.Module):
    def __init__(self,input_size, dim, num_heads,window_size=7,window_size2=7,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=Channel_Layernorm, layer_scale=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention(input_size, dim, num_heads,window_size,qkv_bias, qk_scale, attn_drop, drop)
        self.attn2 = NeighborhoodAttention(input_size, dim, num_heads,window_size2,qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x) + self.attn2(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    





class MS_STN_NAT2(nn.Module):
    def __init__(self, num_classes=16, bands = 200, msize=16):
        super(MS_STN_NAT2, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(bands, 49, kernel_size=1, padding=0),
                                    nn.BatchNorm2d(49),
                                    )  

        self.nat1 =  MSNATLayer((11,11),49,7,window_size=3, window_size2=5)

        self.layer2 = nn.Sequential(nn.Conv2d(49, 16, 1),
                                    nn.BatchNorm2d(16),) 

        self.nat2 = MSNATLayer((11,11),16,4,window_size=3, window_size2=5)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(49, 49, kernel_size=5),
            nn.MaxPool2d(2, stride=1),
            nn.ReLU(True),
            nn.Conv2d(49, 16, kernel_size=3),
            nn.MaxPool2d(2, stride=1),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(16 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # Spatial transformer localization-network
        self.localization2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5),
            nn.MaxPool2d(2, stride=1),
            nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=3),
            nn.MaxPool2d(2, stride=1),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc2 = nn.Sequential(
            nn.Linear(16 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc2[2].weight.data.zero_()
        self.fc_loc2[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.fc = nn.Linear(16, num_classes)


    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 16 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    # Spatial transformer network forward function
    def stn2(self, x):
        xs = self.localization2(x)
        xs = xs.view(-1, 16 * 3 * 3)
        theta = self.fc_loc2(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    

    def forward(self, x):
        n,c,h,w = x.size()

        x = self.layer1(x)
        x = x + self.nat1(self.stn(x))

        x = self.layer2(x)

        x = x + self.nat2(self.stn2(x))
        # x = x + self.nat2(self.stn2(x))


        x = F.avg_pool2d(x, x.size()[-1])
        x = self.fc(x.squeeze())
        
        return x 




import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False):
        super(Residual, self).__init__()
        if batch_norm:
            self.conv_block1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.conv_block2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv_block1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            )
            self.conv_block2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv_block1(x) # Conv + (BN) + ReLu
        x = self.conv_block2(x) # Conv + (BN)
        x = x + residual        # residual connection
        x =  self.relu(x)       # ReLu
        return x

class FreBlock(nn.Module):
    def __init__(self, nc):
        super(FreBlock, self).__init__()
        self.preprosess = nn.Conv2d(nc, nc, 1, 1, 0)
        self.processmag = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.processpha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))

        self.lambda_res = nn.Parameter(torch.tensor(0.),requires_grad=True)

    def forward(self, x):
        _, _, H, W = x.shape
        x_ori = x
        x = self.preprosess(x)

        x_freq = torch.fft.rfft2(x, norm='backward')                # FFT
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.processmag(mag)
        pha = self.processpha(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')  # IFFT

        # Dynamic Skip Connection
        lambda_res = torch.sigmoid(self.lambda_res)                 # learnable weights
        x_out = x_out * lambda_res + x_ori * (1-lambda_res)

        # Adaptive Clip (AdaClip)
        epsilon = 0.5
        x_out = x_out - torch.mean(x_out) + torch.mean(x_ori)
        x_out = torch.clip(x_out, float(x_ori.min()-epsilon), float(x_ori.max()+epsilon))

        return x_out

class ResBlock(nn.Module):
    def __init__( self, dim):
        super(ResBlock, self).__init__()
        self.dim = dim
        self.conv_init = nn.Conv2d(dim, dim, 1)
        self.conv_fina = nn.Conv2d(dim, dim, 1)

        self.mixer_local = Residual(dim, dim, batch_norm=True)

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//2, dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv_init(x)   # [B,C,H,W] -> [B,C,H,W]
        x = self.mixer_local(x) # [B,C,H,W] -> [B,C,H,W]
        # x = self.gelu(x)
        x = self.ca(x) * x
        x = self.conv_fina(x)   # [B,2C,H,W] -> [B,C,H,W]
        return x

class SFFS(nn.Module):
    """
    Spatial-Frequency Fusion Strategy (SFFS)
    """
    def __init__(self, dim):
        super(SFFS, self).__init__()
        self.dim = dim
        self.conv_init = nn.Conv2d(dim, 2*dim, 1)
        self.conv_fina = nn.Conv2d(2*dim, dim, 1)

        self.mixer_local = Residual(dim, dim, batch_norm=True)
        self.mixer_gloal = FreBlock(dim)

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2*dim, dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 2*dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv_init(x)                       # [B,C,H,W] -> [B,2C,H,W]
        x = list(torch.split(x, self.dim, dim=1))   # [B,2C,H,W] -> [B,C,H,W], [B,C,H,W]
        x_local = self.mixer_local(x[0])            # [B,C,H,W] -> [B,C,H,W]
        x_gloal = self.mixer_gloal(x[1])            # [B,C,H,W] -> [B,C,H,W]
        x = torch.cat([x_local, x_gloal], dim=1)    # [B,C,H,W] -> [B,2C,H,W]
        x = self.gelu(x)                            # activate
        x = self.ca(x) * x                          # channel attention
        x = self.conv_fina(x)                       # [B,2C,H,W] -> [B,C,H,W]
        return x

class Bottleneck(nn.Module):
    def __init__(self, channels):
        super(Bottleneck, self).__init__()
        self.channels = channels

        self.block = SFFS(channels)

        self.ca = ChannelAttention_SCA(channels*3)
        self.sa = SpatialAttention_SCA(channels)

    def forward(self, x):
        b1 = self.block(x)
        b2 = self.block(b1)
        b3 = self.block(b2)

        w = self.ca(torch.cat([b1, b2, b3], dim=1))
        w = w.view(-1, 3, self.channels)[:, :, :, None, None]   # [B,C*3,1,1] -> [C,3,C] -> [C,3,C,1,1]

        b1_ca = w[:, 0, :, :] * b1
        b2_ca = w[:, 1, :, :] * b2
        b3_ca = w[:, 2, :, :] * b3

        b = b1_ca + b2_ca + b3_ca
        b_sa = self.sa(b) * b

        return b_sa

class Down(nn.Module):
    def __init__(self, n_feat):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 4, kernel_size=3, stride=1, padding=1, bias=False), # [B,C,H,W] -> [B,C//4,H,W]
            nn.PixelUnshuffle(2)    # [B,C//4,H,W] -> [B,C,H/2,W/2]
        )
    
    def forward(self, x):
        x = self.down(x)            # [B,C,H,W] -> [B,C//4,H,W] -> [B,C,H/2,W/2]
        return x

class Up(nn.Module):
    def __init__(self, n_feat):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 4, kernel_size=3, stride=1, padding=1, bias=False),  # [B,C,H,W] -> [B,4C,H,W]
            nn.PixelShuffle(2)  # [B,4C,H,W] -> [B,C,2H,2W]
        )
        
    def forward(self, x):
        x = self.up(x)          # [B,C,H,W] -> [B,4C,H,W] -> [B,C,2H,2W]
        return x

class ChannelAttention_SCA(nn.Module):
    """Paper: SCANet: Self-Paced Semi-Curricular Attention Network for Non-Homogeneous Image Dehazing"""
    def __init__(self, nc, number=16):
        super(ChannelAttention_SCA, self).__init__()
        self.conv1 = nn.Conv2d(nc, nc, 3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(nc)
        self.prelu = nn.PReLU(nc)

        self.conv2 = nn.Conv2d(nc, nc, 3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(nc)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(nc, number, 1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU()

        self.fc2 = nn.Conv2d(number, nc, 1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        se = self.gap(x)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        return se

class SpatialAttention_SCA(nn.Module):
    """Paper: SCANet: Self-Paced Semi-Curricular Attention Network for Non-Homogeneous Image Dehazing"""
    def __init__(self, nc, number=16):
        super(SpatialAttention_SCA, self).__init__()
        self.conv1 = nn.Conv2d(nc,nc,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(nc)
        self.prelu = nn.PReLU(nc)

        self.conv2 = nn.Conv2d(nc,number,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(number)
        
        self.conv3 = nn.Conv2d(number,number,kernel_size=3,stride=1,padding=3,dilation=3,bias=False)
        self.conv4 = nn.Conv2d(number,number,kernel_size=3,stride=1,padding=5,dilation=5,bias=False)
        self.conv5 = nn.Conv2d(number,number,kernel_size=3,stride=1,padding=7,dilation=7,bias=False)
        
        self.fc1 = nn.Conv2d(number*4,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.relu = nn.ReLU()

        self.fc2 = nn.Conv2d(1, 1, 1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        x1 = x
        x2 = self.conv3(x)
        x3 = self.conv4(x)
        x4 = self.conv5(x)
        
        se = torch.cat([x1, x2, x3, x4], dim=1)
        
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        
        return se

class SkipConnection(nn.Module):
    def __init__(self, nc):
        super(SkipConnection, self).__init__()
        self.merge = nn.Conv2d(2*nc, nc, 1)

    def forward(self, x, y):
        out = self.merge(torch.cat([x, y], dim=1))
        return out
    
class Encoder(nn.Module):
    def __init__(self, in_channels=3, base_dim=64):
        super().__init__()

        self.conv_init1 = nn.Conv2d(in_channels, base_dim, kernel_size=3, stride=1, padding=1)
        self.conv_init2 = nn.Conv2d(in_channels, base_dim, kernel_size=3, stride=1, padding=1)
        self.conv_init3 = nn.Conv2d(in_channels, base_dim, kernel_size=3, stride=1, padding=1)
        self.merge2 = nn.Conv2d(2 * base_dim, base_dim, kernel_size=3, stride=1, padding=1)
        self.merge3 = nn.Conv2d(2 * base_dim, base_dim, kernel_size=3, stride=1, padding=1)

        self.block1 = ResBlock(base_dim)
        self.down1 = Down(base_dim)             # [B,C,H,W] -> [B,C,H//2,W//2]
        self.block2 = ResBlock(base_dim)
        self.down2 = Down(base_dim)             # [B,C,H//2,W//2] -> [B,C,H//4,W//4]

        self.bottleneck = Bottleneck(base_dim)  # [B,C,H//4,W//4] -> [B,C,H//4,W//4]

    def forward(self, x):
        x_1 = x                                     # [B,C,H,W]
        x_2 = F.interpolate(x_1, scale_factor=0.5)  # [B,C,H//2,W//2]
        x_3 = F.interpolate(x_2, scale_factor=0.5)  # [B,C,H//4,W//4]
        x_1 = self.conv_init1(x_1)
        x_2 = self.conv_init2(x_2)
        x_3 = self.conv_init3(x_3)

        x_1 = self.block1(x_1)
        x1 = x_1
        x_1 = self.down1(x_1)
        
        x_2 = self.merge2(torch.cat([x_1, x_2], dim=1))
        x_2 = self.block2(x_2)
        x2 = x_2
        x_2 = self.down2(x_2)

        x_3 = self.merge3(torch.cat([x_2, x_3], dim=1))
        feat = self.bottleneck(x_3)

        return feat, x1, x2

class Decoder(nn.Module):
    def __init__(self, base_dim=64, out_channels=3):
        super().__init__()
        self.skip2 = SkipConnection(base_dim)
        self.skip1 = SkipConnection(base_dim)

        self.up2 = Up(base_dim)             # [B,C,H//4,W//4] -> [B,C,H//2,W//2]
        self.block2 = ResBlock(base_dim)    # [B,C,H//2,W//2] -> [B,C,H//2,W//2]

        self.up1 = Up(base_dim)             # [B,C,H//2,W//2] -> [B,C,H,W]
        self.block1 = ResBlock(base_dim)    # [B,C,H,W] -> [B,C,H,W]

        self.conv_fina = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(base_dim, out_channels, kernel_size=3, stride=1, padding=1)
        )
        
    def forward(self, x, x1, x2):

        x = self.up2(x)
        x = self.skip2(x, x2)
        x = self.block2(x)

        x = self.up1(x)
        x = self.skip1(x, x1)
        x = self.block1(x)

        x = self.conv_fina(x)
        
        return x
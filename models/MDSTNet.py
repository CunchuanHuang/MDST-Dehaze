from torch import nn
import torch
from torch.nn import functional as F

# import sys
# sys.path.append('..')
from models.VGG19 import VGG19
from myutils.utils_tensor import mean_variance_norm


class ZeroConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        # self.conv.weight.data.fill_(1e-5)
        # self.conv.bias.data.fill_(1e-5)
        self.scale = nn.Parameter(torch.zeros(1, out_channels, 1, 1))   # Learnable parameter

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)
        return out

class mix(nn.Module):
    def __init__(self):
        super(mix, self).__init__()
        self.w = nn.Parameter(torch.tensor([1.]), requires_grad=True)
    def forward(self, x_phase, y_phase):
        return x_phase + self.w * y_phase
    
class SkipConnection(nn.Module):
    def __init__(self, in_nc, residual=True):
        super(SkipConnection, self).__init__()
        self.residual = residual
        self.pre = nn.Conv2d(in_nc, in_nc, 1, 1, 0)
        self.mix = mix()
        self.zero_conv = ZeroConv2d(in_nc, in_nc)

    def forward(self, x, y):
        """
        x: Upsampling result
        y: Downsampling result
        """
        _, _, H, W = x.shape
        x_ori = x
        x_freq = torch.fft.rfft2(self.pre(x)+1e-8, norm='backward')
        y_freq = torch.fft.rfft2(self.pre(y)+1e-8, norm='backward')
        plus_freq = self.mix(x_freq, y_freq)

        plus_amp = torch.abs(plus_freq)
        plus_pha = torch.angle(plus_freq)
        x_amp = torch.abs(x_freq)
        # x_pha = torch.angle(x_freq)
        # y_amp = torch.abs(y_freq)
        # y_pha = torch.angle(y_freq)

        real = x_amp * torch.cos(plus_pha)+1e-8
        imag = x_amp * torch.sin(plus_pha)+1e-8
        x_out = torch.complex(real, imag)+1e-8

        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')

        x_out = self.zero_conv(x_out)

        if self.residual:
            return x_out + x_ori
        return x_out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.vgg = VGG19()

    def forward(self, x):
        x_feats = self.vgg(x)
        x_feats = [x_feats['relu1_1'],
                   x_feats['relu2_1'],
                   x_feats['relu3_1'],
                   x_feats['relu4_1'],
                   x_feats['relu5_1']]
        return x_feats


class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.scale4 = nn.Sequential(        # [B,512,H,W] -> [B,256,H,W]
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU())
        self.scale3 = nn.Sequential(        # [B,256,H,W] -> [B,256,2H,2W]
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU()
            )
        self.scale2 = nn.Sequential(        # [B,256,2H,2W] -> [B,128,4H,4W]
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU()
            )
        self.scale1 = nn.Sequential(        # [B,128,4H,4W] -> [B,64,8H,8W]
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU())
        self.out=nn.Sequential(             # [B,64,8H,8W] -> [B,2,8H,8W]
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)))
        
        self.skipconnection2=SkipConnection(128)
        self.skipconnection1=SkipConnection(64)
        
    def forward(self, x, y, z):
        """
        x: [B,512,H,W]
        y: [B,128,4H,4W]
        z: [B,64,8H,8W]
        """
        out4 = self.scale4(x)       # [B,512,H,W] -> [B,256,H,W] (H=W=32)
        out3 = self.scale3(out4)    # [B,256,H,W] -> [B,256,2H,2W]
        out2 = self.scale2(out3)    # [B,256,2H,2W] -> [B,128,4H,4W]

        # out2 = out2                             # Artistic Style Transfer
        # out2 = out2 + y                         # Photo-realistic Style Transfer
        out2 = self.skipconnection2(out2, y)    # Fourier Priors-Guided Style Transfer
        out1 = self.scale1(out2)    # [B,128,4H,4W] -> [B,64,8H,8W]

        # out1 = out1                             # Artistic Style Transfer
        # out1 = out1 + z                         # Photo-realistic Style Transfer
        out1 = self.skipconnection1(out1, z)    # Fourier Priors-Guided Style Transfer
        out  = self.out(out1)       # [B,64,8H,8W] -> [B,3,8H,8W]

        return out

class StyleAttentionNet(nn.Module):
    """
    Paper: Arbitrary Style Transfer with Style-Attentional Networks
    """
    def __init__(self, in_planes):
        super(StyleAttentionNet, self).__init__()

        self.query_conv = nn.Conv2d(in_planes, in_planes, (1, 1))   # query
        self.key_conv = nn.Conv2d(in_planes, in_planes, (1, 1))     # key
        self.value_conv = nn.Conv2d(in_planes, in_planes, (1, 1))   # value

        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))

    def forward(self, content, style):
        proj_query = self.query_conv(mean_variance_norm(content))
        proj_key = self.key_conv(mean_variance_norm(style))
        proj_value = self.value_conv(style)

        b, c, h, w = proj_query.size()
        proj_query = proj_query.view(b, -1, w * h).permute(0, 2, 1)
        b, c, h, w = proj_key.size()
        proj_key = proj_key.view(b, -1, w * h)
        b, c, h, w = proj_value.size()
        proj_value = proj_value.view(b, -1, w * h)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        out = self.out_conv(out)
        out += content

        return out

class TransformNet(nn.Module):
    """
    style transfer module
    """
    def __init__(self, in_planes=512):
        super(TransformNet, self).__init__()
        self.ab4_1 = StyleAttentionNet(in_planes=in_planes)
        self.ab5_1 = StyleAttentionNet(in_planes=in_planes)
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))
        # self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, content4_1, style4_1, content5_1, style5_1):
        """
        content4_1, style4_1: [B,C,H4,W4]
        content5_1, style5_1: [B,C,H5,W5]
        return: [B,C,H4,W4]
        """
        self.upsample5_1 = nn.Upsample(
            size=(content4_1.size()[2], content4_1.size()[3]), 
            mode='nearest')     # -> [B,C,H4,W4]
        
        out = self.merge_conv(
            self.merge_conv_pad(
                self.ab4_1(content4_1, style4_1) + self.upsample5_1(self.ab5_1(content5_1, style5_1))
            )
        )
        return out

class MDSTNet(nn.Module):
    """
    C2H: Clear to Haze
    """
    def __init__(self):
        super(MDSTNet, self).__init__()

        self.encoder = Encoder()        # encoder
        self.transform = TransformNet() # style transfer module
        self.decoder = Decoder()        # decoder

    def forward(self, content, style):
        """
        content:        content image
        style:          style image
        return:
        out:            stylized image
        out_feats:      out's feature list
        content_feats:  content's feature list
        style_feats:    style's feature list
        c_c:            stylized image (c+c)
        s_s:            stylized image (s+s)
        c_c_feats:      c_c's feature list
        s_s_feats:      s_s's feature list
        """
        content_feats = self.encoder(content)
        style_feats = self.encoder(style)
        stylized = self.transform(content_feats[3], style_feats[3], content_feats[4], style_feats[4])
        
        out = self.decoder(stylized, content_feats[1], content_feats[0])
        out_feats = self.encoder(out)

        """FOR IDENTITY LOSSES"""
        c_c = self.decoder((self.transform(content_feats[3], content_feats[3], content_feats[4], content_feats[4])), content_feats[1], content_feats[0])
        s_s = self.decoder((self.transform(style_feats[3], style_feats[3], style_feats[4], style_feats[4])), style_feats[1], style_feats[0])
        c_c_feats = self.encoder(c_c)
        s_s_feats = self.encoder(s_s)

        return out, out_feats, content_feats, style_feats, c_c, s_s, c_c_feats, s_s_feats


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net=MDSTNet().to(device)
    print(net)
    x=torch.randn(1,3,256,256).to(device)
    y=torch.randn(1,3,256,256).to(device)
    res=net(x,y)
    print(res[0].shape) # torch.Size([1, 3, 256, 256])
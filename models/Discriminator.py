from torch import nn
import torch

class MultiDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(MultiDiscriminator, self).__init__()

        def discriminator_block(in_nc, out_nc, normalize=True):
            # [B,in_nc,H,W] -> [B,out_nc,H//2,W//2]
            layers = [nn.Conv2d(in_nc, out_nc, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_nc))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.models = nn.ModuleList([
            nn.Sequential(
                *discriminator_block(in_channels, 64, normalize=False), # [B,3,H,W] -> [B,64,H//2,W//2]
                *discriminator_block(64, 128),                          # -> [B,128,H//4,W//4]
                *discriminator_block(128, 256),                         # -> [B,256,H//8,W//8]
                *discriminator_block(256, 512),                         # -> [B,512,H//16,W//16]
                nn.Conv2d(512, 1, 3, padding=1)                         # -> [B,1,H//16,W//16]
            )
            for _ in range(3)
        ])

        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False) # [B,C,H,W] -> [B,C,(H+1)//2,(W+1)//2]

    def compute_loss(self, x, gt):
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
            # print(f"Before downsample {i}: {x.shape}")
            x = self.downsample(x)
            # print(f"After downsample {i}: {x.shape}")
        return outputs

class MultiDiscriminator_feature(nn.Module):
    def __init__(self, in_channels=256):
        super(MultiDiscriminator_feature, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.models = nn.ModuleList([
            nn.Sequential(
                *discriminator_block(in_channels, 256, normalize=False),# [B,256,H,W] -> [B,256,H//2,W//2]
                *discriminator_block(256, 512),                         # -> [B,512,H//4,W//4]
                nn.Conv2d(512, 1, 3, padding=1)                         # -> [B,1,H//4,W//4]
            )
            for _ in range(3)
        ])

        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False) # [B,C,H,W] -> [B,C,(H+1)//2,(W+1)//2]

    def compute_loss(self, x, gt):
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
            # print(f"Before downsample {i}: {x.shape}")
            x = self.downsample(x)
            # print(f"After downsample {i}: {x.shape}")
        return outputs

class MultiDiscriminator_local(nn.Module):
    def __init__(self, in_channels=3):
        super(MultiDiscriminator_local, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.models = nn.ModuleList([
            nn.Sequential(
                *discriminator_block(in_channels, 64, normalize=False), # [B,3,H,W] -> [B,64,H//2,W//2]
                *discriminator_block(64, 128),                          # -> [B,128,H//4,W//4]
                nn.Conv2d(128, 1, 3, padding=1)                         # -> [B,1,H//4,W//4]
            )
            for _ in range(3)
        ])

        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False) # [B,C,H,W] -> [B,C,(H+1)//2,(W+1)//2]

    def compute_loss(self, x, gt):
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
            # print(f"Before downsample {i}: {x.shape}")
            x = self.downsample(x)
            # print(f"After downsample {i}: {x.shape}")
        return outputs

if __name__ == "__main__":
    model = MultiDiscriminator()
    print(model)


"""
MultiDiscriminator(
  (models): ModuleList(
    (0-2): 3 x Sequential(
      (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
      (1): LeakyReLU(negative_slope=0.2, inplace=True)
      (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
      (3): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (4): LeakyReLU(negative_slope=0.2, inplace=True)
      (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
      (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (7): LeakyReLU(negative_slope=0.2, inplace=True)
      (8): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
      (9): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (10): LeakyReLU(negative_slope=0.2, inplace=True)
      (11): Conv2d(512, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
  )
  (downsample): AvgPool2d(kernel_size=3, stride=2, padding=[1, 1])
)
"""
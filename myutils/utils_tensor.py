import os
import cv2
import torch
import numpy as np


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()  # (B,C,H,W)
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()  # (B,C,H,W)
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

def tensor_to_image_moving(tensor):
    image_tensor = (tensor * 0.5) + 0.5    # [-1,1] -> [0,1]
    return image_tensor

def tensor_to_image_minmax(tensor):
    image_tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # [-n,m] -> [0,1]
    return image_tensor

def save_featuremap(tensor, saved_dir='./', name='feature_map'):
    """
    tensor (Tensor): [B=1,C,H,W]
    saved_dir (str): image save directory
    name (str):      image name (w/o suffix)
    """
    assert tensor.shape[0] == 1, "The number of channels of the input tensor must be 1"

    tensor = torch.mean(tensor, dim=1)                      # [1,C,H,W] -> [1,H,W]
    inp = tensor.squeeze().detach().cpu().numpy()           # [1,H,W] -> [H,W]
    inp = (inp - np.min(inp)) / (np.max(inp) - np.min(inp))
    os.makedirs(saved_dir, exist_ok=True)
    saved_path = os.path.join(saved_dir, name + '.png')
    inp = cv2.applyColorMap(np.uint8(inp * 255.0),cv2.COLORMAP_JET)
    cv2.imwrite(saved_path, inp)

    print(f'INFO:   Saved to: {saved_path}')


if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256)
    save_featuremap(x)
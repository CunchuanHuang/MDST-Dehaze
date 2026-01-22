import sys
sys.path.append("..")

import numpy as np
import cv2
import torch

from metrics import psnr, ssim, psnr_pt, ssim_pt, psnr_pt2np


def test(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    img1 = img1 / 255
    img2 = img2 / 255

    print('================== Numpy ==========================')
    p1 = psnr(img1, img2)
    s1 = ssim(img1, img2)
    print(f'PSNR: {p1} dB')
    print(f'SSIM: {s1}')

    print('================== PyTorch (CPU) ==================')
    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0) # [1,C,H,W]
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0) # [1,C,H,W]

    device = torch.device("cpu")
    img1 = img1.to(device)
    img2 = img2.to(device)

    p2 = psnr_pt(img1, img2)
    s2 = ssim_pt(img1, img2)
    p2_pt2np = psnr_pt2np(img1, img2)
    print(f'PSNR: {p2_pt2np} dB')
    print(f'PSNR: {p2.item()}')
    print(f'SSIM: {s2.item()}')

    print('================== PyTorch (GPU) ==================')
    device = torch.device("cuda")
    img1 = img1.to(device)
    img2 = img2.to(device)

    p3 = psnr_pt(img1, img2)
    s3 = ssim_pt(img1, img2)
    p3_pt2np = psnr_pt2np(img1, img2)
    print(f'PSNR: {p3_pt2np} dB')
    print(f'PSNR: {p3.item()}')
    print(f'SSIM: {s3.item()}')


if __name__ == '__main__':
    test('test_img1.png', 'test_img2.png')
import numpy as np
import cv2
import torch
import torch.nn.functional as F


####################################### Numpy version #######################################

def psnr(img1, img2):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray): Images with range [0, 1].
        img2 (ndarray): Images with range [0, 1].

    Returns:
        float: psnr result. (numpy.float64)
    """
    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return np.inf
    return 20 * np.log10(1 / np.sqrt(mse))


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 1] with order 'HWC'.
        img2 (ndarray): Images with range [0, 1] with order 'HWC'.

    Returns:
        float: ssim result. (numpy.float64)
    """
    C1 = (0.01 * 1) ** 2
    C2 = (0.03 * 1) ** 2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def ssim(img1, img2):
    """Calculate SSIM (structural similarity).

    Ref:
    Paper: Image quality assessment: From error visibility to structural similarity.

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.
    
    For three-channel images, SSIM is calculated for each channel and then averaged.

    Args:
        img1 (ndarray): Images with range [0, 1].
        img2 (ndarray): Images with range [0, 1].
        
    Returns:
        float: ssim result. (numpy.float64)
    """
    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    assert img1.shape[2] == 3
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))

    return np.mean(ssims)


###################################### PyTorch version ######################################

def psnr_pt2np(img1, img2):
    """Calculate PSNR (Peak Signal-to-Noise Ratio) (PyTorch version).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        
    Returns:
        float: PSNR result. (numpy.float64)
    """
    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    img1 = img1.cpu().detach().numpy().astype(np.float64)
    img2 = img2.cpu().detach().numpy().astype(np.float64)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return np.inf
    return 20 * np.log10(1 / np.sqrt(mse))


def psnr_pt(img1, img2):
    """Calculate PSNR (Peak Signal-to-Noise Ratio) (PyTorch version).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        
    Returns:
        float: PSNR result. (Tensor, type(.item())=float)
    """
    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    img1 = img1.to(torch.float64)
    img2 = img2.to(torch.float64)

    mse = torch.mean((img1 - img2)**2, dim=[1, 2, 3])
    return 10. * torch.log10(1. / (mse + 1e-8))


def _ssim_pt(img1, img2):
    """Calculate SSIM (structural similarity) (PyTorch version).

    It is called by func:`calculate_ssim_pt`.

    Args:
        img1 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).

    Returns:
        float: SSIM result. (Tensor, type(.item())=float)
    """
    C1 = (0.01 * 1) ** 2
    C2 = (0.03 * 1) ** 2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    window = torch.from_numpy(window).view(1, 1, 11, 11).expand(img1.size(1), 1, 11, 11).to(img1.dtype).to(img1.device)

    mu1 = F.conv2d(img1, window, stride=1, padding=0, groups=img1.shape[1])
    mu2 = F.conv2d(img2, window, stride=1, padding=0, groups=img2.shape[1])

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, stride=1, padding=0, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=0, groups=img1.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, stride=1, padding=0, groups=img1.shape[1]) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean([1, 2, 3])


def ssim_pt(img1, img2):
    """Calculate SSIM (structural similarity) (PyTorch version).

    Ref:
    Paper: Image quality assessment: From error visibility to structural similarity.

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then averaged.

    Args:
        img1 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        
    Returns:
        float: SSIM result. (Tensor, type(.item())=float)
    """
    assert img1.shape == img2.shape, (f'Image shapes are different: {img1.shape}, {img2.shape}.')
    img1 = img1.to(torch.float64)
    img2 = img2.to(torch.float64)

    ssim = _ssim_pt(img1, img2)

    return ssim



####################################### JS Divergence #######################################
def calculate_js_divergence(P, Q):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    P = P.view(P.size(0), -1).to(device)
    Q = Q.view(Q.size(0), -1).to(device)
    P = F.softmax(P, dim=1)
    Q = F.softmax(Q, dim=1)
    M = 0.5 * (P + Q)
    kl1 = F.kl_div(P.log(), M, reduction='batchmean')
    kl2 = F.kl_div(Q.log(), M, reduction='batchmean')
    jsd = 0.5 * kl1 + 0.5 * kl2
    return jsd.item()
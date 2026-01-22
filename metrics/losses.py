import torch.nn as nn
import torch

from models.VGG19 import VGG19
from myutils.utils_tensor import mean_variance_norm, calc_mean_std

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PerceptualLoss(nn.Module):
    """
    Perceptual loss, VGG-based
    Paper: https://arxiv.org/abs/1603.08155
    """
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGG19()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        """
        x:  input image
        y:  target image
        """
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        loss = 0
        loss += self.weights[0] * self.l1(x_vgg['relu1_1'], y_vgg['relu1_1'].detach())
        loss += self.weights[1] * self.l1(x_vgg['relu2_1'], y_vgg['relu2_1'].detach())
        loss += self.weights[2] * self.l1(x_vgg['relu3_1'], y_vgg['relu3_1'].detach())
        loss += self.weights[3] * self.l1(x_vgg['relu4_1'], y_vgg['relu4_1'].detach())
        loss += self.weights[4] * self.l1(x_vgg['relu5_1'], y_vgg['relu5_1'].detach())

        return loss

class ContrastLoss(nn.Module):
    """
    Contrastive loss, VGG-based
    """
    def __init__(self):
        super(ContrastLoss, self).__init__()

        self.vgg = VGG19()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, a, p, inp, candidate_list, weight):
        """
        a:              Anchor image
        p:              Positive image
        inp:            Input image
        candidate_list: Negative candidates
        weight:         weight list
        """
        a_vgg, p_vgg, inp_vgg = self.vgg(a), self.vgg(p), self.vgg(inp)
        n1_vgg, n2_vgg, n3_vgg, n4_vgg, n5_vgg = (
            self.vgg(candidate_list[0].to(device)),
            self.vgg(candidate_list[1].to(device)),
            self.vgg(candidate_list[2].to(device)),
            self.vgg(candidate_list[3].to(device)),
            self.vgg(candidate_list[4].to(device))
        )

        loss = 0
        layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        n1_weight, n2_weight, n3_weight, n4_weight, n5_weight, inp_weight = weight
        for i, layer in enumerate(layers):
            d_ap  = self.l1(a_vgg[layer], p_vgg[layer].detach())
            d_inp = self.l1(a_vgg[layer], inp_vgg[layer].detach())
            d_an1 = self.l1(a_vgg[layer], n1_vgg[layer].detach())
            d_an2 = self.l1(a_vgg[layer], n2_vgg[layer].detach())
            d_an3 = self.l1(a_vgg[layer], n3_vgg[layer].detach())
            d_an4 = self.l1(a_vgg[layer], n4_vgg[layer].detach())
            d_an5 = self.l1(a_vgg[layer], n5_vgg[layer].detach())

            # contrastive = d_ap / (d_inp + 1e-7) # without stylized database
            contrastive = d_ap / (d_an1 * n1_weight + d_an2 * n2_weight + d_an3 * n3_weight + d_an4 * n4_weight + d_an5 * n5_weight + d_inp * inp_weight + 1e-7)
            loss += self.weights[i] * contrastive
        
        return loss

class ContrastLoss_pixel(nn.Module):
    """
    Contrastive loss, pixel-based
    """
    def __init__(self):
        super(ContrastLoss_pixel, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, a, p, inp, candidate_list, weight):
        n1_weight, n2_weight, n3_weight, n4_weight, n5_weight, inp_weight = weight
        d_ap = self.l1(a, p.detach())
        d_inp = self.l1(a, inp.detach())
        d_an1 = self.l1(a, candidate_list[0].to(device).detach())
        d_an2 = self.l1(a, candidate_list[1].to(device).detach())
        d_an3 = self.l1(a, candidate_list[2].to(device).detach())
        d_an4 = self.l1(a, candidate_list[3].to(device).detach())
        d_an5 = self.l1(a, candidate_list[4].to(device).detach())

        loss = d_ap / (d_an1 * n1_weight + d_an2 * n2_weight + d_an3 * n3_weight + d_an4 * n4_weight + d_an5 * n5_weight + d_inp * inp_weight + 1e-7)
        
        return loss


################################################ loss function ################################################

def calc_content_loss(input, target, norm=False):
    """content loss"""
    loss = nn.MSELoss()
    if (norm == False):
        loss_feat0 = loss(input[0], target[0])
        loss_feat1 = loss(input[1], target[1])
        loss_feat2 = loss(input[2], target[2])
        loss_feat3 = loss(input[3], target[3])
        loss_feat4 = loss(input[4], target[4])
        content_loss = loss_feat0 + loss_feat1 + loss_feat2 + loss_feat3 + loss_feat4
    else:
        loss_feat3 = loss(mean_variance_norm(input[3]), mean_variance_norm(target[3]))
        loss_feat4 = loss(mean_variance_norm(input[4]), mean_variance_norm(target[4]))
        content_loss = loss_feat3 + loss_feat4
    return content_loss

def calc_style_loss(input, target):
    """style loss"""
    loss = nn.MSELoss()
    style_loss = 0
    for i in range(len(input)):
        input_mean, input_std = calc_mean_std(input[i])
        target_mean, target_std = calc_mean_std(target[i])
        style_loss += (loss(input_mean, target_mean) + loss(input_std, target_std))
    return style_loss

def calc_phase_loss(img1, img2, method='cos'):
    """phase loss
    Args:
        img1 (Tensor): Images with shape (B,C,H,W)
        img2 (Tensor): Images with shape (B,C,H,W)
        mehtod: str, ['cos', 'mse']
    
    Returns:
        phase_loss (Tensor): Phase loss result.
    """
    fft1 = torch.fft.fft2(img1, dim=(-2, -1))
    fft2 = torch.fft.fft2(img2, dim=(-2, -1))
    # amp2 = torch.abs(fft2)
    phase1 = torch.angle(fft1)
    phase2 = torch.angle(fft2)
    
    if method == 'cos':
        cos_phase = torch.cos(phase1 - phase2)
        phase_loss = -cos_phase.mean()
    if method == 'mse':
        loss = nn.MSELoss()
        phase_loss = loss(img1, img2)
    return phase_loss

def calc_self_perceptual_loss(output:list, target:list):
    """self-perceptual loss
    output: [layer1, layer2, latent]
    target: [layer1, layer2, latent]
    """
    self_perceptual_loss = 0
    weight = [0.1, 0.1, 1]
    loss_l1 = nn.L1Loss()
    for i in range(len(output)):
        self_perceptual_loss += loss_l1(output[i], target[i]) * weight[i]
    return self_perceptual_loss

def calc_perceptual_loss(output, target):
    """perceptual loss"""
    loss = nn.MSELoss()
    perceptual_loss = 0
    weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
    for i in range(len(output)):
        perceptual_loss += loss(output[i], target[i]) * weights[i]
    return perceptual_loss


def gram_matrix(feature_map):
    assert len(feature_map.shape) == 3  # 3D Tensor [C,H,W]
    C, H, W = feature_map.shape
    features = feature_map.view(C, H * W)
    G = torch.mm(features, features.t())
    G = G.div(C * H * W)
    return G

def calc_style_loss_gram(input, target):
    # assert (input.shape == target.shape)
    calc_loss = nn.MSELoss()
    loss = 0
    for i in range(len(input)):
        loss_layer = 0
        for j in range(len(input[i])):
            gi = gram_matrix(input[i][j])
            gt = gram_matrix(target[i][j])
            loss_layer += calc_loss(gi, gt)
        loss += loss_layer
    return loss
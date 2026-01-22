import sys
sys.path.append('..')
import os
import json
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from configs import configs_trainC2H as configs
from models.MDSTNet import MDSTNet
from models.Discriminator import MultiDiscriminator, MultiDiscriminator_feature
from metrics.losses import calc_content_loss, calc_style_loss, calc_phase_loss
from myutils.dataloader import Dataloader
from myutils import utils_visual

import torch.backends.cudnn as cudnn
from PIL import Image
from PIL import ImageFile
import warnings
cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')

"""
../results/
 |-- stylized_database/DATASET_NAME/    # Stylized database
 |    |-- reliable_style_bank
 |    |    |-- xxx.png
 |    |    |-- ...
 |    |-- stylizedEPOCH1
 |    |    |-- xxx.png
 |    |    |-- ...
 |    |-- stylizedEPOCH2
 |    |-- stylizedEPOCH3
 |    |-- stylizedEPOCH4
 |    |-- stylizedEPOCH5
 |    |-- stylizedEPOCH6
 |-- checkpoints/DATASET_NAME/          # Model
 |    |-- C2H.pth                       # C2H (MDST)
 |-- logs/DATASET_NAME/                 # Loss log
 |    |-- train_losses_C2H.json
"""

if __name__ == "__main__":
    train_dataset = Dataloader(haze_dir=configs.style_dir, clear_dir=configs.content_dir, transform=configs.transforms_train, unaligned=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=configs.n_threads)

    logger_train = utils_visual.Logger(configs.num_epoch, len(train_loader))

    # 加载模型
    Gen = MDSTNet().to(configs.device)
    D_P = MultiDiscriminator().to(configs.device)
    D_F = MultiDiscriminator_feature().to(configs.device)
    print('The models are initialized successfully!')

    Gen.train()
    D_P.train()
    D_F.train()

    total_params_Gen = sum(p.numel() for p in Gen.parameters() if p.requires_grad)
    total_params_D_P = sum(p.numel() for p in D_P.parameters() if p.requires_grad)
    total_params_D_F = sum(p.numel() for p in D_F.parameters() if p.requires_grad)
    print("Total params (Gen): ==> {}".format(total_params_Gen))
    print("Total params (D_P): ==> {}".format(total_params_D_P))
    print("Total params (D_F): ==> {}".format(total_params_D_F))
    print("Total params: ==> {}".format(total_params_Gen + total_params_D_P + total_params_D_F))

    label_real = torch.ones([1], dtype=torch.float, requires_grad=False).to(configs.device)
    label_fake = torch.zeros([1], dtype=torch.float, requires_grad=False).to(configs.device)

    opt_Gen = optim.Adam(Gen.parameters(), lr=configs.lr, betas=(0.5, 0.999))
    opt_D_P = optim.Adam(D_P.parameters(), lr=configs.lr, betas=(0.5, 0.999))
    opt_D_F = optim.Adam(D_F.parameters(), lr=configs.lr, betas=(0.5, 0.999))

    lr_scheduler_Gen = torch.optim.lr_scheduler.LambdaLR(opt_Gen, lr_lambda=utils_visual.LambdaLR(configs.num_epoch, configs.start_epoch, configs.decay_epoch).step)
    lr_scheduler_D_P = torch.optim.lr_scheduler.LambdaLR(opt_D_P, lr_lambda=utils_visual.LambdaLR(configs.num_epoch, configs.start_epoch, configs.decay_epoch).step)
    lr_scheduler_D_F = torch.optim.lr_scheduler.LambdaLR(opt_D_F, lr_lambda=utils_visual.LambdaLR(configs.num_epoch, configs.start_epoch, configs.decay_epoch).step)

    mse_loss = nn.MSELoss().to(configs.device)

    #! 开始训练
    for epoch in range(configs.start_epoch, configs.num_epoch + 1):

        for i, batch in enumerate(train_loader):
            style = batch[0].to(configs.device)     # style (haze) image, Tenosr [B,C,H,W]
            content = batch[1].to(configs.device)   # content (clear) image, Tenosr [B,C,H,W]
            image_name= batch[2]                    # content (clear) image name, tuple [B]
            
            out, out_feats, content_feats, style_feats, c_c, s_s, c_c_feats, s_s_feats = Gen(content, style)
            
            loss_gan_d_p = D_P.compute_loss(style, label_real) + D_P.compute_loss(out.detach(), label_fake)                     # Adversarial Loss
            loss_gan_d_f = D_F.compute_loss(style_feats[2], label_real) + D_F.compute_loss(out_feats[2].detach(), label_fake)   # Adversarial Loss

            opt_D_P.zero_grad()
            loss_gan_d_p.backward()
            opt_D_P.step()

            opt_D_F.zero_grad()
            loss_gan_d_f.backward()
            opt_D_F.step()


            loss_gan_g_p = D_P.compute_loss(out, label_real) * configs.gan_p_weight
            loss_gan_g_f = D_F.compute_loss(out_feats[2], label_real) * configs.gan_f_weight
            
            loss_c =  calc_content_loss(out_feats, content_feats, norm=True) * configs.content_weight
            loss_s = calc_style_loss(out_feats, style_feats) * configs.style_weight
            
            loss_identity_p = (mse_loss(c_c, content) + mse_loss(s_s, style)) * configs.identity_p_weight
            loss_identity_f = (calc_content_loss(c_c_feats, content_feats) + calc_content_loss(s_s_feats, style_feats)) * configs.identity_f_weight
            loss_identity_pha = (calc_phase_loss(c_c, content, 'cos') + calc_phase_loss(s_s, style, 'cos')) * configs.identity_pha_weight
            
            loss = loss_gan_g_p + loss_gan_g_f + loss_c + loss_s + loss_identity_p + loss_identity_f + loss_identity_pha

            opt_Gen.zero_grad()
            loss.backward()
            opt_Gen.step()

            # initialize and updata reliable_style_bank
            if epoch <= configs.decay_epoch:
                utils_visual.initialize_reliable_style_bank(configs.save_stylized_dir + 'reliable_style_bank/', image_name)
            elif epoch > configs.decay_epoch:
                utils_visual.update_reliable_style_bank(out, style, configs.save_stylized_dir + 'reliable_style_bank/', image_name)
            
            epoch_losses = logger_train.output_log(
                losses={
                    'loss_c': loss_c,
                    'loss_s': loss_s,
                    'loss_identity_p': loss_identity_p,
                    'loss_identity_f': loss_identity_f,
                    'loss_identity_pha': loss_identity_pha,
                    'loss_gan_g_p': loss_gan_g_p,
                    'loss_gan_g_f': loss_gan_g_f,
                    'loss': loss
                },
                images={
                    'Style': style,
                    'Content': content,
                    'Output': out
                }
            )

            if epoch in configs.save_list:
                for i in range(out.shape[0]):
                    save_path = configs.save_stylized_dir + f"stylized{epoch}/{image_name[i]}"
                    save_dir = os.path.dirname(save_path)
                    os.makedirs(save_dir, exist_ok=True)
                    save_image(out[i], save_path)

        lr_scheduler_Gen.step()
        lr_scheduler_D_P.step()
        lr_scheduler_D_F.step()

    assert epoch_losses!=None, 'epoch_losses is None.'
    os.makedirs(configs.save_log_dir, exist_ok=True)
    with open(configs.save_log_dir + "train_losses_C2H.json", "w") as file:
        json.dump(epoch_losses, file)

    os.makedirs(configs.save_model_dir, exist_ok=True)
    torch.save(Gen.state_dict(), configs.save_model_dir + "C2H.pth")
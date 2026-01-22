import sys
sys.path.append('..')
import time
import numpy as np
import torch
import os
import json
from torchvision.utils import save_image
from torch import optim, nn
from torch.utils.data import DataLoader

from metrics.metrics import psnr_pt2np as calc_psnr, ssim_pt as calc_ssim
from myutils.dataloader import Dataloader
from configs import configs_trainC2C as configs
from myutils import utils_visual
from models import DSNet
from metrics.losses import PerceptualLoss

import warnings
import torch.backends.cudnn as cudnn
from PIL import Image
from PIL import ImageFile
cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')


"""
../results/
 |-- checkpoints/DATASET_NAME/  # Model
 |    |-- Ten.pth               # Best PSNR-based
 |    |-- Tde.pth               # Best PSNR-based
 |    |-- Ten_Best_SSIM.pth     # Best SSIM-based
 |    |-- Ten_Best_SSIM.pth     # Best SSIM-based
 |-- recimg/DATASET_NAME/       # Reconstructed images
 |    |-- xxx.png
 |    |-- ...
 |-- logs/
 |    |-- DATASET_NAME/         # Loss log
 |    |    |-- train_loss_C2C.json
 |    |-- logs.txt              # Log
"""

if __name__ == "__main__":
    train_dataset = Dataloader(haze_dir=configs.clear_dir, clear_dir=configs.clear_dir, transform=configs.transforms_train, unaligned=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=configs.n_threads)

    logger_train = utils_visual.Logger(configs.num_epoch, len(train_loader))

    Ten = DSNet.Encoder().to(configs.device)
    Tde = DSNet.Decoder().to(configs.device)

    print('The models are initialized successfully!')

    Ten.train()
    Tde.train()

    total_params_Ten = sum(p.numel() for p in Ten.parameters() if p.requires_grad)
    total_params_Tde = sum(p.numel() for p in Tde.parameters() if p.requires_grad)
    print("Total_params (Ten): ==> {}".format(total_params_Ten))
    print("Total_params (Tde): ==> {}".format(total_params_Tde))
    print("Total_params: ==> {}".format(total_params_Ten + total_params_Tde))

    opt_Ten = optim.Adam(Ten.parameters(), lr=configs.lr, betas=(0.5, 0.999))
    opt_Tde = optim.Adam(Tde.parameters(), lr=configs.lr, betas=(0.5, 0.999))

    lr_scheduler_Ten = torch.optim.lr_scheduler.LambdaLR(opt_Ten, lr_lambda=utils_visual.LambdaLR(configs.num_epoch, configs.start_epoch, configs.decay_epoch).step)
    lr_scheduler_Tde = torch.optim.lr_scheduler.LambdaLR(opt_Tde, lr_lambda=utils_visual.LambdaLR(configs.num_epoch, configs.start_epoch, configs.decay_epoch).step)

    loss_l1 = nn.L1Loss().to(configs.device)
    loss_per = PerceptualLoss().to(configs.device)

    max_ssim = 0
    max_psnr = 0

    for epoch in range(configs.start_epoch, configs.num_epoch + 1):
        ssims = []
        psnrs = []

        start_time = time.time()

        images_output = []
        images_name = []

        for i, batch in enumerate(train_loader):

            x = batch[0].to(configs.device)         # clear image
            image_name = batch[2][0]                # clear image name
            
            feat, x1, x2 = Ten(x)
            output = Tde(feat, x1, x2)

            images_output.append(output)
            images_name.append(image_name)
            
            loss_L1 =  loss_l1(output, x) * configs.l1_weight
            loss_Per = loss_per(output, x) * configs.perceptual_weight
            loss = loss_L1 + loss_Per

            opt_Ten.zero_grad()
            opt_Tde.zero_grad()
            loss.backward()
            opt_Ten.step()
            opt_Tde.step()

            psnr1 = calc_psnr(output, x)
            ssim1 = calc_ssim(output, x).item()
            psnrs.append(psnr1)
            ssims.append(ssim1)

            epoch_losses = logger_train.output_log(
                losses={
                    'loss_L1': loss_L1,
                    'loss_Perceptual': loss_Per,
                    'loss': loss,
                    'Val_PSNR': torch.tensor(psnr1),
                    'Val_SSIM': torch.tensor(ssim1),
                    'Max_PSNR': torch.tensor(max_psnr),
                    'Max_SSIM': torch.tensor(max_ssim)
                },
                images={
                    'Input': x,
                    'Output': output
                }
            )

        one_epoch_time = time.time() - start_time
        psnr_eval = np.mean(psnrs)
        ssim_eval = np.mean(ssims)

        if psnr_eval > max_psnr:
            max_psnr = max(max_psnr, psnr_eval)
            os.makedirs(configs.save_model_dir, exist_ok=True)
            torch.save(Ten.state_dict(), configs.save_model_dir + "Ten.pth")
            torch.save(Tde.state_dict(), configs.save_model_dir + "Tde.pth")
            os.makedirs(configs.save_image_dir, exist_ok=True)
            for i in range(len(images_name)):
                save_image(images_output[i], configs.save_image_dir + images_name[i])

        if ssim_eval > max_ssim:
            max_ssim = max(max_ssim, ssim_eval)
            os.makedirs(configs.save_model_dir, exist_ok=True)
            torch.save(Ten.state_dict(), configs.save_model_dir + "Ten_Best_SSIM.pth")
            torch.save(Tde.state_dict(), configs.save_model_dir + "Tde_Best_SSIM.pth")
            # os.makedirs(configs.save_image_dir, exist_ok=True)
            # for i in range(len(images_name)):
            #     save_image(images_output[i], configs.save_image_dir + images_name[i])

        lr_scheduler_Ten.step()
        lr_scheduler_Tde.step()

        print(f'Epoch [{epoch:03d}/{configs.num_epoch:03d}] -- Val_PSNR: {psnr_eval:.2f} | Val_SSIM: {ssim_eval:.4f} | Max_PSNR {max_psnr:.2f} | Max_SSIM: {max_ssim:.4f}')

        os.makedirs(os.path.dirname(configs.save_log_path), exist_ok=True)
        with open(configs.save_log_path, 'a') as f:
            line = 'Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Val_PSNR: {4:.2f}, Val_SSIM: {5:.4f}'.format(
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                one_epoch_time, epoch, configs.num_epoch, psnr_eval, ssim_eval
            )
            print(line, file=f)

    assert epoch_losses!=None, 'epoch_losses is None.'
    os.makedirs(configs.save_log_dir, exist_ok=True)
    with open(configs.save_log_dir + "train_losses_C2C.json", "w") as file:
        json.dump(epoch_losses, file)
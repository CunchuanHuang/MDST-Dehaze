import sys
sys.path.append('..')
import os
import numpy as np
import json
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from configs import configs_trainH2C as configs
from myutils.dataloader import Dataloader, Dataloader_TrainH2C
from models import DSNet
from myutils import utils_visual
from metrics.losses import PerceptualLoss, ContrastLoss, ContrastLoss_pixel, calc_self_perceptual_loss
from metrics.metrics import psnr_pt2np as calc_psnr, ssim_pt as calc_ssim

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
 |    |-- Sen.pth               # Final epoch (unsupervised)
 |    |-- Sen_Best_PSNR.pth     # Best PSNR-base (only when semi-supervised)
 |    |-- Sen_Best_SSIM.pth     # Best SSIM-base (only when semi-supervised)
 |-- dehaze/DATASET_NAME/       # Dehazing images
 |    |-- val_psnr              # Best PSNR-based (only when semi-supervised)
 |    |    |-- xxx.png
 |    |    |-- ...
 |    |-- val_ssim              # Best SSIM-based (only when semi-supervised)
 |-- logs/DATASET_NAME/
 |    |-- train_losses_H2C.json # Loss log
 |    |-- val_losses_H2C.json   # Loss log (only when semi-supervised)
"""

if __name__ == "__main__":
    def curriculum_weight(difficulty):
        diff_list = [18, 20, 25, 27, 32]
        weights = [(1 + configs.cl_lambda) if difficulty > x else (1 - configs.cl_lambda) for x in diff_list]
        weights.append(len(diff_list))
        new_weights = [i / sum(weights) for i in weights]
        return new_weights
    
    train_dataset = Dataloader_TrainH2C(configs.haze_dir, configs.clear_dir, configs.stylized_dir, transform=configs.transforms_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=configs.n_threads)
    val_dataset = Dataloader(configs.haze_val_dir, configs.clear_val_dir, transform=configs.transforms_train, unaligned=False, model='val')
    val_loader = DataLoader(dataset=val_dataset, batch_size=configs.batch_size//configs.batch_size, shuffle=False)

    logger_train = utils_visual.Logger(configs.num_epoch, len(train_loader))
    logger_val = utils_visual.Logger(configs.num_epoch, len(val_loader))

    encoder = DSNet.Encoder().to(configs.device)
    Tde = DSNet.Decoder().to(configs.device)
    Ten = DSNet.Encoder().to(configs.device)
    Tde.load_state_dict(torch.load(configs.saved_model_Tde_path))
    Ten.load_state_dict(torch.load(configs.saved_model_Ten_path))
    Tde.eval()
    Ten.eval()
    print('The models are initialized successfully!')

    encoder.train()

    opt_encoder = optim.Adam(encoder.parameters(), lr=configs.lr, betas=(0.5, 0.999))

    lr_scheduler_encoder = torch.optim.lr_scheduler.LambdaLR(opt_encoder, lr_lambda=utils_visual.LambdaLR(configs.num_epoch, configs.start_epoch, configs.decay_epoch).step)

    loss_l1 = nn.L1Loss().to(configs.device)
    loss_per = PerceptualLoss().to(configs.device)
    loss_cl1 = ContrastLoss().to(configs.device)
    loss_cl2 = ContrastLoss_pixel().to(configs.device)
    
    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print("Total_params: ==> {}".format(total_params))

    max_ssim = 0
    max_psnr = 0

    for epoch in range(configs.start_epoch, configs.num_epoch + 1):

        ssims = []
        psnrs = []

        if epoch == configs.start_epoch:
            weights = curriculum_weight(0)
            print(f' n1_weight:{weights[0]}| n2_weight:{weights[1]}| n3_weight:{weights[2]}| n4_weight:{weights[3]}| n5_weight:{weights[4]}|inp_weight:{weights[5]}')
        else:
            weights = curriculum_weight(max_psnr)
            print(f' max_psnr:{max_psnr}| n1_weight:{weights[0]}| n2_weight:{weights[1]}| n3_weight:{weights[2]}| n4_weight:{weights[3]}| n5_weight:{weights[4]}|inp_weight:{weights[5]}')
        
        for i, batch in enumerate(train_loader):

            x = batch[0].to(configs.device)     # haze image
            clear = batch[1].to(configs.device) # clear image
            image_name= batch[2][0]             # haze image name
            # print(image_name)
            candidate_list = batch[3]           # unaligned
            candidate_list_a = batch[4]         # aligned

            feat, x1, x2 = encoder(x)
            feat_gt, x1_gt, x2_gt = Ten(clear)

            output = Tde(feat, x1, x2)
            
            loss_SelfPer = calc_self_perceptual_loss([x1, x2, feat], [x1_gt, x2_gt, feat_gt]) * configs.selfperceptual_weight
            loss_L1 = loss_l1(output, clear) * configs.l1_weight
            loss_Per = loss_per(output, clear) * configs.perceptual_weight
            loss_CL1 = loss_cl1(output, clear, x, candidate_list, weights) * configs.cl1_weight
            # loss_CL2 = loss_cl2(output, clear, x, candidate_list_a, weights) * configs.cl2_weight

            loss = loss_SelfPer + loss_L1 + loss_Per + loss_CL1

            opt_encoder.zero_grad()
            loss.backward()
            opt_encoder.step()

            epoch_losses = logger_train.output_log(
                losses={
                    'loss_L1': loss_L1,
                    'loss_SelfPer': loss_SelfPer,
                    'loss_CL1': loss_CL1,
                    # 'loss_CL2': loss_CL2,
                    'loss': loss
                },
                images={
                    'Haze': x,
                    'Clear': clear,
                    'Output': output
                }
            )

        lr_scheduler_encoder.step()

        if configs.unsupervised: continue

        ########################################## Val ##########################################
        with torch.no_grad():
            encoder.eval()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            images_val = []
            images_name = []

            print(f"epoch:{epoch} ---> Metrics are being evaluated!")

            for a, batch_val in enumerate(val_loader):

                haze_val  = batch_val[0].to(configs.device) # haze image
                clear_val = batch_val[1].to(configs.device) # clear image
                image_name= batch_val[3][0]                 # haze image name

                feat, x1, x2 = encoder(haze_val)
                output_val = Tde(feat, x1, x2)

                images_val.append(output_val)
                images_name.append(image_name)

                psnr1 = calc_psnr(output_val, clear_val)
                ssim1 = calc_ssim(output_val, clear_val).item()

                psnrs.append(psnr1)
                ssims.append(ssim1)

                epoch_losses_val = logger_val.output_log(
                    losses={
                        'Val_PSNR': torch.tensor(psnr1),
                        'Val_SSIM': torch.tensor(ssim1),
                        'Max_PSNR': torch.tensor(max_psnr),
                        'Max_SSIM': torch.tensor(max_ssim)
                    },
                    images={
                        'val_output': output_val,
                        'val_gt': clear_val
                    }
                )

            psnr_eval = np.mean(psnrs)
            ssim_eval = np.mean(ssims)

            if psnr_eval > max_psnr:
                max_psnr = max(max_psnr, psnr_eval)
                os.makedirs(configs.save_model_dir, exist_ok=True)
                torch.save(encoder.state_dict(), configs.save_model_dir + "Sen_Best_PSNR.pth")
                save_image_psnr_dir = configs.save_image_dir + 'val_psnr/'
                os.makedirs(save_image_psnr_dir, exist_ok=True)
                for i in range(len(images_name)):
                    save_image(images_val[i], save_image_psnr_dir + images_name[i])

            if ssim_eval > max_ssim:
                max_ssim = max(max_ssim, ssim_eval)
                os.makedirs(configs.save_model_dir, exist_ok=True)
                torch.save(encoder.state_dict(), configs.save_model_dir + "Sen_Best_SSIM.pth")
                save_image_ssim_dir = configs.save_image_dir + 'val_ssim/'
                os.makedirs(save_image_ssim_dir, exist_ok=True)
                for i in range(len(images_name)):
                    save_image(images_val[i], save_image_ssim_dir + images_name[i])
        
        # if epoch in [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,55,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]:
        #     os.makedirs(configs.save_model_dir, exist_ok=True)
        #     torch.save(encoder.state_dict(), configs.save_model_dir + f"Sen_{epoch}.pth")

    assert epoch_losses!=None, 'epoch_losses is None.'
    os.makedirs(configs.save_log_dir, exist_ok=True)
    with open(configs.save_log_dir + "train_losses_H2C.json", "w") as file:
        json.dump(epoch_losses, file)
    
    if not configs.unsupervised:
        assert epoch_losses_val!=None, 'epoch_losses_val is None.'
        with open(configs.save_log_dir + "val_losses_H2C.json", "w") as file:
            json.dump(epoch_losses_val, file)
    
    os.makedirs(configs.save_model_dir, exist_ok=True)
    torch.save(encoder.state_dict(), configs.save_model_dir + "Sen.pth")
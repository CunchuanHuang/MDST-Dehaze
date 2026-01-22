import time
import datetime
import sys
import os
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms as tf
# from  visdom import Visdom

from metrics.metrics import calculate_js_divergence


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0)

        self.n_epochs=n_epochs
        self.offset=offset
        self.decay_start_epoch=decay_start_epoch

    def step(self, epoch):
        return 1.0-max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


class Logger():
    def __init__(self, n_epochs, batches_epoch, vis=False):
        self.vis = vis
        if self.vis:
            from visdom import Visdom
            self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.epoch_losses = {}
        self.loss_windows = {}
        self.image_windows = {}

    def output_log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write(f'\rEpoch {self.epoch:03d}/{self.n_epochs:03d} [{self.batch:04d}/{self.batches_epoch:04d}] -- ')

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            if (i+1) == len(losses.keys()):
                sys.stdout.write(f'{loss_name}: {(self.losses[loss_name]/self.batch):.4f} -- ')
            else:
                sys.stdout.write(f'{loss_name}: {(self.losses[loss_name]/self.batch):.4f} | ')
        
        batches_done = self.batches_epoch * (self.epoch - 1) + self.batch
        batches_left = self.batches_epoch * (self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        sys.stdout.write(f'ETA: {datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)}')

        if self.vis:
            self._visdom_images(images)

        if (self.batch % self.batches_epoch) == 0:
            if self.vis:
                self._visdom_losses()
            
            self._get_epoch_losses()
            if (self.epoch % self.n_epochs) == 0:
                return self.epoch_losses

            for loss_name, loss in self.losses.items():
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1 
        
    def _visdom_images(self, images:dict):
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor, opts={'title':image_name})
            else:
                self.viz.image(tensor, win=self.image_windows[image_name], opts={'title':image_name})
    
    def _visdom_losses(self):
        for loss_name, loss in self.losses.items():
            if loss_name not in self.loss_windows:
                self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]),
                                                            Y=np.array([loss/self.batch]),
                                                            opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
            else:
                self.viz.line(X=np.array([self.epoch]),
                            Y=np.array([loss/self.batch]),
                            win=self.loss_windows[loss_name],
                            update='append')

    def _get_epoch_losses(self):
        """
        epoch_losses = {
            loss_name: [loss_epoch1, loss_epoch2, ...],
            loss_name: [...]
        }"""
        for loss_name, loss in self.losses.items():
            if loss_name not in self.epoch_losses:
                self.epoch_losses[loss_name] = [loss/self.batch]
            else:
                self.epoch_losses[loss_name].append(loss/self.batch)


def initialize_reliable_style_bank(save_dir, image_name):
    os.makedirs(save_dir, exist_ok=True)
    init_img = torch.zeros(1,3,256,256)
    for i in range(len(image_name)):
        save_path = os.path.join(save_dir, image_name[i])
        torchvision.utils.save_image(init_img, save_path)


def update_reliable_style_bank(stylized, style, save_dir, positive_list):
    old_images = []
    T = tf.ToTensor()

    for i in range(len(positive_list)):
        image_path = os.path.join(save_dir, positive_list[i])
        old_image = Image.open(image_path).convert("RGB")
        old_image = T(old_image)                        # [C,H,W]
        old_image = torch.unsqueeze(old_image, dim=0)   # [1,C,H,W]
        old_images.append(old_image)
    in_bank = torch.cat(old_images, dim=0)              # [B,C,H,W]

    score_stylized = calculate_js_divergence(stylized.detach(), style)
    score_bank = calculate_js_divergence(in_bank.detach(), style)
    # print('score_bank:', score_bank)
    # print('score_stylized', score_stylized)

    for i in range(len(positive_list)):
        if score_stylized < score_bank:
            # update_reliable_style_bank
            save_path = os.path.join(save_dir, positive_list[i])
            torchvision.utils.save_image(stylized[i], save_path)
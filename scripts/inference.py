import sys
sys.path.append('..')
import warnings
import torch.backends.cudnn as cudnn
warnings.filterwarnings('ignore')
cudnn.benchmark = True

import time
import torch
import numpy as np
import os
from PIL import Image
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

from configs import configs_inferenceH2C as configs
from models import DSNet
from typing import Tuple

def resize_image(image: Image.Image, resize: Tuple[int, int], save_path: str) -> Tuple[Image.Image, Tuple[int, int]]:
    original_size = image.size  # (width, height)
    resized_image = image.resize(resize, Image.Resampling.LANCZOS)
    resized_image.save(save_path)
    return resized_image, original_size

def resize_images_in_directory(input_dir: str, output_dir: str, resize: Tuple[int, int]):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        try:
            with Image.open(file_path) as img:
                save_path = os.path.join(output_dir, filename)
                resize_image(img, resize, save_path)
                print(f"Processed and saved {filename} to {output_dir}")
        except IOError:
            print(f"Failed to process {filename}")


def load_image(image_path, resize:int=None, return_tensor:bool=True):
    image = Image.open(image_path).convert("RGB")
    if resize is not None: image = image.resize((resize, resize), Image.LANCZOS)
    image = np.array(image) / 255.0     # [0,255] -> [0,1]
    if not return_tensor: return image
    image_tensor = torch.from_numpy(image).float()  # ndarray -> tensor
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(configs.device)    # [H,W,C] -> [C,H,W] -> [1,C,H,W]
    return image_tensor

def load_model():
    encoder = DSNet.Encoder().to(configs.device)
    decoder = DSNet.Decoder().to(configs.device)
    encoder.load_state_dict(torch.load(configs.saved_model_Sen_path, map_location=configs.device))
    decoder.load_state_dict(torch.load(configs.saved_model_Tde_path, map_location=configs.device))
    encoder.eval()
    decoder.eval()
    return encoder, decoder

def dehaze_image(hazy_path):
    """single image dehazing"""
    image = load_image(hazy_path)
    encoder, decoder = load_model()

    start_time = time.time()

    with torch.no_grad():
        feat, x1, x2 = encoder(image)
        output = decoder(feat, x1, x2)
    
    total_time = time.time() - start_time
    os.makedirs(configs.save_image_dir, exist_ok=True)
    save_image(output, os.path.join(configs.save_image_dir, os.path.basename(hazy_path)))
    return total_time

def dehaze_images(hazy_dir):
    """batch image dehazing"""
    image_names = os.listdir(hazy_dir)
    image_paths = [os.path.join(hazy_dir, f) for f in image_names]
    print(f"Total number: {len(image_paths)}, Processing...")

    all_time = []
    for i, image_path in enumerate(image_paths, start=1):
        per_time = dehaze_image(image_path)
        all_time.append(per_time)
        print(f"[{i:04d}/{len(image_paths):04d}] done!: {os.path.basename(image_path)}, Time Cost: {per_time}s")
    
    avg_time = sum(all_time) / len(all_time)    # s/per
    fps = 1.0 / avg_time                        # per/s
    latency = avg_time * 1000                   # ms/per
    print(f"Finish! FPS: {fps}, Latency: {latency}ms")


def evaluate(dehaze_dir, clear_dir):
    image_names = os.listdir(dehaze_dir)

    psnr_list = []
    ssim_list = []

    for image_name in image_names:
        dehaze_path = os.path.join(dehaze_dir, image_name)
        clear_path = os.path.join(clear_dir, image_name)
        dehaze_image = load_image(dehaze_path, return_tensor=False)
        clear_image = load_image(clear_path, return_tensor=False)

        psnr_val = psnr(clear_image, dehaze_image, data_range=1.0)
        ssim_val = ssim(clear_image, dehaze_image, channel_axis=-1, data_range=1.0)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        print(f"[{image_name}] PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}")

    print("===============================")
    print(f"PSNR: {np.mean(psnr_list):.4f}")
    print(f"SSIM: {np.mean(ssim_list):.4f}")
    print("Finish!")
    return psnr_list, ssim_list


if __name__ == "__main__":
    dehaze_images('../datasets/IO-haze/test/haze1024/')
    # resize_images_in_directory('../datasets/IO-haze-orignsize/test/clear/', '../datasets/IO-haze-orignsize/test/clear1024/', (1024,1024))
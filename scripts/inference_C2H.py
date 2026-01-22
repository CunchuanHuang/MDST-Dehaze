import sys
sys.path.append('..')
import time
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os

from configs import configs_inferenceC2H as configs
from myutils.dataloader import Dataloader_TestC2H
from models.MDSTNet import MDSTNet

import warnings
import torch.backends.cudnn as cudnn
from PIL import Image
from PIL import ImageFile
cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')


if __name__ == "__main__":

    test_dataset = Dataloader_TestC2H(haze_dir=configs.target_dir, clear_dir=configs.source_dir, transform=configs.transforms_test, unaligned=True, model='test')
    dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=configs.n_threads)

    Gen = MDSTNet().to(configs.device)
    Gen.load_state_dict(torch.load(configs.saved_model_path, map_location=configs.device))
    Gen.eval()
    print('The models are initialized successfully!')

    start_time = time.time()
    
    for i, batch in enumerate(dataloader):
        haze = batch[0].to(configs.device)  # haze image
        clear = batch[1].to(configs.device) # clear image
        image_name=batch[2][0]              # clear image name
        # image_name=batch[3][0]            # haze image name

        with torch.no_grad():
            output, _, _, _, _, _, _, _ = Gen(clear, haze)

            os.makedirs(configs.save_image_dir, exist_ok=True)
            save_image(output, configs.save_image_dir + image_name)
            print(f"[{i+1:04d}/{len(dataloader):04d}] {image_name}")

    total_time = time.time() - start_time
    avg_time = total_time / len(dataloader) # s/per
    fps = 1.0 / avg_time                    # per/s
    latency = avg_time * 1000               # ms/per
    print(f"Finish! FPS: {fps}, Latency: {latency}ms")
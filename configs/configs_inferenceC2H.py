import torch
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
# in_channels = 3
# load_size = 256
n_threads = 0

transforms_test = [
    transforms.ToTensor()
]


source = 'example'  # source domain (clear)
target = 'example'  # target domain (haze)
source_dir = f'../datasets/{source}/train/clear/'
target_dir = f'../datasets/{target}/train/haze/'
saved_model_path = f'../results/checkpoints/{target}/C2H.pth'
save_image_dir = f'../results/gehaze/source-{source}/target-{target}/'
import torch
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
n_threads = 0

transforms_test = [
    transforms.ToTensor()
]

category = 'IO-haze'
haze_dir = '../datasets/IO-haze/test/haze/'
saved_model_Sen_path = '../results/checkpoints/IO-haze/Sen.pth'
saved_model_Tde_path = '../results/checkpoints/IO-haze/Tde.pth'
save_image_dir = f'../results/dehaze/{category}/all/'

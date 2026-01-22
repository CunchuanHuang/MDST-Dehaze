import torch
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
# in_channels = 3
# load_size = 256
n_threads = 0

transforms_test = [
    transforms.ToTensor()
]


category = 'example'                                            # dataset name
haze_dir = '../datasets/example/test/haze/'                     # haze dataset
saved_model_Sen_path = '../results/checkpoints/example/Sen.pth' # Sen model
saved_model_Tde_path = '../results/checkpoints/example/Tde.pth' # Tde model
save_image_dir = f'../results/dehaze/{category}/inference/'
import torch
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
# in_channels = 3
# load_size = 256
batch_size = 1
num_epoch = 100
lr = 1e-4
decay_epoch = 50
start_epoch = 1
n_threads = 0

l1_weight = 1.0
perceptual_weight = 0.1

transforms_train = [
    transforms.ToTensor()
]


category = 'example'                                        # dataset name
clear_dir = f'../datasets/{category}/train/clear/'
save_image_dir = f'../results/recimg/{category}/'
save_model_dir = f'../results/checkpoints/{category}/'
save_log_dir = f'../results/logs/{category}/'
save_log_path = '../results/logs/logs.txt'
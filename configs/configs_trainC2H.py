import torch
from torchvision import transforms


device = "cuda" if torch.cuda.is_available() else "cpu"
# in_channels = 3
# load_size = 256
batch_size = 8
num_epoch = 200
lr = 1e-4
decay_epoch = 100
start_epoch = 1
n_threads = 0
save_list = [150, 160, 170, 180, 190, 200]

style_weight = 1.0
content_weight = 1.0
identity_p_weight = 50.0
identity_f_weight = 1.0
identity_pha_weight = 10.0
gan_p_weight = 5.0
gan_f_weight = 0.5

transforms_train = [
    transforms.ToTensor()
]

category = 'example'                                            # dataset name
content_dir = '../datasets/example/train/clear/'                # content (clear) dataset
style_dir = '../datasets/example/train/haze/'                   # style (haze) dataset
save_stylized_dir = f'../results/stylized_database/{category}/'
save_model_dir = f'../results/checkpoints/{category}/'
save_log_dir = f'../results/logs/{category}/'
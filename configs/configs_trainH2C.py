import torch
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
# in_channels = 3
# load_size = 256
batch_size = 1
num_epoch = 200
lr = 1e-4
decay_epoch = 50
start_epoch = 1
n_threads = 0

cl_lambda = 0.25
l1_weight = 10.0
selfperceptual_weight = 1.0
perceptual_weight = 1.0
cl1_weight = 0.1
cl2_weight = 0.1

transforms_train = [
    transforms.ToTensor()
]

unsupervised = False         # True: unsupervised training, False: Semi-supervised training


category = 'example'                                            # dataset name
clear_dir = '../datasets/example/train/clear/'                  # training set: clear dataset
haze_dir = '../results/stylized_database/example/stylized200/'  # training set: stylized dataset
stylized_dir = '../results/stylized_database/example/'          # training set: stylized database
clear_val_dir = '../datasets/example/test/clear/'               # test/val set: clear dataset
haze_val_dir = '../datasets/example/test/haze/'                 # test/val set: haze dataset
saved_model_Ten_path = '../results/checkpoints/example/Ten.pth' # Ten model
saved_model_Tde_path = '../results/checkpoints/example/Tde.pth' # Tde model
save_image_dir = f'../results/dehaze/{category}/'
save_model_dir = f'../results/checkpoints/{category}/'
save_log_dir = f'../results/logs/{category}/'
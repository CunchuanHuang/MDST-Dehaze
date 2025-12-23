import random
import os
from PIL import Image
from torchvision import transforms as tf
from torch.utils.data import Dataset

random.seed(2024)


class Dataloader(Dataset):
    def __init__(self, haze_dir, clear_dir, transform=None, unaligned=True, model="train"):
        self.transform = tf.Compose(transform)
        self.unaligned = unaligned
        self.model = model

        self.haze_names, self.clear_names = os.listdir(haze_dir), os.listdir(clear_dir)
        self.haze_names, self.clear_names = sorted(self.haze_names), sorted(self.clear_names)
        
        assert self.haze_names == self.clear_names, 'Image names are differnet.'

        self.haze_paths, self.clear_paths = [], []
        for name in self.clear_names:
            self.haze_paths.append(os.path.join(haze_dir, name))
            self.clear_paths.append(os.path.join(clear_dir, name))
        
        # print(self.clear_paths)
        # print(self.haze_paths)

        print("Total {} examples:".format(model), max(len(self.haze_paths), len(self.clear_paths)))

    def __getitem__(self, index):
        clear_path = self.clear_paths[index % len(self.clear_paths)]
        if self.unaligned:
            haze_path = self.haze_paths[random.randint(0, len(self.haze_paths) - 1)]
        else:
            haze_path = self.haze_paths[index % len(self.haze_paths)]
        
        haze_name = os.path.basename(haze_path)
        clear_name = os.path.basename(clear_path)
        
        haze = Image.open(haze_path).convert("RGB")
        clear = Image.open(clear_path).convert("RGB")
        haze = self.transform(haze)
        clear = self.transform(clear)

        return haze, clear, clear_name, haze_name

    def __len__(self):
        return max(len(self.haze_paths), len(self.clear_paths))
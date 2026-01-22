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
        
        haze = Image.open(haze_path).convert("RGB")     # [W,H]
        clear = Image.open(clear_path).convert("RGB")   # [W,H]
        haze = self.transform(haze)                     # [C,H,W]
        clear = self.transform(clear)                   # [C,H,W]

        return haze, clear, clear_name, haze_name

    def __len__(self):
        return max(len(self.haze_paths), len(self.clear_paths))


class Dataloader_TestC2H(Dataset):
    def __init__(self, haze_dir, clear_dir, transform=None, unaligned=True, model="test"):
        self.transform = tf.Compose(transform)
        self.unaligned = unaligned
        self.model = model
        self.haze_paths = [os.path.join(haze_dir, name) for name in sorted(os.listdir(haze_dir))]
        self.clear_paths = [os.path.join(clear_dir, name) for name in sorted(os.listdir(clear_dir))]
        print(f"Total {model} examples:")
        print(f"  - Haze images: {len(self.haze_paths)}")
        print(f"  - Clear images: {len(self.clear_paths)}")

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
        if self.transform is not None:
            haze = self.transform(haze)
            clear = self.transform(clear)
        return haze, clear, clear_name, haze_name

    def __len__(self):
        return len(self.clear_paths)


class Dataloader_TrainH2C(Dataset):
    def __init__(self, haze_dir, clear_dir, stylized_database_dir, transform=None, model="train"):
        self.transform = tf.Compose(transform)
        self.model = model

        self.haze_names, self.clear_names = os.listdir(haze_dir), os.listdir(clear_dir)
        self.haze_names, self.clear_names = sorted(self.haze_names), sorted(self.clear_names)

        assert self.haze_names == self.clear_names, 'Image names are differnet.'

        self.haze_paths, self.clear_paths = [], []
        for name in self.clear_names:
            self.haze_paths.append(os.path.join(haze_dir, name))
            self.clear_paths.append(os.path.join(clear_dir, name))
        
        self.random_stylized_database = Randomized_Style_Version(stylized_database_dir)

        # print(self.clear_paths)
        # print(self.haze_paths)
        # print(self.random_stylized_database)

        print("Total {} examples:".format(model), max(len(self.haze_paths), len(self.clear_paths)))

    def __getitem__(self, index):
        haze_path = self.haze_paths[index % len(self.haze_paths)]
        clear_path = self.clear_paths[index % len(self.clear_paths)]

        name = os.path.basename(haze_path)

        Candidate1 = self.random_stylized_database[0][random.randint(0, len(self.random_stylized_database[0]) - 1)]
        Candidate2 = self.random_stylized_database[1][random.randint(0, len(self.random_stylized_database[1]) - 1)]
        Candidate3 = self.random_stylized_database[2][random.randint(0, len(self.random_stylized_database[2]) - 1)]
        Candidate4 = self.random_stylized_database[3][random.randint(0, len(self.random_stylized_database[3]) - 1)]
        Candidate5 = self.random_stylized_database[4][random.randint(0, len(self.random_stylized_database[4]) - 1)]
        Candidate1_a = self.random_stylized_database[0][index % len(self.random_stylized_database[0])]
        Candidate2_a = self.random_stylized_database[1][index % len(self.random_stylized_database[1])]
        Candidate3_a = self.random_stylized_database[2][index % len(self.random_stylized_database[2])]
        Candidate4_a = self.random_stylized_database[3][index % len(self.random_stylized_database[3])]
        Candidate5_a = self.random_stylized_database[4][index % len(self.random_stylized_database[4])]

        # print(clear_path)
        # print(haze_path)
        # print(Candidate1, Candidate2, Candidate3, Candidate4, Candidate5)
        # print(Candidate1_a, Candidate2_a, Candidate3_a, Candidate4_a, Candidate5_a)

        haze = self.transform(Image.open(haze_path).convert("RGB"))
        clear = self.transform(Image.open(clear_path).convert("RGB"))

        Candidate1 = self.transform(Image.open(Candidate1).convert("RGB"))
        Candidate2 = self.transform(Image.open(Candidate2).convert("RGB"))
        Candidate3 = self.transform(Image.open(Candidate3).convert("RGB"))
        Candidate4 = self.transform(Image.open(Candidate4).convert("RGB"))
        Candidate5 = self.transform(Image.open(Candidate5).convert("RGB"))
        
        Candidate1_a = self.transform(Image.open(Candidate1_a).convert("RGB"))
        Candidate2_a = self.transform(Image.open(Candidate2_a).convert("RGB"))
        Candidate3_a = self.transform(Image.open(Candidate3_a).convert("RGB"))
        Candidate4_a = self.transform(Image.open(Candidate4_a).convert("RGB"))
        Candidate5_a = self.transform(Image.open(Candidate5_a).convert("RGB"))

        Candidate_list = [Candidate1, Candidate2, Candidate3, Candidate4, Candidate5]
        Candidate_a_list = [Candidate1_a, Candidate2_a, Candidate3_a, Candidate4_a, Candidate5_a]

        return haze, clear, name, Candidate_list, Candidate_a_list

    def __len__(self):
        return max(len(self.haze_paths), len(self.clear_paths))





def Randomized_Style_Version(stylized_database_dir):
    random_list = random.sample(range(5), 5)
    random_stylized_database = []
    version = ["stylized150/","stylized160/","stylized170/","stylized180/","stylized190/"]
    for i in range(len(random_list)):
        random_index =  random_list[i]

        stylized_dir = stylized_database_dir + version[random_index]
        stylized_names = sorted(os.listdir(stylized_dir))
        stylized_paths = []
        for name in stylized_names:
            stylized_paths.append(os.path.join(stylized_dir, name))

        random_stylized_database.append(stylized_paths)

    return random_stylized_database


if  __name__ == "__main__":
    haze_path= "../datasets/example/train/haze/"
    clear_path= "../datasets/example/train/clear/"
    s = '../results/stylized_database/stylized_example/'
    transform_ = [
        tf.ToTensor()
    ]

    train_sets=Dataloader(haze_path, clear_path, transform_)
    train_sets[2]
    # dataload = DataLoader(train_sets, batch_size=1, shuffle=True, num_workers=4)

    # for i, batch in enumerate(dataload):
    #     print((batch[8].shape))
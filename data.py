import os
import numpy as np, torch
import imageio as io
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

class VideoTrainDataset(Dataset):
    def __init__(self, opt, transform=None):
        self.fps = opt.fps
        self.win_size = opt.lr_window
        self.transform = transform
        self.shuffle = True

        lr_dir = os.path.join(opt.dataset, 'lr_frames', opt.scene)
        hr_dir = os.path.join(opt.dataset, 'hr_frames', opt.scene)
        num_lr = len(os.listdir(lr_dir)) - 1
        num_hr = num_lr * 120 // opt.fps
        
        self.lr_idx = []
        self.hr_idx = []
        for i in range(1 + opt.fps, num_lr - opt.fps, 120 // opt.fps):
            self.lr_idx.append([os.path.join(lr_dir, f'frame_{x:05d}.png')
                                for x in range(i - opt.lr_window, i + opt.lr_window + 1)])
            self.hr_idx.append(os.path.join(hr_dir, f'frame_{i:05d}.png'))

    def __len__(self):
        return len(self.hr_idx)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        lr_imgs_np = np.concatenate([io.imread(f) for f in self.lr_idx[idx]],
                                    axis=-1).transpose(-1, 0, 1)
        lr_images = torch.FloatTensor(lr_imgs_np)  / 255.
        hr_image = torch.FloatTensor(io.imread(self.hr_idx[idx]).transpose(-1, 0, 1)) / 255.

        sample = {'hr': hr_image, 'lr': lr_images}

        if self.transform:
            sample = self.transform(sample)

        return sample


class VideoAllDataset(Dataset):
    def __init__(self, opt, transform=None):
        self.fps = opt.fps
        self.win_size = opt.lr_window
        self.transform = transform
        self.shuffle = False

        lr_dir = os.path.join(opt.dataset, 'lr_frames', opt.scene)
        hr_dir = os.path.join(opt.dataset, 'hr_frames', opt.scene)
        num_lr = len(os.listdir(lr_dir)) - 1
        num_hr = num_lr * 120 // opt.fps
        
        self.lr_idx = []
        self.hr_idx = []
        for i in range(1 + opt.fps, num_lr - opt.fps):
            self.lr_idx.append([os.path.join(lr_dir, f'frame_{x:05d}.png')
                                for x in range(i - opt.lr_window, i + opt.lr_window + 1)])
            self.hr_idx.append(os.path.join(hr_dir, f'frame_{i:05d}.png'))

    def __len__(self):
        return len(self.hr_idx)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        lr_imgs_np = np.concatenate([io.imread(f) for f in self.lr_idx[idx]],
                                    axis=-1).transpose(-1, 0, 1)
        lr_images = torch.FloatTensor(lr_imgs_np)  / 255.
        hr_image = torch.FloatTensor(io.imread(self.hr_idx[idx]).transpose(-1, 0, 1)) / 255.

        sample = {'hr': hr_image, 'lr': lr_images}

        if self.transform:
            sample = self.transform(sample)

        return sample

def get_loader(dataset, batch=4):
    return DataLoader(dataset, batch_size=batch, shuffle=dataset.shuffle, num_workers=4)

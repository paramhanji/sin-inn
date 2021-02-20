import os
import numpy as np, torch
import imageio as io
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

'''
Base class for single video dataset
Args(contained within opt):
    lr_window: # of LR frames on either side
    fps: fps of HR frames
'''
class VideoDataset(Dataset):
    def __init__(self, opt, transform=None):
        self.fps = opt.fps
        self.win_size = opt.lr_window
        self.transform = transform

        lr_dir = os.path.join(opt.dataset, 'lr_frames', opt.scene)
        hr_dir = os.path.join(opt.dataset, 'hr_frames', opt.scene)
        num_lr = len(os.listdir(lr_dir)) - 1
        
        self.lr_files = []
        self.hr_files = []
        self.populate_files(lr_dir, hr_dir, num_lr, opt)

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        lr_imgs_np = np.concatenate([io.imread(f) for f in self.lr_files[idx]],
                                    axis=-1).transpose(-1, 0, 1)
        lr_images = torch.FloatTensor(lr_imgs_np)  / 255.
        hr_image = torch.FloatTensor(io.imread(self.hr_files[idx]).transpose(-1, 0, 1)) / 255.

        sample = {'hr': hr_image, 'lr': lr_images}

        if self.transform:
            sample = self.transform(sample)

        return sample

'''
Training dataset consisting of sparse HR frames with corresponding LR windows
'''
class VideoTrainDataset(VideoDataset):
    def __init__(self, opt, transform=None):
        super(VideoTrainDataset, self).__init__(opt, transform)
        self.shuffle = True

    def populate_files(self, lr_dir, hr_dir, num_lr, opt):
        for i in range(1 + opt.fps, num_lr - opt.fps, 120 // opt.fps):
            self.lr_files.append([os.path.join(lr_dir, f'frame_{x:05d}.png')
                                for x in range(i - opt.lr_window, i + opt.lr_window + 1)])
            self.hr_files.append(os.path.join(hr_dir, f'frame_{i:05d}.png'))

'''
All LR windows to be used during HR generation (post-training).
'''
class VideoAllDataset(VideoDataset):
    def __init__(self, opt, transform=None):
        super(VideoAllDataset, self).__init__(opt, transform)
        self.shuffle = False

    def populate_files(self, lr_dir, hr_dir, num_lr, opt):
        for i in range(1 + opt.fps, num_lr - opt.fps):
            self.lr_files.append([os.path.join(lr_dir, f'frame_{x:05d}.png')
                                for x in range(i - opt.lr_window, i + opt.lr_window + 1)])
            self.hr_files.append(os.path.join(hr_dir, f'frame_{i:05d}.png'))

'''
Uniformly sample "k" pairs for validation
'''
class VideoValDataset(VideoDataset):
    def __init__(self, opt, k, transform=None):
        self.k = k
        super(VideoValDataset, self).__init__(opt, transform)
        self.shuffle = False

    def populate_files(self, lr_dir, hr_dir, num_lr, opt):
        num = 0
        for i in torch.randperm(num_lr - 2*opt.lr_window):
            i += opt.lr_window
            # Skip images from train set
            if (i + opt.fps) % (120 // opt.fps) == 0:
                continue
            self.lr_files.append([os.path.join(lr_dir, f'frame_{x:05d}.png')
                                for x in range(i - opt.lr_window, i + opt.lr_window + 1)])
            self.hr_files.append(os.path.join(hr_dir, f'frame_{i:05d}.png'))
            if num == self.k:
                break

'''
Wrap supervised and unsupervised datasets into single class
'''
class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def get_loader(dataset, batch=4):
    return DataLoader(dataset, batch_size=batch, shuffle=dataset.shuffle, num_workers=4)

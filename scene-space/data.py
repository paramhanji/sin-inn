import torch

from poses.pose_utils import load_data

class ImagesData(torch.utils.data.Dataset):
    def __init__(self, dir: str, length=10):
        super().__init__()
        self.dir = dir
        pose, _, _, _ = load_data(self.dir, index=0)
        self.K = torch.eye(4)
        self.K[0,0] = pose[2,4]
        self.K[1,1] = pose[2,4]
        self.K[0,2] = pose[0,5]
        self.K[1,2] = pose[1,5]
        self.K_inv = torch.inverse(self.K)
        self.len = length

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        pose, bds, img, depth = load_data(self.dir, index=index)
        cam2world = torch.zeros(4, 4)
        cam2world[:3,:] = torch.tensor(pose[...,:4])
        cam2world[3,3] = 1
        return cam2world, bds, img, depth

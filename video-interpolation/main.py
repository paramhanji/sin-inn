import trainer as T
import data as D
import model as M

from glob import glob
import os.path as path
import ipdb, argparse

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def net(in_channels, activation, out_channels=3):
    return M.MLP(
        in_channels,
        out_channels=out_channels,
        hidden_dim=512,
        hidden_layers=4,
        # hidden_dim=256,
        # hidden_layers=3,
        activation=activation)

siren = net(3, 'siren')
siren_flow = net(3, 'siren', out_channels=2)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['train', 'test'])
    parser.add_argument('--model-path')
    return parser.parse_args()

def train_model():
    # video_clip = D.FlowImagesModule('/anfs/gfxdisp/video/adobe240f/hr_frames/GOPR9634_binning_4x/', 10, 50, 10)
    video_clip = D.VideoModule('../datasets/dancing.mp4', 0, 450, step=10, batch=6)
    logger = TensorBoardLogger('logs', name="dancing", version=f'siren/{video_clip.dataset.step}step_unsup')
    model = T.FlowTrainer(loss=F.l1_loss, net=siren_flow, lr=1e-4)
    latest_ckpt = max(glob(f'{logger.save_dir}/{logger.name}/{logger.version}/checkpoints/*.ckpt'), key=path.getmtime, default=None)
    trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=50000,
        # auto_lr_find=True,
        callbacks=[ModelCheckpoint()],
        resume_from_checkpoint=latest_ckpt,
        check_val_every_n_epoch=500,
        num_sanity_val_steps=0)
    with ipdb.launch_ipdb_on_exception():
        if latest_ckpt is None:
            trainer.tune(model, video_clip.train_dataloader())
        trainer.fit(model, video_clip)

def test_model():
    video_clip = D.FlowImagesModule('/anfs/gfxdisp/video/adobe240f/hr_frames/IMG_0028_binning_4x', 20, 49, 5, batch=1)
    logger = TensorBoardLogger('logs', name="stair", version='siren/5stepflow')
    model = T.FlowTrainer(loss=F.l1_loss, net=siren_flow, lr=1e-4)
    latest_ckpt = max(glob(f'{logger.save_dir}/{logger.name}/{logger.version}/checkpoints/*.ckpt'), key=path.getmtime, default=None)
    trainer = pl.Trainer(gpus=1, logger=logger)
    trainer.test(model, dataloaders=video_clip.train_dataloader(), ckpt_path=latest_ckpt)



if __name__ == "__main__":
    args = get_args()
    if args.operation == 'train':
        train_model()
    elif args.operation == 'test':
        test_model()

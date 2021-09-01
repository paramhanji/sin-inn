import trainer as T
import data as D
import model as M

from glob import glob
import os.path as path
import ipdb, argparse
from math import ceil

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def net(in_channels, activation, out_channels=3):
    return M.MLP(
        in_channels,
        out_channels=out_channels,
        hidden_dim=256,
        # hidden_dim=512,
        hidden_layers=3,
        # hidden_layers=4,
        activation=activation)

siren = net(3, 'siren', out_channels=2)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['train', 'test'])
    parser.add_argument('--input-video', default='../datasets/dancing.mp4')
    parser.add_argument('--end', default=450, type=int)
    parser.add_argument('--step', default=10, type=int)
    parser.add_argument('--batch', default=16, type=int)
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--log-iter', default=1000, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    return parser.parse_args()

def train_model(video, logger, ckpt, args):
    model = T.FlowTrainer(loss=F.l1_loss, net=siren, lr=args.lr,
                          step=args.step, flow_scale=video.dataset.flow.max())
    trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=args.epochs,
                         callbacks=[ModelCheckpoint()], resume_from_checkpoint=ckpt,
                         check_val_every_n_epoch=args.log_iter,
                         num_sanity_val_steps=ceil(len(video.dataset)/args.batch))
    with ipdb.launch_ipdb_on_exception():
        trainer.fit(model, video)

def test_model(video, ckpt, step):
    model = T.FlowTrainer.load_from_checkpoint(ckpt, net=siren, step=step,
                                               flow_scale=video.dataset.flow.max(),
                                               log_gt=False)
    trainer = pl.Trainer(gpus=1, logger=None)
    trainer.test(model, dataloaders=video.test_dataloader())


if __name__ == "__main__":
    args = get_args()
    video_clip = D.VideoModule(args.input_video, 0, args.end, step=args.step, batch=args.batch)
    scene, _ = path.splitext(path.basename(args.input_video))
    logger = TensorBoardLogger('logs', name=scene, version=f'{args.step}_step_unsup')
    latest_ckpt = max(glob(f'{logger.save_dir}/{logger.name}/{logger.version}/checkpoints/*.ckpt'),
                      key=path.getmtime, default=None)
    if args.operation == 'train':
        train_model(video_clip, logger, latest_ckpt, args)
    elif args.operation == 'test':
        test_model(video_clip, latest_ckpt, args.step)

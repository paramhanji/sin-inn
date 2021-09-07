import trainer as T
import data as D

from glob import glob
import os.path as path
import ipdb, argparse
from math import ceil

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['train', 'plot'])
    # Data options
    parser.add_argument('--input-video', default='../datasets/sintel/training/final/alley_1')
    parser.add_argument('--name', required=True)
    parser.add_argument('--end', default=450, type=int)
    parser.add_argument('--step', default=1, type=int)
    parser.add_argument('--size', default=200, type=int)
    parser.add_argument('--batch', default=8, type=int)
    # Train options
    parser.add_argument('--epochs', default=5000, type=int)
    parser.add_argument('--log-iter', default=200, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--logger', default=None, choices=['wandb', None])
    parser.add_argument('--loss-photo', default='l1', choices=['l1'])
    parser.add_argument('--loss-smooth1', default=0.1, type=float)
    parser.add_argument('--loss-smooth2', default=0, type=float)
    parser.add_argument('--edge-constant', default=150, type=float)
    parser.add_argument('--edge-func', default='gauss', choices=['exp','gauss'])
    parser.add_argument('--occl', default='brox', choices=['brox', None])
    parser.add_argument('--occl-lambda', default=1, type=float)
    return parser.parse_args()

def train_model(video, logger, ckpt, args):
    dataset = video.dataset
    clbcks = [ModelCheckpoint(every_n_epochs=args.epochs//100, dirpath=path.dirname(ckpt))] \
              if ckpt else []
    if ckpt and not path.isfile(ckpt):
        logger.experiment.log({'source': wandb.Video((dataset.video * 255).type(torch.uint8))})
        ckpt = None

    model = T.FlowTrainer(args, flow_scale=dataset.flow.max().item())
    trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=args.epochs,
                         callbacks=clbcks, auto_lr_find=True,
                         resume_from_checkpoint=ckpt,
                         check_val_every_n_epoch=args.log_iter,
                         num_sanity_val_steps=0)
    with ipdb.launch_ipdb_on_exception():
        if ckpt is None:
            trainer.tune(model, video)
        args.lr = model.lr
        if logger:
            logger.log_hyperparams(args)
        trainer.fit(model, video)

def plot_fit(video, ckpt):
    import matplotlib.pyplot as plt
    figure = plt.gcf()
    figure.set_size_inches(16, 12)
    dataset = video.testset
    model = T.FlowTrainer.load_from_checkpoint(ckpt, flow_scale=dataset.flow.max().item())
    model.eval()
    with torch.no_grad():
        t, _, h, w = dataset.video.shape
        xyt = torch.zeros(10000, 3)
        xyt[:,1] = torch.linspace(-1, 1, 10000)
        f = model.net(xyt) * model.flow_scale.cpu()
        plt.plot(xyt[:,1], f[:,0], label='Varying x', color='g', alpha=0.5)
        f = dataset.flow[t//2,0,:,w//2].cpu()
        f[f == 0] = float('nan')
        plt.scatter(torch.linspace(-1, 1, h), f, color='g', marker='.')
        xyt = torch.zeros(10000, 3)
        xyt[:,0] = torch.linspace(-1, 1, 10000)
        f = model.net(xyt) * model.flow_scale.cpu()
        plt.plot(xyt[:,0], f[:,0], label='Varying t', color='r', alpha=0.5)
        f = dataset.flow[:,0,h//2,w//2].cpu()
        f[f == 0] = float('nan')
        plt.scatter(torch.linspace(-1, 1, t)[:-1], f, color='r', marker='.')
        plt.legend()
        plt.show()
        # plt.savefig('fits.pdf')
    # trainer = pl.Trainer(gpus=1, logger=None)
    # trainer.test(model, dataloaders=video.test_dataloader())


if __name__ == "__main__":
    args = get_args()
    if path.isdir(path.join(args.input_video)):
        print('Loading MPI sintel sequence')
        video_clip = D.ImagesModule(args.input_video, size=args.size, batch=args.batch)
    else:
        print('Extracting frames from video')
        video_clip = D.VideoModule(args.input_video, 0, args.end, step=args.step, batch=args.batch, size=args.size)
    scene, _ = path.splitext(path.basename(args.input_video))
    logger, latest_ckpt = None, None
    if args.logger == 'wandb':
        unique_name = f'{scene}_{args.step}_{args.name}'
        logger = WandbLogger(project='optical_flow', name=unique_name)
        latest_ckpt = max(glob(path.join('checkpoints', unique_name, '*.ckpt')),
                          key=path.getmtime, default=path.join('checkpoints', unique_name,'temp'))

    if args.operation == 'train':
        train_model(video_clip, logger, latest_ckpt, args)
    elif args.operation == 'plot':
        plot_fit(video_clip, latest_ckpt)

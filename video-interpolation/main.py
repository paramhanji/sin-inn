import trainer as T
import data as D

from glob import glob
import os.path as path
import ipdb, argparse
from math import ceil

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['train', 'plot'])
    # Data options
    parser.add_argument('--input-video', default='../datasets/dancing.mp4')
    parser.add_argument('--end', default=450, type=int)
    parser.add_argument('--step', default=10, type=int)
    parser.add_argument('--size', default=200, type=int)
    parser.add_argument('--batch', default=16, type=int)
    # Train options
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--log-iter', default=1000, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    return parser.parse_args()

def train_model(video, logger, ckpt, args):
    dataset = video.testset
    if ckpt is None:
        logger.experiment.add_video('source', dataset.video.unsqueeze(0), 0)

    model = T.FlowTrainer(loss=F.l1_loss, lr=args.lr, step=args.step,
                          flow_scale=dataset.flow.max().item())
    trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=args.epochs,
                         callbacks=[ModelCheckpoint(every_n_epochs=args.epochs//100)],
                         resume_from_checkpoint=ckpt,
                         check_val_every_n_epoch=args.log_iter,
                         num_sanity_val_steps=ceil(len(dataset)/args.batch))
    with ipdb.launch_ipdb_on_exception():
        trainer.fit(model, video)

def plot_fit(video, ckpt, step):
    import matplotlib.pyplot as plt
    figure = plt.gcf()
    figure.set_size_inches(16, 12)
    dataset = video.testset
    model = T.FlowTrainer.load_from_checkpoint(ckpt, step=step,
                                               flow_scale=dataset.flow.max().item())
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
    video_clip = D.VideoModule(args.input_video, 0, args.end, step=args.step, batch=args.batch, size=args.size)
    scene, _ = path.splitext(path.basename(args.input_video))
    logger = TensorBoardLogger('logs', name=scene, version=f'{args.step}_step_unsup')
    latest_ckpt = max(glob(f'{logger.save_dir}/{logger.name}/{logger.version}/checkpoints/*.ckpt'),
                      key=path.getmtime, default=None)
    if args.operation == 'train':
        train_model(video_clip, logger, latest_ckpt, args)
    elif args.operation == 'plot':
        plot_fit(video_clip, latest_ckpt, args.step)

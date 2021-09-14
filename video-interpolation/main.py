from glob import glob
import os, os.path as path
import ipdb, argparse, copy, logging, warnings
from tqdm import trange, tqdm

import torch, numpy as np, wandb, pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from data import get_video
import trainer as T
from my_utils.flow_viz import flow2img

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['metatrain', 'train', 'plot', 'test'])
    # Data options
    parser.add_argument('--input-video', default='../datasets/sintel/training/final/ambush_5')
    parser.add_argument('--name', default='temp')
    parser.add_argument('--end', type=int)
    parser.add_argument('--step', type=int)
    parser.add_argument('--size', default=200, type=int)
    parser.add_argument('--batch', default=8, type=int)
    # Train options
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--meta-epochs', type=int)
    parser.add_argument('--log-iter', default=1000, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--meta-lr', type=float)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--loss-photo', default='l1', choices=['l1', 'census', 'ssim', 'charbonnier'])
    parser.add_argument('--loss-smooth1', default=0.1, type=float)
    parser.add_argument('--loss-smooth2', default=0, type=float)
    parser.add_argument('--edge-constant', default=150, type=float)
    parser.add_argument('--edge-func', default='gauss', choices=['exp', 'gauss'])
    parser.add_argument('--occl', default=None, choices=['brox', 'wang', None])
    parser.add_argument('--occl-delay', default=2500, type=int)
    parser.add_argument('--occl-thresh', default=0.6, type=float)
    return parser.parse_args()


def train_metamodel(args):
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    root = path.dirname(args.input_video)
    video_paths = [path.join(root, f) for f in sorted(os.listdir(root)) if f != args.input_video]

    meta_net = T.default_net
    meta_optim = torch.optim.Adam(meta_net.parameters(), lr=args.meta_lr)

    for epoch in trange(args.meta_epochs):
        for video_path in tqdm(np.random.permutation(video_paths), leave=False):
            video, _ = get_video(video_path, args)
            meta_optim.zero_grad()
            inner_net = copy.deepcopy(meta_net)
            
            # Run inner loop using a pl trainer
            model = T.FlowTrainer(args, net=inner_net)
            trainer = pl.Trainer(gpus=1, logger=None, max_epochs=args.epochs, checkpoint_callback=False,
                                 check_val_every_n_epoch=args.epochs + 1, num_sanity_val_steps=0,
                                 weights_summary=None, progress_bar_refresh_rate=0)
            trainer.fit(model, video)
            with torch.no_grad():
                for meta_param, inner_param in zip(meta_net.parameters(), model.net.parameters()):
                    meta_param.grad = meta_param - inner_param
            meta_optim.step()
        
        # Validation
        if (epoch + 1) % args.log_iter == 0:
            video, _ = get_video(args.input_video, args)
            net = copy.deepcopy(meta_net)
            model = T.FlowTrainer(args, net=net, test_tag=f'meta_{epoch}')
            model.lr = args.lr / 5
            trainer = pl.Trainer(gpus=1, logger=None, max_epochs=args.epochs*10, checkpoint_callback=False,
                                 check_val_every_n_epoch=args.epochs*10 + 1, num_sanity_val_steps=0,
                                 weights_summary=None, progress_bar_refresh_rate=0)
            trainer.fit(model, video)
            trainer.test(model, video)

            torch.save({'epoch': epoch,
                        'model_state_dict': meta_net.state_dict(),
                        'optim_state_dict': meta_optim.state_dict(),
                       }, f'checkpoints/meta/epoch_{epoch}.pth')


def train_model(args):
    video, scene = get_video(args.input_video, args)
    logger, latest_ckpt = None, None
    if args.wandb:
        logger = WandbLogger(project='optical_flow', name=f'{scene}_{args.name}')
        logger.log_hyperparams(args)
        latest_ckpt = max(glob(path.join('checkpoints', scene, args.name, '*.ckpt')),
                          default=path.join('checkpoints', scene, args.name, 'temp'),
                          key=path.getmtime)
        clbks = [ModelCheckpoint(monitor='EPE', every_n_epochs=args.epochs//100,
                                 dirpath=path.dirname(latest_ckpt))]
    else:
        clbks = []
    dataset = video.testset
    if latest_ckpt and not path.isfile(latest_ckpt):
        logger.experiment.log({'source': wandb.Video((dataset.video * 255).type(torch.uint8))})
        if dataset.gt_available:
            flows = torch.stack([flow2img(f) for f in dataset.flow])
            logger.experiment.log({'gt_flow': wandb.Video(flows)})
        latest_ckpt = None

    model = T.FlowTrainer(args)
    trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=args.epochs,
                         checkpoint_callback=logger is not None,
                         callbacks=clbks,
                         resume_from_checkpoint=latest_ckpt,
                         check_val_every_n_epoch=args.log_iter,
                         num_sanity_val_steps=0, weights_summary=None)
    with ipdb.launch_ipdb_on_exception():
        trainer.fit(model, video)


def plot_fit(args):
    import matplotlib.pyplot as plt
    video, scene = get_video(args.input_video, args)
    latest_ckpt = max(glob(path.join('checkpoints', scene, args.name, '*.ckpt')),
                        default=path.join('checkpoints', scene, args.name, 'temp'),
                        key=path.getmtime)
    figure = plt.gcf()
    figure.set_size_inches(16, 12)
    dataset = video.testset
    model = T.FlowTrainer.load_from_checkpoint(latest_ckpt, args=args)
    model.eval()
    with torch.no_grad():
        t, _, h, w = dataset.video.shape
        xyt = torch.zeros(10000, 3)
        xyt[:,1] = torch.linspace(-1, 1, 10000)
        f = model.net(xyt) * dataset.flow_scale
        plt.plot(xyt[:,1], f[:,0], label='Varying x', color='g', alpha=0.5)
        f = dataset.flow[t//2,0,:,w//2].cpu()
        f[f == 0] = float('nan')
        plt.scatter(torch.linspace(-1, 1, h), f, color='g', marker='.')
        xyt = torch.zeros(10000, 3)
        xyt[:,0] = torch.linspace(-1, 1, 10000)
        f = model.net(xyt) * dataset.flow_scale
        plt.plot(xyt[:,0], f[:,0], label='Varying t', color='r', alpha=0.5)
        f = dataset.flow[:,0,h//2,w//2].cpu()
        f[f == 0] = float('nan')
        plt.scatter(torch.linspace(-1, 1, t)[:-1], f, color='r', marker='.')
        plt.legend()
        plt.savefig(f'results/fit_{scene}_{args.name}.pdf')


def test_model(args):
    video, scene = get_video(args.input_video, args)
    unique_name = f'{scene}_{args.step}_{args.name}'
    latest_ckpt = max(glob(path.join('checkpoints', scene, args.name, '*.ckpt')),
                        default=path.join('checkpoints', scene, args.name, 'temp'),
                        key=path.getmtime)
    dataset = video.testset
    model = T.FlowTrainer.load_from_checkpoint(latest_ckpt, args=args, test_tag=unique_name)
    trainer = pl.Trainer(gpus=1, logger=None)
    trainer.test(model, video)


if __name__ == "__main__":
    args = get_args()

    if args.operation == 'metatrain':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            train_metamodel(args)
    elif args.operation == 'train':
        train_model(args)
    elif args.operation == 'plot':
        plot_fit(args)
    elif args.operation == 'test':
        test_model(args)

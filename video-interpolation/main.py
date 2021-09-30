from glob import glob
import os, os.path as path
import argparse
from tqdm import tqdm

import torch, numpy as np, wandb, pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from data import get_video
import trainer as T
import model as M
import progressive_controller as C
from my_utils.flow_viz import flow2img
from my_utils.utils import writeFlow

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['train', 'test', 'summarize', 'sintel'])
    parser.add_argument('--ngpus', default=1, type=int)
    # Data options
    parser.add_argument('--input-video', default='../datasets/sintel/training/final/alley_1')
    parser.add_argument('--name', default='temp')
    parser.add_argument('--end', type=int)
    parser.add_argument('--step', type=int)
    parser.add_argument('--size', default=436, type=int)
    parser.add_argument('--batch', default=1, type=int)
    parser.add_argument('--test-size', default=436, type=int)
    parser.add_argument('--test-batch', default=1, type=int)
    # Network options
    parser.add_argument('--net', default='RBF')
    parser.add_argument('--spatially-adaptive', action='store_true')
    # Train options
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--val-iter', type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--loss-l1', default=1, type=float)
    parser.add_argument('--loss-census', default=0.1, type=float)
    parser.add_argument('--loss-ssim', default=0, type=float)
    parser.add_argument('--census-width', default=3, type=int)
    parser.add_argument('--loss-smooth1', default=0.1, type=float)
    parser.add_argument('--edge-constant', default=150, type=float)
    parser.add_argument('--edge-func', default='gauss', choices=['exp', 'gauss'])
    parser.add_argument('--occl', default='wang', choices=['brox', 'wang', None])
    parser.add_argument('--occl-thresh', default=0.7, type=float)
    # Logging options
    parser.add_argument('--wandb', choices=['optical_flow_exp', 'optical_flow', 'optical_flow_test'])
    parser.add_argument('--log-gt', action='store_true')
    return parser.parse_args()


def train_model(args):
    video, scene = get_video(args.input_video, args)
    dataset = video.testset
    if not args.val_iter:
        args.val_iter = args.epochs + 1

    logger, latest_ckpt, clbks = None, None, []
    if args.wandb:
        logger = WandbLogger(project=args.wandb, name=f'{scene}_{args.name}')
        logger.log_hyperparams(args)
        ckpt_dir = path.join('checkpoints', scene, args.name)
        clbks = [ModelCheckpoint(every_n_epochs=args.epochs//100, dirpath=ckpt_dir)]
        latest_ckpt = max(glob(path.join(ckpt_dir, '*.ckpt')), default=None, key=path.getmtime)
    if args.log_gt:
        logger.experiment.log({'source': wandb.Video((dataset.video * 255).type(torch.uint8))})
        if dataset.gt_available:
            flows = torch.stack([flow2img(f) for f in dataset.flow])
            logger.experiment.log({'gt_flow': wandb.Video(flows)})

    model = T.FlowTrainer(args)
    trainer = pl.Trainer(gpus=args.ngpus, logger=logger, max_epochs=args.epochs,
                         checkpoint_callback=logger is not None,
                         callbacks=clbks,
                         resume_from_checkpoint=latest_ckpt,
                         check_val_every_n_epoch=args.val_iter,
                         num_sanity_val_steps=0)
    trainer.fit(model, video)
    if model.completed_training:
        trainer.test(model, video)


def test_model(args):
    video, scene = get_video(args.input_video, args)
    unique_name = f'{scene}_{args.name}'
    latest_ckpt = max(glob(path.join('checkpoints', scene, args.name, '*.ckpt')),
                      key=path.getmtime)
    model = T.FlowTrainer.load_from_checkpoint(latest_ckpt, args=args, test_tag=unique_name)
    trainer = pl.Trainer(gpus=args.ngpus, logger=None)
    trainer.test(model, video)
    flow_files = [path.join('results', f) for f in os.listdir('results') \
                  if f.startswith(f'flow_{unique_name}_epe')]
    return flow_files, len(video.testset)


def summarize_model(args):
    root = path.dirname(args.input_video)
    epe_accum, frame_accum = 0, 0
    for scene in os.listdir(root):
        args.input_video = path.join(root, scene)
        files, num_frames = test_model(args)
        assert len(files) == 1
        epe = float(path.splitext(files[0])[0].split('_')[-1])
        epe_accum += epe*num_frames
        frame_accum += num_frames
    print(f'Normalized AEPE: {epe_accum/frame_accum}')


def sintel_submission(args):
    with torch.no_grad():
        device = 'cuda' if args.ngpus == 0 else 'cuda'
        root = path.dirname(args.input_video)
        for scene in tqdm(os.listdir(root)):
            video, _ = get_video(path.join(root, scene), args)
            latest_ckpt = max(glob(path.join('checkpoints', scene, args.name, '*.ckpt')),
                              key=path.getmtime)
            model = T.FlowTrainer.load_from_checkpoint(latest_ckpt, args=args)
            model.net.to(device)
            if args.name.endswith('clean'):
                outdir = path.join('sintel_submission', 'clean', scene)
            if args.name.endswith('final'):
                outdir = path.join('sintel_submission', 'final', scene)
            if not path.isdir(outdir):
                os.makedirs(outdir)
            for i, batch in enumerate(video.testset):
                f1, _, t, s = batch[:4]
                f1 ,t = f1.to(device).unsqueeze(0), t.to(device).unsqueeze(0)
                flow, _ = model(f1, t, s)
                flow = flow.squeeze(0).permute(1,2,0).cpu().numpy()
                writeFlow(path.join(outdir, f'frame_{i+1:04d}.flo'), flow)


if __name__ == "__main__":
    args = get_args()

    params = M.ModelParams()
    args.net = M.model_dict[args.net](params)
    if args.net.is_progressive:
        if args.spatially_adaptive:
            block_iterations = max(int(2 * args.epochs / args.net.encoding_dim), 1)
            args.net = C.StashedSpatialController(args.net, 50, args.epochs, epsilon=1e-3)
        else:
            args.net = C.LinearControllerEarly(args.net, args.epochs, epsilon=1e-3)

    if args.operation == 'train':
        train_model(args)
    elif args.operation == 'test':
        test_model(args)
    elif args.operation == 'summarize':
        summarize_model(args)
    elif args.operation == 'sintel':
        sintel_submission(args)

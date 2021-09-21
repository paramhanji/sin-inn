from glob import glob
import os, os.path as path
import argparse

import torch, numpy as np, wandb, pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from data import get_video
import trainer as T
import model as M
import progressive_controller as C
from my_utils.flow_viz import flow2img

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['train', 'test', 'summarize'])
    parser.add_argument('--ngpus', default=1, type=int)
    # Data options
    parser.add_argument('--input-video', default='../datasets/sintel/training/final/temple_3')
    parser.add_argument('--name', default='temp')
    parser.add_argument('--end', type=int)
    parser.add_argument('--step', type=int)
    parser.add_argument('--size', default=436, type=int)
    parser.add_argument('--batch', default=2, type=int)
    parser.add_argument('--test-size', default=436, type=int)
    parser.add_argument('--test-batch', default=2, type=int)
    parser.add_argument('--downsample', type=int)
    parser.add_argument('--downsample-type', choices=['nearest', 'bilinear', 'bicubic', 'blurpool'])
    # Network options
    parser.add_argument('--net', default='siren')
    parser.add_argument('--spatially-adaptive', action='store_true')
    # Train options
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--val-iter', default=50, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--wandb', choices=['optical_flow_exp', 'optical_flow'])
    parser.add_argument('--loss-photo', default='both', choices=['l1', 'census', 'both'])
    parser.add_argument('--census-width', default=3, type=int)
    parser.add_argument('--loss-smooth1', default=0.1, type=float)
    parser.add_argument('--edge-constant', default=150, type=float)
    parser.add_argument('--edge-func', default='gauss', choices=['exp', 'gauss'])
    parser.add_argument('--occl', default='wang', choices=['brox', 'wang', None])
    parser.add_argument('--occl-thresh', default=0.7, type=float)
    return parser.parse_args()


def train_model(args):
    video, scene = get_video(args.input_video, args)
    dataset = video.testset
    if not dataset.gt_available:
        args.val_iter = args.epochs + 1
    logger, latest_ckpt = None, None

    if args.wandb:
        logger = WandbLogger(project=args.wandb, name=f'{scene}_{args.name}')
        logger.log_hyperparams(args)
        latest_ckpt = max(glob(path.join('checkpoints', scene, args.name, '*.ckpt')),
                          default=path.join('checkpoints', scene, args.name, 'temp'),
                          key=path.getmtime)
        clbks = [ModelCheckpoint(every_n_epochs=args.epochs//100,
                                    dirpath=path.dirname(latest_ckpt))]

    else:
        clbks = []
    if latest_ckpt and not path.isfile(latest_ckpt):
        logger.experiment.log({'source': wandb.Video((dataset.video * 255).type(torch.uint8))})
        if dataset.gt_available:
            flows = torch.stack([flow2img(f) for f in dataset.flow])
            logger.experiment.log({'gt_flow': wandb.Video(flows)})
        latest_ckpt = None

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
                        default=path.join('checkpoints', scene, args.name, 'temp'),
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

import os, torch, argparse, logging
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from data import *
from lit_wrapper import SingleVideoINN

def get_args():
    ap = argparse.ArgumentParser(description='Train SR-Flow on single image')
    ap.add_argument('operation', choices=['train', 'test'])
    ap.add_argument('-g', '--gpu_ids', nargs='+', type=int, default=[0],
                    help='GPU id if multiple GPUs are present (check with nvidia-smi)')

    # Dataset opts
    ap.add_argument('--dataset', default='datasets/adobe240f',
                    help='root directory of dataset')
    ap.add_argument('-s','--scene', default='IMG_0028_binning_4x',
                    help='name of video from which files were extracted')
    ap.add_argument('--suffix', default='default', help='suffix for experiment name')
    ap.add_argument('-f', '--fps', type=int, default=10,
                    help='FPS of high-res frames (low-res frames are assumed to be 120fps)')
    ap.add_argument('--lr_window', type=int, default=10,
                    help='# of input low-res frames to use on either side of 1 high-res frame')
    ap.add_argument('-b', '--batch_size', type=int, default=8,
                    help='batch size to use; on 12GB 1080ti with default architecture and '
                          '640x360 HR frames use 8 for training and 40 for inference')

    # Architecture opts
    ap.add_argument('-a', '--architecture', choices=['SRF', 'IRN'],
                    default='SRF')
    ap.add_argument('--scale', type=int, default=4,
                    help='difference in resolution between HR and LR')
    ap.add_argument('-c', '--num_coupling', type=int, default=4,
                    help='number of GLOW blocks between downsamples')
    ap.add_argument('-r', '--resume_state', default=None, help='checkpoint to resume training')

    # Training log opts
    ap.add_argument('-w', '--working_dir', default='experiments',
                    help='directory to save logs and intermediate files')
    ap.add_argument('-e', '--epochs', type=int, default=10000)
    ap.add_argument('--save_iter', type=int, default=100,
                    help='frequency of checkpointing (in terms of # of epochs)')
    ap.add_argument('-p', '--print_iter', type=int, default=10,
                    help='frequency of logging (in terms of # of epochs)')

    # Training opts
    ap.add_argument('-l', '--learning_rate', type=float, default=1e-4,
                    help='initial learning rate')
    ap.add_argument('--adam_betas', type=float, nargs=2, default=[0.9, 0.99])
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    ap.add_argument('--lambda_fwd_rec', type=float, default=1)
    ap.add_argument('--lambda_fwd_mmd', type=float, default=0)
    ap.add_argument('--lambda_latent_nll', type=float, default=0)
    ap.add_argument('--lambda_bwd_rec', type=float, default=1)
    ap.add_argument('--lambda_bwd_mmd', type=float, default=0)
    ap.add_argument('--random_seed', type=int, default=0)

    # TCR opts
    ap.add_argument('--lambda_bwd_tcr', type=float, default=0)
    ap.add_argument('--rotation', type=float, default=5, help='in degrees')
    ap.add_argument('--translation', type=float, default=5, help='in pixels')
    ap.add_argument('--tcr_iters', type=float, default=5, help='samples per image')

    ap.add_argument('-t', '--temp', type=float, default=0.8, help='temperature to sample latents')
    ap.add_argument('--lr_dims', type=int, default=-1, help='internal: dimensionality of LR')
    ap.add_argument('--z_dims', type=int, default=-1, help='internal: dimensionality of latents')

    # TODO: based on https://github.com/VLL-HD/FrEIA#useful-tips-engineering-heuristics
    # ap.add_argument('-n', '--noise', type=float, default=0.01,
    #                 help='noise added to input to stabilise training')
    
    args = ap.parse_args()
    args.lr_dims = (2*args.lr_window + 1)*4
    args.z_dims = args.scale*args.scale*3*4 - args.lr_dims
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(args.random_seed)

    assert args.scale % 4 == 0
    if args.operation == 'test':
        assert os.path.isfile(args.resume_state)

    return args

if __name__ == '__main__':
    args = get_args()

    # Create datasets
    sup_data = VideoTrainDataset(args)
    unsup_data = VideoAllDataset(args)
    # TODO: shuffle unsup data
    train_data = ConcatDataset(sup_data, unsup_data)
    val_data = VideoValDataset(args, len(train_data)*4//6)      # 60-40 train-test ratio

    # Get image dimensions from dataset
    loader = get_loader(unsup_data)
    hr_img, lr_img = (b[0] for b in next(iter(loader)).values())
    model = SingleVideoINN(*hr_img.shape, args)

    if args.operation == 'train':
        exp_dir = os.path.join(args.working_dir, args.operation, f'{args.scene}_{args.architecture}_{args.suffix}')
        if not os.path.isdir(exp_dir):
            os.mkdir(exp_dir)
        wandb_logger = WandbLogger(project='sin-inn', save_dir=exp_dir,
                                   name=os.path.basename(exp_dir))
        wandb_logger.log_hyperparams(args)
        trainer = Trainer(auto_lr_find=True,
                          auto_scale_batch_size=True,
                          check_val_every_n_epoch=args.print_iter,
                          default_root_dir=exp_dir,
                          gpus=args.gpu_ids,
                          logger=wandb_logger,
                          max_epochs=args.epochs,
                          resume_from_checkpoint=args.resume_state,
                          callbacks=[ModelCheckpoint(period=args.save_iter)])
        loader = LitTrainLoader(train_data, val_data, args.batch_size)
        trainer.fit(model, loader)

    # TODO: test

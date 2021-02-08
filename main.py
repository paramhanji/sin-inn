import os, torch, argparse, logging
import imageio as io

from data import *
from models import UnconditionalSRFlow

data_root = '/local/scratch/pmh64/datasets/adobe240f'

def get_args():
    ap = argparse.ArgumentParser(description='Train SR-Flow on single image')
    ap.add_argument('operation', choices=['train', 'test'])
    ap.add_argument('-n', '--name', default='pilot', help='unique identifier')
    ap.add_argument('-d', '--debug', action="store_const", dest="loglevel",
                    const=logging.DEBUG, default=logging.WARNING)
    ap.add_argument('-v', '--verbose', action="store_const", dest="loglevel", const=logging.INFO)

    # Dataset opts
    ap.add_argument('--dataset', default='datasets/adobe240f',
                    help='root directory of dataset')
    ap.add_argument('-s','--scene', default='GOPR9653',
                    help='name of video from which files were extracted')
    ap.add_argument('-f', '--fps', type=int, default=10,
                    help='FPS of high-res frames (low-res frames are assumed to be 120fps)')
    ap.add_argument('--lr_window', type=int, default=10,
                    help='# of input low-res frames to use on either side of 1 high-res frame')

    # Architecture opts
    ap.add_argument('--scale', type=int, default=16,
                    help='difference in resolution between the 2 input streams')
    ap.add_argument('-c', '--num_coupling', type=int, default=2,
                    help='number of GLOW blocks between downsamples')
    ap.add_argument('-r', '--resume_state', help='checkpoint to resume training')
    ap.add_argument('-z', '--z_dims', type=int, default=-1,
                    help='number of latent channels')
    ap.add_argument('-x', '--x_dims', type=int, default=-1,
                    help='number of latent channels')

    # Training log opts
    ap.add_argument('-w', '--working_dir', default='experiments',
                    help='directory to save logs and intermediate files')
    ap.add_argument('-e', '--epochs', type=int, default=1000)
    ap.add_argument('--save_iter', type=int, default=100,
                    help='frequency of checkpointing (in terms of # of epochs)')
    ap.add_argument('-p', '--print_iter', type=int, default=10,
                    help='frequency of logging (in terms of # of epochs)')

    # Training opts
    ap.add_argument('-l', '--learning_rate', type=float, default=1e-4,
                    help='initial learning rate')
    ap.add_argument('-a', '--adam_betas', nargs=2, default=[0.9, 0.95])
    ap.add_argument('--lambda_fwd_rec', type=float, default=1)
    ap.add_argument('--lambda_fwd_mmd', type=float, default=50)
    ap.add_argument('--lambda_bwd_rec', type=float, default=1)
    ap.add_argument('--lambda_bwd_mmd', type=float, default=500)
    ap.add_argument('--random_seed', type=int, default=0)

    # TODO: based on https://github.com/VLL-HD/FrEIA#useful-tips-engineering-heuristics
    # ap.add_argument('-n', '--noise', type=float, default=0.01,
    #                 help='noise added to input to stabilise training')
    ap.add_argument('--sigma', type=float, default=1, help='sigma for latent space')
    
    args = ap.parse_args()
    logging.basicConfig(level=args.loglevel)
    torch.manual_seed(args.random_seed)

    assert args.scale % 4 == 0
    if args.operation == 'test':
        assert os.path.isfile(args.resume_state)

    return args

# test = os.path.join(data_root, 'hr_frames', 'GOPR9653', 'frame_00001.png')
# img = torch.FloatTensor([io.imread(test).transpose(-1, 0, 1)/255.])


if __name__ == '__main__':
    args = get_args()

    # Create data loader
    if args.operation == 'train':
        data = VideoTrainDataset(args)
    elif args.operation == 'test':
        data = VideoAllDataset(args)
    loader = get_loader(data, batch=6)
    hr_img = data[0]['hr'].to('cuda')
    lr_img = data[0]['lr']

    # Define the model
    inn = UnconditionalSRFlow(*hr_img.shape, args)

    # Figure out number of latent variables
    out = inn.infer([next(iter(loader))], args)
    args.x_dims = lr_img.shape[0]
    args.z_dims = out.shape[1] - args.x_dims + 1
    logging.info(f'HR_dims: {hr_img.shape}')
    logging.info(f'LR_dims: {lr_img.shape}')
    latents = torch.normal(0, args.sigma, size=out[0,-args.z_dims:,:,:].squeeze().shape)
    logging.info(f'latent_dims: {latents.shape}')
    assert torch.cat((lr_img, latents), dim=0).numel() == hr_img.numel()

    if args.operation == 'train':
        inn.bidirectional_train(loader, args)
    elif args.operation == 'test':
        # TODO add test loader
        inn.infer(loader, args, rev=True, save_videos=True)

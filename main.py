import os, torch, argparse, logging
import imageio as io

from data import *
from models import SingleVideoINN

data_root = '/local/scratch/pmh64/datasets/adobe240f'

def get_args():
    ap = argparse.ArgumentParser(description='Train SR-Flow on single image')
    ap.add_argument('operation', choices=['train', 'test'])
    ap.add_argument('-d', '--debug', action="store_const", dest="loglevel",
                    const=logging.DEBUG, default=logging.WARNING)
    ap.add_argument('-v', '--verbose', action="store_const", dest="loglevel", const=logging.INFO)
    ap.add_argument('-g', '--gpu_id', type=int, default=0,
                    help='GPU id if multiple GPUs are present (check with nvidia-smi)')

    # Dataset opts
    ap.add_argument('--dataset', default='datasets/adobe240f',
                    help='root directory of dataset')
    ap.add_argument('-s','--scene', default='GOPR9653',
                    help='name of video from which files were extracted')
    ap.add_argument('--suffix', help='suffix for experiment name')
    ap.add_argument('-f', '--fps', type=int, default=10,
                    help='FPS of high-res frames (low-res frames are assumed to be 120fps)')
    ap.add_argument('--lr_window', type=int, default=10,
                    help='# of input low-res frames to use on either side of 1 high-res frame')
    ap.add_argument('-b', '--batch_size', type=int, default=6,
                    help='batch size to use; on 12GB 1080ti with default architecture and\
                    22x40 input-frame resolution use 6 for training and 40 for inference')

    # Architecture opts
    ap.add_argument('-a', '--architecture', choices=['UncondSRFlow', 'InvRescaleNet'],
                    default='UncondSRFlow')
    ap.add_argument('--scale', type=int, default=16,
                    help='difference in resolution between the 2 input streams')
    ap.add_argument('-c', '--num_coupling', type=int, default=2,
                    help='number of GLOW blocks between downsamples')
    ap.add_argument('-r', '--resume_state', help='checkpoint to resume training')

    # Training log opts
    ap.add_argument('-w', '--working_dir', default='experiments',
                    help='directory to save logs and intermediate files')
    ap.add_argument('-e', '--epochs', type=int, default=10000)
    ap.add_argument('--save_iter', type=int, default=1000,
                    help='frequency of checkpointing (in terms of # of epochs)')
    ap.add_argument('-p', '--print_iter', type=int, default=10,
                    help='frequency of logging (in terms of # of epochs)')

    # Training opts
    ap.add_argument('-l', '--learning_rate', type=float, default=1e-4,
                    help='initial learning rate')
    ap.add_argument('--adam_betas', nargs=2, default=[0.9, 0.95])
    ap.add_argument('--lambda_fwd_rec', type=float, default=1)
    ap.add_argument('--lambda_fwd_mmd', type=float, default=0)
    ap.add_argument('--lambda_latent_nll', type=float, default=1)
    ap.add_argument('--lambda_bwd_rec', type=float, default=1)
    ap.add_argument('--lambda_bwd_mmd', type=float, default=0)
    ap.add_argument('--random_seed', type=int, default=0)

    ap.add_argument('-t', '--temp', type=float, default=0.8, help='temperature to sample latents')
    ap.add_argument('--lr_dims', type=int, default=-1, help='internal: dimensionality of LR')
    ap.add_argument('--z_dims', type=int, default=-1, help='internal: dimensionality of latents')

    # TODO: based on https://github.com/VLL-HD/FrEIA#useful-tips-engineering-heuristics
    # ap.add_argument('-n', '--noise', type=float, default=0.01,
    #                 help='noise added to input to stabilise training')
    
    args = ap.parse_args()
    # Assuming 16x SR, 16*16*h*w*3 == lr_dims*h*w + z_dims*h*w
    args.lr_dims = (2*args.lr_window + 1)*3
    args.z_dims = 16*16*3 - args.lr_dims
    logging.basicConfig(level=args.loglevel)
    torch.manual_seed(args.random_seed)
    torch.cuda.set_device(args.gpu_id)

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
        val_data = VideoValDataset(args, 15)
        val_loader = get_loader(data, batch=15)
    elif args.operation == 'test':
        data = VideoAllDataset(args)
    loader = get_loader(data, batch=args.batch_size)

    # Modify the name for experiment
    args.scene = f'{args.scene}_{args.architecture}'
    if args.suffix:
        args.scene = f'{args.scene}_{args.suffix}'

    hr_img = data[0]['hr'].to('cuda')
    lr_img = data[0]['lr']

    # Define the model
    inn = SingleVideoINN(*hr_img.shape, args)

    # Figure out number of latent variables
    if args.loglevel == logging.DEBUG:
        out, _ = inn.infer([next(iter(loader))], args)
        latents = torch.randn(out[0,args.lr_dims:,:,:].shape)
        logging.debug(f'HR_dims: {hr_img.shape}')
        logging.debug(f'LR_dims: {lr_img.shape}')
        logging.debug(f'Latent_dims: {latents.shape}')
        assert lr_img.numel() + latents.numel() == hr_img.numel()

    if args.operation == 'train':
        inn.bidirectional_train(loader, val_loader, args)
    elif args.operation == 'test':
        # args.temp = 0.01    # remove this after comparing with previous results
        inn.infer(loader, args, rev=True, save_videos=True)

import logging, os, subprocess as sp
from alive_progress import alive_bar
import numpy as np
from PIL import Image

import torch, torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import FrEIA.framework as Ff
import FrEIA.modules as Fm

import loss

def subnet_fc(c_in, c_out):
    return nn.Sequential(nn.Linear(c_in, 512), nn.ReLU(),
                         nn.Linear(512,  c_out))

def subnet_conv(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, 256,   3, padding=1), nn.ReLU(),
                         nn.Conv2d(256,  c_out, 3, padding=1))

def subnet_conv_1x1(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, 256,   1), nn.ReLU(),
                         nn.Conv2d(256,  c_out, 1))


class UnconditionalSRFlow():
    """
    Unconditional version of SR flow (ECCV '20)
    https://github.com/andreas128/SRFlow
    """
    def __init__(self, c, h, w, opt):
        self.scale = opt.scale
        self.k = opt.num_coupling
        
        # Define the model
        self.nodes = [Ff.InputNode(c, h, w, name='input')]

        for ss in range(self.scale//4):
            # Squeeze
            self.nodes.append(Ff.Node(self.nodes[-1],
                                      Fm.IRevNetDownsampling,
                                      {},
                                      name=f'squeeze_{ss}'))

            # Single transition step
            # TODO: How do we compute M to pass to 1x1Conv?
            # SR-flow paper states this prevents block artifacts
            # self.nodes.append(Ff.Node(self.nodes[-1],
            #                           Fm.ActNorm,
            #                           {},
            #                           name=f'actnorm_{ss}'))
            # self.nodes.append(Ff.Node(self.nodes[-1],
            #                           Fm.Fixed1x1Conv,
            #                           {},
            #                           name=f'conv_1x1_{ss}'))

            # k GLOW blocks
            for kk in range(self.k):
                # Use mixture of regular conv and 1x1 conv
                # https://github.com/VLL-HD/FrEIA#useful-tips-engineering-heuristics
                if kk % 2 == 0:
                    subnet = subnet_conv
                else:
                    subnet = subnet_conv_1x1
                self.nodes.append(Ff.Node(self.nodes[-1],
                                          Fm.GLOWCouplingBlock,
                                          {'subnet_constructor':subnet, 'clamp':1.2},
                                          name=f'glow_{ss}_{kk}'))
                self.nodes.append(Ff.Node(self.nodes[-1],
                                          Fm.PermuteRandom,
                                          {'seed':kk},
                                          name=f'permute_{ss}_{kk}'))

        self.nodes.append(Ff.OutputNode(self.nodes[-1], name='output'))
        self.inn = Ff.ReversibleGraphNet(self.nodes, verbose=(opt.loglevel == logging.INFO)).to('cuda')

        if opt.operation == 'train':
            params_trainable = list(filter(lambda p: p.requires_grad, self.inn.parameters()))

            # TODO: initalise params, adam opts, weight decay
            # https://github.com/VLL-HD/analyzing_inverse_problems/blob/6f1334622995ba64b2ddfea8d992f61eb045c875/inverse_problems_science/model.py
            self.optimizer = torch.optim.Adam(params_trainable,
                                              lr=opt.learning_rate,
                                              betas=opt.adam_betas)
            self.epoch_start = 0

        if opt.resume_state:
            checkpoint = torch.load(opt.resume_state)
            self.inn.load_state_dict(checkpoint['model_state_dict'])
            if opt.operation == 'train':
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epoch_start = checkpoint['epoch']

    def bidirectional_train(self, loader, opt):
        """
        Start with forward pass (HR->LR) and compute reconstruction + likelihood + MMD loss.
        Then reverse pass (LR->HR) and compute reconstruction + MMD loss.
        Finally, update the gradients with the total loss.
        TODO: add_noise
        """

        # Archive experiment dir if it exists
        exp_dir = os.path.join(opt.working_dir, opt.operation, f'{opt.scene}_{opt.fps}')
        if os.path.isdir(exp_dir) and not opt.resume_state:
            from datetime import datetime
            os.rename(exp_dir, f'{exp_dir}_{datetime.now()}')
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir)
        # TODO: What about multiple runs?
        writer = SummaryWriter(exp_dir)

        self.inn.train()
        with alive_bar(opt.epochs - self.epoch_start) as bar:
            for e in range(self.epoch_start, opt.epochs):
                for batch in loader:
                    # Forward pass
                    hr = Variable(batch['hr']).to('cuda')
                    lr_hat = self.inn(hr)

                    # Sample latents and construct GT [lr, y]
                    lr = batch['lr']
                    z = torch.normal(0, opt.sigma, size=lr_hat[:,-opt.z_dims:,:,:].shape)
                    lr = torch.cat((lr, z), dim=1)
                    lr = Variable(lr).to('cuda')
                    # Backward pass
                    hr_hat = self.inn(lr, rev=True)
                    self.optimizer.zero_grad()

                    losses= {}
                    # Add all forward losses
                    losses['fwd_rec'] = opt.lambda_fwd_rec * loss.reconstruction(lr_hat[:,:-opt.z_dims,:,:], lr[:,:-opt.z_dims,:,:])
                    lr_grad_blocked = torch.cat((lr_hat[:,:-opt.z_dims,:,:],
                                                 lr_hat[:,-opt.z_dims:,:,:].data), dim=1)
                    losses['fwd_mmd'] = opt.lambda_fwd_mmd * loss.mmd(lr_grad_blocked, lr)


                    # Add all inverse losses
                    losses['bwd_rec'] = opt.lambda_bwd_rec * loss.reconstruction(hr_hat, hr)
                    losses['bwd_mmd'] = opt.lambda_bwd_mmd * loss.mmd(hr_hat, hr, rev=True)
                    
                    total_loss = torch.stack(list(losses.values()), dim=0).sum()
                    total_loss.backward()
                    self.optimizer.step()

                    losses = {k: v.item() for k, v in losses.items()}

                if (e+1) % opt.save_iter == 0:
                    save_path = os.path.join(exp_dir, f'epoch_{e+1:05d}.pth')
                    logging.info(f'Saving state at {save_path}')
                    torch.save({'model_state_dict': self.inn.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'epoch': e+1}, save_path)

                if (e+1) % opt.print_iter == 0:
                    logging.info(losses)
                    writer.add_scalar('Loss/train', total_loss, e)
                    for l in losses:
                        writer.add_scalar(f'Loss/{loss}', losses[l], e)
                bar()

        writer.close()


    def infer(self, loader, opt, rev=False, temp=0.8, save_images=False, save_videos=False):
        self.inn.eval()
        trans = transforms.ToPILImage()
        
        if save_videos:
            save_path = os.path.join(opt.working_dir, opt.operation,
                                     f'{opt.scene}_{opt.fps}_{os.path.basename(opt.resume_state).strip(".pth")}',
                                     'videos')
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            # Create subprocess pipes to feed ffmpeg
            dump = open(os.devnull, 'w')
            video_in = sp.Popen(['ffmpeg', '-framerate', '60', '-i', '-', '-vf', 'scale=iw*16:ih*16',
                                 '-c:v', 'libx264', '-preset', 'ultrafast', '-y',
                                 os.path.join(save_path, 'in.avi')],
                                stdin=sp.PIPE, stderr=dump)
            video_out = sp.Popen(['ffmpeg', '-framerate', '60', '-i', '-', '-c:v', 'libx264',
                                  '-preset', 'ultrafast', '-y', os.path.join(save_path, 'out.avi')],
                                  stdin=sp.PIPE, stderr=dump)
            video_gt = sp.Popen(['ffmpeg', '-framerate', '60', '-i', '-', '-c:v', 'libx264',
                                 '-preset', 'ultrafast', '-y', os.path.join(save_path, 'gt.avi')],
                                stdin=sp.PIPE, stderr=dump)

        with alive_bar(len(loader)) as bar:
            for bb, batch in enumerate(loader):
                b, c, h, w = batch['lr'].shape

                if rev:                    
                    imgs_in = []
                    gt = [trans(img) for img in batch['hr']]
                    for lr in batch['lr']:
                        col = Image.new('RGB', (w, h*c//3))
                        for i in range(0, c, 3):
                            img = trans(lr[i: i+3])
                            col.paste(img, (0, i//3 * h))
                        imgs_in.append(col)

                    # Sample from latents
                    z = torch.normal(0, temp*opt.sigma, size=(b, opt.z_dims, h, w))
                    input = torch.cat((batch['lr'], z), dim=1)
                else:
                    input = batch['hr']
                    imgs_in = [trans(img) for img in batch['hr']]

                # The forward/backward pass
                with torch.no_grad():
                    input = input.to('cuda')
                    output = self.inn.forward(input, rev=rev).to('cpu')

                if rev:
                    imgs_out = [trans(img) for img in output]
                else:
                    # Remove latents
                    output = output[:,:opt.x_dims,:,:]

                    imgs_out = []
                    for imgs in output:
                        row = Image.new('RGB', (w, h*c//3))
                        for i in range(0, c, 3):
                            img = trans(imgs[i: i+3])
                            row.paste(img, (0, i//3 * h))
                        imgs_out.append(row)

                if save_images:
                    save_path = os.path.join(opt.working_dir, opt.operation,
                                             f'{opt.scene}_{opt.fps}_{os.path.basename(opt.resume_state).strip(".pth")}',
                                             'frames')
                    if not os.path.isdir(save_path):
                        os.makedirs(save_path)
                    for i, (im_in, im_out, im_gt) in enumerate(zip(imgs_in, imgs_out, gt)):
                        im_in.save(os.path.join(save_path, f'in_{bb:04d}_{i:02d}.png'))
                        im_out.save(os.path.join(save_path, f'out_{bb:04d}_{i:02d}.png'))
                        im_gt.save(os.path.join(save_path, f'gt_{bb:04d}_{i:02d}.png'))

                if save_videos:
                    for i, (im_in, im_out, im_gt) in enumerate(zip(imgs_in, imgs_out, gt)):
                        # Extract only middle frame
                        im_in = im_in.crop((0, h*opt.lr_window//2, w, h*(opt.lr_window//2 + 1)))
                        im_in.save(video_in.stdin, 'PNG')
                        im_out.save(video_out.stdin, 'PNG')
                        im_gt.save(video_gt.stdin, 'PNG')
                bar()

        if save_videos:
            video_in.stdin.close(); video_out.communicate()
            video_out.stdin.close(); video_out.communicate()
            video_gt.stdin.close(); video_gt.communicate()

        return output
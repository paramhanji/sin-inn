import logging, os, subprocess as sp, pydoc
from alive_progress import alive_bar
import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import loss

class SingleVideoINN():
    def __init__(self, c, h, w, opt):
        self.opt = opt
        self.inn = None
        for arch in ('UncondSRFlow', 'InvRescaleNet'):
            if opt.architecture == arch:
                arch_module = pydoc.locate(f'archs.{arch}')
                # self.inn = arch_module(c, h, w, opt).inn
                self.inn = arch_module(c, h, w, opt).to('cuda')
        assert self.inn is not None, f'Architecture not defined'
        logging.debug(self.inn)

        params = sum(p.numel() for p in self.inn.parameters())
        logging.info(f'Loaded model with {params} parameters to GPU {opt.gpu_id}')
        
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

    def bidirectional_train(self, train_loader, val_loader, opt):
        """
        Start with forward pass (HR->LR) and compute reconstruction + likelihood + MMD loss.
        Then reverse pass (LR->HR) and compute reconstruction + MMD loss.
        Finally, update the gradients with the total loss.
        TODO: add_noise
        """

        # # Archive experiment dir if it exists
        exp_dir = os.path.join(opt.working_dir, opt.operation, f'{opt.scene}_{opt.fps}')
        if os.path.isdir(exp_dir) and not opt.resume_state:
            from datetime import datetime
            os.rename(exp_dir, f'{exp_dir}_{datetime.now()}')
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir)
        writer = SummaryWriter(exp_dir)

        self.inn.train()
        with alive_bar(opt.epochs - self.epoch_start) as bar:
            for e in range(self.epoch_start, opt.epochs):
                for batch in train_loader:
                    ''' TODO:
                    1. Perturb with gaussian noise
                    2. Backward MMD might need to be ramped up
                    3. Can model be trained without backward rec loss?
                    4. Hyperparameters (lambdas)
                    '''
                    self.optimizer.zero_grad()
                    b, _, h, w = batch['lr'].shape


                    hr = Variable(batch['hr']).to('cuda')
                    lr = batch['lr'].to('cuda')
                    z = torch.randn(b, opt.z_dims, h, w).to('cuda')
                    lr_z = Variable(torch.cat((lr, z), dim=1))

                    # Forward pass
                    lr_z_hat = self.inn(hr)
                    blocked_lr_z_hat = torch.cat((lr_z_hat[:,:opt.lr_dims,:,:].data,
                                                  lr_z_hat[:,opt.lr_dims:,:,:]), dim=1)
                    loss_fwd = opt.lambda_fwd_rec * loss.reconstruction(lr_z_hat[:,:opt.lr_dims,:,:], lr)
                    loss_fwd += opt.lambda_fwd_mmd * loss.mmd(blocked_lr_z_hat, lr_z)
                    loss_fwd += opt.lambda_latent_nll * loss.latent_nll(lr_z_hat[:,opt.lr_dims:,:,:])
                    loss_fwd.backward()

                    # Backward pass
                    # TODO: perturb lr_hat, retain z and compute reconstruction loss
                    hr_hat = self.inn(lr_z, rev=True)
                    loss_bwd = opt.lambda_bwd_rec * loss.reconstruction(hr_hat, hr)
                    loss_bwd += opt.lambda_bwd_mmd * loss.mmd(hr_hat, hr, rev=True)
                    loss_bwd.backward()

                    self.optimizer.step()

                if (e+1) % opt.save_iter == 0:
                    save_path = os.path.join(exp_dir, f'epoch_{e+1:05d}.pth')
                    logging.info(f'Saving state at {save_path}')
                    torch.save({'model_state_dict': self.inn.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'epoch': e+1}, save_path)

                if (e+1) % opt.print_iter == 0:
                    writer.add_scalar('Loss/train', loss_fwd + loss_bwd, e)
                    del hr, lr
                    _, fwd_losses = self.infer(val_loader, opt)
                    _, bwd_losses = self.infer(val_loader, opt, rev=True)
                    all_losses = {**fwd_losses, **bwd_losses}
                    logging.info(all_losses)
                    self.inn.train()
                    for l in all_losses:
                        writer.add_scalar(f'Loss/{l}', all_losses[l], e)
                bar()

        writer.close()


    def infer(self, loader, opt, rev=False, save_images=False, save_videos=False):
        self.inn.eval()
        trans = transforms.ToPILImage()
        losses = dict.fromkeys(['fwd_rec','fwd_mmd','latent_nll','bwd_rec','bwd_mmd'], 0)

        
        if save_videos:
            save_path = os.path.join(opt.working_dir, opt.operation,
                                     f'{opt.scene}_{opt.fps}_{os.path.basename(opt.resume_state).strip(".pth")}',
                                     'videos')
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            # Create subprocess pipes to feed ffmpeg
            dump = open(os.devnull, 'w')
            video_in = sp.Popen(['ffmpeg', '-framerate', '120', '-i', '-', '-vf', 'scale=iw*16:ih*16',
                                 '-c:v', 'libx264', '-preset', 'ultrafast', '-y',
                                 os.path.join(save_path, 'in.avi')],
                                stdin=sp.PIPE, stderr=dump)
            video_out = sp.Popen(['ffmpeg', '-framerate', '120', '-i', '-', '-c:v', 'libx264',
                                  '-preset', 'ultrafast', '-y', os.path.join(save_path, 'out.avi')],
                                  stdin=sp.PIPE, stderr=dump)
            video_gt = sp.Popen(['ffmpeg', '-framerate', '120', '-i', '-', '-c:v', 'libx264',
                                 '-preset', 'ultrafast', '-y', os.path.join(save_path, 'gt.avi')],
                                stdin=sp.PIPE, stderr=dump)

        with alive_bar(len(loader)) as bar:
            for bb, batch in enumerate(loader):
                b, c, h, w = batch['lr'].shape
                lr = batch['lr'].to('cuda')
                hr = batch['hr'].to('cuda')


                if rev:
                    if save_images or save_videos:
                        imgs_in = []
                        gt = [trans(img) for img in hr]
                        for lr_img in lr:
                            col = Image.new('RGB', (w, h*c//3))
                            for i in range(0, c, 3):
                                img = trans(lr_img[i: i+3])
                                col.paste(img, (0, i//3 * h))
                            imgs_in.append(col)

                    # Sample from latents
                    z = opt.temp * torch.randn(b, opt.z_dims, h, w).to('cuda')
                    input = torch.cat((lr, z), dim=1)
                else:
                    input = hr
                    imgs_in = [trans(img) for img in hr]

                # The forward/backward pass
                with torch.no_grad():
                    input = input.to('cuda')
                    output = self.inn.forward(input, rev=rev)

                if rev:
                    losses['bwd_rec'] += loss.reconstruction(output, hr)
                    losses['bwd_mmd'] += loss.mmd(output, hr, rev=True)

                    if save_images or save_videos:
                        output.to('cpu')
                        imgs_out = [trans(img) for img in output]
                else:
                    losses['fwd_rec'] += loss.reconstruction(output[:,:opt.lr_dims,:,:], lr)
                    losses['fwd_mmd'] += loss.mmd(output[:,:opt.lr_dims,:,:], lr)
                    losses['latent_nll'] += loss.latent_nll(output[:,opt.lr_dims:,:,:])

                    if save_images or save_videos:
                        # Remove latents
                        output = output[:,:opt.lr_dims,:,:]
                        output.to('cpu')
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

        losses = {k: v.item()/len(loader) for k, v in losses.items() if v != 0}
        return output, losses
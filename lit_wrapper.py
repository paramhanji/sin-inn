import logging, os, subprocess as sp
from tqdm import tqdm

import torch, pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import ProgressBarBase
import torchvision.transforms as transforms

from archs import UncondSRFlow, InvRescaleNet
from tcr import TCR
import loss

class SingleVideoINN(pl.LightningModule):

    def __init__(self, c, h, w, opt):
        super().__init__()
        self.save_hyperparameters()
        arch_module = {'SRF': UncondSRFlow, 'IRN':InvRescaleNet}
        self.opt = opt
        self.inn = arch_module[opt.architecture](c, h, w, opt)

        params = sum(p.numel() for p in self.inn.parameters())
        logging.info(f'Created model with {params/1e5:.2f}M parameters. Using GPUs {opt.gpu_ids}')

        self.tcr = TCR(opt.rotation, opt.translation)

        # Disable automatic optimization - needed since we use multiple backward() calls
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        """
        Start with forward pass (HR->LR) and compute reconstruction + likelihood + MMD loss.
        Then reverse pass (LR->HR) and compute reconstruction + MMD loss.
        Finally, update the gradients with the total loss.
        TODO: add_noise
        """
        opt = self.optimizers()
        opt.zero_grad()
        b, _, h, w = batch[0]['lr'].shape

        hr, lr = (batch[0][k] for k in ('hr', 'lr'))
        z = torch.randn(b, self.opt.z_dims, h, w, device=hr.device)
        lr_z = torch.cat((lr, z), dim=1)

        # Forward pass
        lr_z_hat = self.inn(hr)
        fwd_loss = self.opt.lambda_fwd_rec * loss.reconstruction(lr_z_hat[:,:self.opt.lr_dims,:,:], lr)
        fwd_loss += self.opt.lambda_fwd_mmd * loss.mmd(lr_z_hat, lr_z)
        fwd_loss += self.opt.lambda_latent_nll * loss.latent_nll(lr_z_hat[:,self.opt.lr_dims:,:,:])
        self.manual_backward(fwd_loss)

        # Backward pass
        # TODO: perturb lr_hat, retain z and compute reconstruction loss
        hr_hat = self.inn(lr_z, rev=True)
        bwd_loss = self.opt.lambda_bwd_rec * loss.reconstruction(hr_hat, hr)
        bwd_loss += self.opt.lambda_bwd_mmd * loss.mmd(hr_hat, hr, rev=True)
        self.manual_backward(bwd_loss)

        if self.opt.lambda_bwd_tcr > 0:
            # TCR - Apply in backward pass to unseen LR in unsupervised manner
            # TODO: Apply to provided HR?
            # TODO: Use a transformation with gradient
            hr, lr = (batch[1][k] for k in ('hr', 'lr'))
            for i in range(self.opt.tcr_iters):
                rand = torch.rand(b, 3)
                z = torch.randn(b, self.opt.z_dims, h, w, device=hr.device)
                lr_z = torch.cat((lr, z), dim=1)

                tcr_lr_z = torch.cat((self.tcr(lr, rand, scale=1/self.opt.scale), z), dim=1)
                tcr_hr_hat = self.inn(tcr_lr_z, rev=True)
                hr_hat_tcr = self.tcr(self.inn(lr_z, rev=True), rand)
                tcr_loss = self.opt.lambda_bwd_tcr / self.opt.tcr_iters * loss.reconstruction(tcr_hr_hat, hr_hat_tcr)
                self.manual_backward(tcr_loss)
        else:
            tcr_loss = 0

        opt.step()
        self.log('train', fwd_loss + bwd_loss + tcr_loss)

    def validation_step(self, batch, batch_idx):
        b, _, h, w = batch['lr'].shape
        hr, lr = (batch[k] for k in ('hr', 'lr'))
        z = torch.randn(b, self.opt.z_dims, h, w, device=hr.device)
        lr_z = torch.cat((lr, z), dim=1)

        lr_z_hat = self.inn(hr)
        hr_hat = self.inn(lr_z, rev=True)
        self.log('lr_acc', loss.reconstruction(lr_z_hat[:,:self.opt.lr_dims,:,:], lr))
        self.log('hr_acc', loss.reconstruction(hr_hat, hr))
        self.log('z_nll', loss.latent_nll(lr_z_hat[:,self.opt.lr_dims:,:,:]))

    def infer(self, loader, opt, save_images=None, save_video=None):
        self.inn.eval()
        self.inn.to('cuda')
        trans = transforms.ToPILImage()

        if save_video:
            # Create subprocess pipes to feed ffmpeg
            dump = open(os.devnull, 'w')
            fps = '30'
            crf = '18'
            video = sp.Popen(['ffmpeg', '-framerate', fps, '-i', '-',
                              '-c:v', 'libx264', '-preset', 'veryslow', '-crf', crf, '-y',
                              save_video], stdin=sp.PIPE, stderr=dump)

        for bb, batch in enumerate(tqdm(loader)):
            b, c, h, w = batch['lr'].shape
            lr = batch['lr'].to('cuda')

            # Sample from latents
            z = opt.temp * torch.randn(b, opt.z_dims, h, w, device=lr.device)
            lr_z = torch.cat((lr, z), dim=1)

            # The reverse pass
            with torch.no_grad():
                hr_hat = self.inn.forward(lr_z, rev=True)

            if save_images or save_video:
                hr_hat.to('cpu')
                for i, im_out in enumerate(hr_hat):
                    im_out = trans(im_out)
                    if save_images:
                        im_out.save(os.path.join(save_path, f'out_{bb:04d}_{i:02d}.png'))
                    else:
                        im_out.save(video.stdin, 'PNG')

        if save_video:
            video.stdin.close()
            video.communicate()


    def configure_optimizers(self):
        # TODO: initalise params, adam opts, weight decay
        # https://github.com/VLL-HD/analyzing_inverse_problems/blob/6f1334622995ba64b2ddfea8d992f61eb045c875/inverse_problems_science/model.py
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.opt.learning_rate,
                                     betas=self.opt.adam_betas,
                                     weight_decay=self.opt.weight_decay)
        return optimizer

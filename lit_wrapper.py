import pydoc, logging
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import loss
from tcr import TCR

class SingleVideoINN(pl.LightningModule):

    def __init__(self, c, h, w, train_data, val_data, test_data, opt):
        super().__init__()
        self.opt = opt
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.inn = None
        for arch in ('UncondSRFlow', 'InvRescaleNet'):
            if opt.architecture == arch:
                arch_module = pydoc.locate(f'archs.{arch}')
                self.inn = arch_module(c, h, w, opt).to('cuda')
        assert self.inn is not None, f'Architecture not defined'
        logging.debug(self.inn)

        params = sum(p.numel() for p in self.inn.parameters())
        logging.info(f'Loaded model with {params} parameters to GPUs {opt.gpu_ids}')

        self.tcr = TCR(opt.rotation, opt.translation)

    def training_step(self, batch, batch_idx):
        """
        Start with forward pass (HR->LR) and compute reconstruction + likelihood + MMD loss.
        Then reverse pass (LR->HR) and compute reconstruction + MMD loss.
        Finally, update the gradients with the total loss.
        TODO: add_noise
        """
        b, _, h, w = batch[0]['lr'].shape

        hr, lr = (batch[0][k] for k in ('hr', 'lr'))
        z = torch.randn(b, self.opt.z_dims, h, w, device=hr.device)
        lr_z = torch.cat((lr, z), dim=1)

        # Forward pass
        lr_z_hat = self.inn(hr)
        total_loss = self.opt.lambda_fwd_rec * loss.reconstruction(lr_z_hat[:,:self.opt.lr_dims,:,:], lr)
        total_loss += self.opt.lambda_fwd_mmd * loss.mmd(lr_z_hat, lr_z)
        total_loss += self.opt.lambda_latent_nll * loss.latent_nll(lr_z_hat[:,self.opt.lr_dims:,:,:])

        # Backward pass
        # TODO: perturb lr_hat, retain z and compute reconstruction loss
        hr_hat = self.inn(lr_z, rev=True)
        total_loss += self.opt.lambda_bwd_rec * loss.reconstruction(hr_hat, hr)
        total_loss += self.opt.lambda_bwd_mmd * loss.mmd(hr_hat, hr, rev=True)

        # TCR
        # Latents do not need tcr. Should we resample z?
        # Apply in backward pass to unseen LR in unsupervised manner
        # (Create a separate loader for LR without HR)
        # https://github.com/aamir-mustafa/Transformation-CR/blob/master/train_tcr.py
        # TODO: Apply to provided HR?
        rand = torch.rand(b, 3)
        hr, lr = (batch[1][k] for k in ('hr', 'lr'))
        # z = torch.randn(b, opt.z_dims, h, w).to('cuda')
        lr_z = torch.cat((lr, z), dim=1)

        tcr_lr_z = torch.cat((self.tcr(lr, rand, scale=1/self.opt.scale), z), dim=1)
        tcr_hr_hat = self.inn(tcr_lr_z, rev=True)
        hr_hat_tcr = self.tcr(self.inn(lr_z, rev=True), rand)
        total_loss += self.opt.lambda_bwd_tcr * loss.reconstruction(tcr_hr_hat, hr_hat_tcr)

        # self.log(total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        b, _, h, w = batch['lr'].shape
        hr, lr = (batch[k] for k in ('hr', 'lr'))
        z = torch.randn(b, self.opt.z_dims, h, w, device=hr.device)
        lr_z = torch.cat((lr, z), dim=1)

        lr_z_hat = self.inn(hr)
        hr_hat = self.inn(lr_z, rev=True)
        losses = {'lr_acc': loss.reconstruction(lr_z_hat[:,:self.opt.lr_dims,:,:], lr),
                  'hr_acc': loss.reconstruction(hr_hat, hr),
                  'z_nll': loss.latent_nll(lr_z_hat[:,self.opt.lr_dims:,:,:])}
        self.log_dict(losses)

    # TODO: forward


    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.opt.batch_size,
                          shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.opt.batch_size,
                          shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.opt.batch_size,
                          shuffle=False, num_workers=4)

    def configure_optimizers(self):
        # TODO: initalise params, adam opts, weight decay
        # https://github.com/VLL-HD/analyzing_inverse_problems/blob/6f1334622995ba64b2ddfea8d992f61eb045c875/inverse_problems_science/model.py
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.opt.learning_rate,
                                     betas=self.opt.adam_betas,
                                     weight_decay=self.opt.weight_decay)
        return optimizer

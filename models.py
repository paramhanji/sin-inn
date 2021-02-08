import logging, os, tqdm
import torch, torch.nn as nn
from torch.autograd import Variable
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
            # The paper states this prevents block artifacts
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
        # self.inn = Ff.ReversibleGraphNet(self.nodes, verbose=(opt.loglevel == logging.INFO))
        self.inn = Ff.ReversibleGraphNet(self.nodes, verbose=False).to('cuda')

        if opt.operation == 'train':
            params_trainable = list(filter(lambda p: p.requires_grad, self.inn.parameters()))

            # TODO: initalise params, adam opts, weight decay
            # https://github.com/VLL-HD/analyzing_inverse_problems/blob/6f1334622995ba64b2ddfea8d992f61eb045c875/inverse_problems_science/model.py
            self.optimizer = torch.optim.Adam(params_trainable,
                                              lr=opt.learning_rate,
                                              betas=opt.adam_betas)
            self.epoch_start = 0

        if opt.resume_state:
            checkpoint = opt.resume_state
            self.inn.load_state_dict(torch.load(checkpoint['model_state_dict']))
            if opt.operation == 'train':
                self.optimizer.load_state_dict(torch.load(checkpoint['optimizer_state_dict']))
                self.epoch_start = checkpoint['epoch']

    def bidirectional_train(self, loader, opt):
        """
        Start with forward pass (HR->LR) and compute reconstruction + likelihood + MMD loss.
        Then reverse pass (LR->HR) and compute reconstruction + MMD loss.
        Finally, update the gradients with the total loss.
        TODO: add_noise
        """

        # Archive experiment dir if it exists
        exp_dir = os.path.join(opt.working_dir, opt.operation, opt.name)
        if os.path.isdir(exp_dir):
            os.rename(exp_dir, f'{exp_dir}_old')
        os.makedirs(exp_dir)

        self.inn.train()
        for e in range(self.epoch_start, opt.epochs):
            if e != 0 and e % opt.save_iter == 0:
                save_path = os.path.join(exp_dir, f'epoch_{e:05d}.pth')
                logging.info(f'Saving state at {save_path}')
                torch.save({'model_state_dict': self.inn.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'epoch': e}, save_path)

            with tqdm.tqdm(loader, unit="batch") as tepoch:
                for batch in tepoch:
                    tepoch.set_description(f"Epoch {e}/{opt.epochs}")

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
                    tepoch.set_postfix(losses)


    def infer(self, x):
        self.inn.eval()
        with torch.no_grad():
            y_hat = self.inn.forward(x)
        return y_hat
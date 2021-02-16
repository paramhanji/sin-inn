import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

import FrEIA.framework as Ff
import FrEIA.modules as Fm

def subnet_fc(c_in, c_out):
    return nn.Sequential(nn.Linear(c_in, 512), nn.ReLU(),
                         nn.Linear(512,  c_out))

def subnet_conv(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, 256,   3, padding=1), nn.ReLU(),
                         nn.Conv2d(256,  c_out, 3, padding=1))

def subnet_conv_1x1(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, 256,   1), nn.ReLU(),
                         nn.Conv2d(256,  c_out, 1))

class UncondSRFlow():
    """
    Unconditional version of SR flow (ECCV '20)
    https://github.com/andreas128/SRFlow
    """
    def __new__(self, c, h, w, opt):        
        # Define the model
        self.nodes = [Ff.InputNode(c, h, w, name='input')]

        for ss in range(opt.scale//4):
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
            for kk in range(opt.num_coupling):
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
        return Ff.ReversibleGraphNet(self.nodes, verbose=False)


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            self.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        self.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5

    def initialize_weights(self, net_l, scale=1):
        if not isinstance(net_l, list):
            net_l = [net_l]
        for net in net_l:
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                    m.weight.data *= scale  # for residual block
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                    m.weight.data *= scale
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias.data, 0.0)

    def initialize_weights_xavier(self, net_l, scale=1):
        if not isinstance(net_l, list):
            net_l = [net_l]
        for net in net_l:
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)
                    m.weight.data *= scale  # for residual block
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    m.weight.data *= scale
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias.data, 0.0)

class InvBlockExp(nn.Module):
    def __init__(self, channel_num, channel_split_num, clamp=1.):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        self.F = DenseBlock(self.split_len2, self.split_len1)
        self.G = DenseBlock(self.split_len1, self.split_len2)
        self.H = DenseBlock(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)

class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1/16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)

class InvRescaleNet(nn.Module):
    def __init__(self, c, h, w, opt):
        super(InvRescaleNet, self).__init__()

        channel_out = opt.lr_dims
        operations = []

        current_channel = c
        for i in range(opt.scale//4):
            b = HaarDownsampling(current_channel)
            operations.append(b)
            current_channel *= 4
            for j in range(opt.num_coupling):
                # In the first few layers, we don't have enough channels
                b = InvBlockExp(current_channel, min(channel_out, current_channel//2))
                operations.append(b)

        self.operations = nn.ModuleList(operations)

    def forward(self, x, rev=False):
        out = x

        if not rev:
            for op in self.operations:
                out = op.forward(out, rev)
        else:
            for op in reversed(self.operations):
                out = op.forward(out, rev)

        return out

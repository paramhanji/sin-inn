from typing import Union, List, Tuple
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as nnf
import functools
import abc

T = torch.tensor
EPSILON = 1e-4


class ModelParams:

    def fill_args(self, **kwargs):
        for key, item in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, item)

    def __init__(self, **kwargs):
        self.domain_dim = 3
        self.num_frequencies = 256
        self.std = 25
        self.power = 20
        self.num_layers = 3
        self.hidden_dim = 256
        self.output_channels = 4
        self.num_frequencies_pe = 4
        self.std_rbf = 12
        self.fill_args(**kwargs)


class MLP(nn.Module):

    def forward(self, x: T) -> T:
        return self.model(x)

    def __init__(self, layers: Union[List[int], Tuple[int, ...]]):
        super(MLP, self).__init__()
        layers_ = []
        for i in range(len(layers) -1):
            layers_.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                layers_.append(nn.ReLU(True))
        self.model = nn.Sequential(*layers_)


class EncodingLayer(nn.Module, abc.ABC):

    @property
    @abc.abstractmethod
    def output_channels(self) -> int:
        raise NotImplemented


class EncodedMlpModel(nn.Module, abc.ABC):

    def update_progress(self):
        return

    def stash_iteration(self, *args):
        return

    @property
    def is_progressive(self) -> bool:
        return False

    @property
    def domain_dim(self):
        return self.opt.domain_dim

    @staticmethod
    def unsqueeze_mask(mask: T, ref: T):
        while mask.dim() != ref.dim():
            mask = mask.unsqueeze(0)
        return mask.to(ref.device)

    @property
    @abc.abstractmethod
    def encoding_dim(self):
        raise NotImplemented

    @abc.abstractmethod
    def get_encoding(self, x: T) -> T:
        raise NotImplemented

    @abc.abstractmethod
    def mlp_forward(self, x: T):
        raise NotImplemented

    def apply_control(self, x: T, *_, **kwargs) -> T:
        if 'override_mask' in kwargs and kwargs['override_mask'] is not None:
            mask = self.unsqueeze_mask(kwargs['override_mask'], x)
            x = x * mask
        return x

    def forward(self, x: T, *args, **kwargs) -> T:
        base_code = self.get_encoding(x)
        base_code = self.apply_control(base_code, *args, **kwargs)
        out = self.mlp_forward(base_code)
        return out

    def __init__(self, opt: ModelParams):
        super(EncodedMlpModel, self).__init__()
        self.opt = opt


class BaseModel(EncodedMlpModel):

    def get_encoding(self, x: T) -> T:
        return x

    def mlp_forward(self, x: T):
        return self.model(x)

    @property
    def encoding_dim(self):
        return self.opt.domain_dim

    def __init__(self, opt: ModelParams):
        super(BaseModel, self).__init__(opt)
        self.model = MLP([opt.domain_dim] + opt.num_layers * [opt.hidden_dim] + [opt.output_channels])


class SineLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SirenModel(EncodedMlpModel):

    def get_encoding(self, x: T) -> T:
        return x

    def mlp_forward(self, x: T):
        return self.model(x)

    @property
    def encoding_dim(self):
        return self.opt.domain_dim

    def __init__(self, opt: ModelParams):
        super(SirenModel, self).__init__(opt)
        self.model = []
        self.model.append(SineLayer(opt.domain_dim, opt.hidden_dim, is_first=True, omega_0=30))
        for i in range(opt.num_layers):
            self.model.append(SineLayer(opt.hidden_dim, opt.hidden_dim, is_first=False, omega_0=30))
        final_linear = nn.Linear(opt.hidden_dim, opt.output_channels)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / opt.hidden_dim) / 30, np.sqrt(6 / opt.hidden_dim) / 30)
            self.model.append(final_linear)
        self.model = nn.Sequential(*self.model)


class PolynomialEncoding(EncodingLayer):

    def expand(self, x: T) -> T:

        @functools.lru_cache
        def expand_(*shape) -> T:
            out = []
            for multipliers in self.kernel:
                out.append(1)
                for i in multipliers:
                    out[-1] = out[-1] * x[:, i]
            out = torch.stack(out, dim=1)
            return out

        return expand_(*x.shape)

    @property
    def output_channels(self) -> int:
        return len(self.kernel)

    @staticmethod
    def get_kerenl(domain_dim: int, power: int) -> List[Tuple[int, ...]]:
        last_added = kernel_str = {(i,) for i in range(domain_dim)}
        for _ in range(power - 1):
            added_ = set()
            for item in last_added:
                for i in range(domain_dim):
                    new_item = (list(item) + [i])
                    new_item.sort()
                    added_.add(tuple(new_item))
            kernel_str = kernel_str.union(added_)
            last_added = added_
        kernel = sorted(list(kernel_str), key=lambda x: len(x))
        kernel = kernel[domain_dim:]
        return kernel

    def forward(self, x: T):
        shape = x.shape[:-1]
        if x.dim() != 2:
            x = x.view(-1, self.domain_dim)
        x = self.expand(x)
        return x.view(*shape, self.output_channels)

    def __init__(self, domain_dim: int, power: int, *_):
        super(PolynomialEncoding, self).__init__()
        self.domain_dim = domain_dim
        self.power: int = power
        self.kernel = self.get_kerenl(domain_dim, power)


class FourierFeatures(EncodingLayer, abc.ABC):

    @property
    def output_channels(self) -> int:
        return self.num_frequencies * 2

    def forward(self, x: T):
        shape = x.shape[:-1]
        if x.dim() != 2:
            x = x.view(-1, self.domain_dim)
        x = x * 2 * np.pi
        out = torch.matmul(x, self.frequencies)
        out = torch.sin(out), torch.cos(out)
        out = torch.stack(out, dim=2).view(*shape, self.output_channels)
        return out

    @abc.abstractmethod
    def init_frequencies(self, std: float) -> T:
        raise NotImplemented

    def __init__(self, domain_dim: int, num_frequencies: int, std: float):
        super(FourierFeatures, self).__init__()
        self.domain_dim = domain_dim
        self.num_frequencies: int = num_frequencies
        frequencies = self.init_frequencies(std)
        self.register_buffer("frequencies", frequencies)


class GaussianRandomFourierFeatures(FourierFeatures):

    def init_frequencies(self, std: float) -> T:
        magnitude = torch.randn(self.num_frequencies) * std
        order = magnitude.abs().argsort(0)
        magnitude = magnitude[order]
        frequencies: T = torch.randn(self.domain_dim, self.num_frequencies)
        frequencies = nnf.normalize(frequencies, p=2, dim=0) * magnitude[None, :]
        return frequencies


class RotatedFourierFeatures(EncodingLayer, abc.ABC):

    @property
    def output_channels(self) -> int:
        return self.num_frequencies * 2

    def forward(self, x: T):
        shape = x.shape[:-1]
        if x.dim() != 2:
            x = x.view(-1, self.domain_dim)
        x = x * 2 * np.pi
        frequencies = nnf.normalize(self.frequencies, p=2, dim=0) * self.magnitudes[None, :]
        out = torch.matmul(x, frequencies)
        out = torch.sin(out), torch.cos(out)
        out = torch.stack(out, dim=2).view(*shape, self.output_channels)
        return out

    def init_frequencies(self, ) -> T:
        frequencies: T = torch.randn(self.domain_dim, self.num_frequencies)
        frequencies = nnf.normalize(frequencies, p=2, dim=0)
        return frequencies

    @abc.abstractmethod
    def init_magnitudes(self, std: float) -> T:
        raise NotImplemented

    def __init__(self, domain_dim: int, num_frequencies: int, std: float):
        super(RotatedFourierFeatures, self).__init__()
        self.domain_dim = domain_dim
        self.num_frequencies: int = num_frequencies
        magnitudes = self.init_magnitudes(std)
        frequencies = nn.Parameter(self.init_frequencies())
        self.register_buffer("magnitudes", magnitudes)
        self.register_parameter("frequencies", frequencies)


class GaussianRotatedFourierFeatures(RotatedFourierFeatures):

    def init_magnitudes(self, std: float) -> T:
        magnitude = torch.randn(self.num_frequencies) * std
        order = magnitude.abs().argsort(0)
        magnitude = magnitude[order]
        frequencies: T = torch.randn(self.domain_dim, self.num_frequencies)
        frequencies = nnf.normalize(frequencies, p=2, dim=0) * magnitude[None, :]
        return magnitude

class UniformFourierFeatures(FourierFeatures):

    def init_frequencies(self, std: float) -> T:
        std = std / np.sqrt(3)
        magnitude = torch.linspace(-std, std, self.num_frequencies) + EPSILON
        order = magnitude.abs().argsort(0)
        magnitude = magnitude[order]
        frequencies: T = torch.randn(self.domain_dim, self.num_frequencies)
        frequencies = nnf.normalize(frequencies, p=2, dim=0) * magnitude[None, :]
        return frequencies


class PositionalEncoding(EncodingLayer):

    @property
    def output_channels(self) -> int:
        return self.num_frequencies * self.domain_dim * 2

    def forward(self, x: T):
        shape = x.shape[:-1]
        if x.dim() != 2:
            x = x.view(-1, self.domain_dim)
        out: T = torch.einsum('f,nd->nfd', self.freqs, x)
        out = torch.cat((torch.cos(out), torch.sin(out)), dim=2).view(-1, self.output_channels - self.domain_dim)
        return out.view(*shape, -1)

    def __init__(self, domain_dim: int, num_frequencies: int):
        super(PositionalEncoding, self).__init__()
        self.domain_dim = domain_dim
        self.num_frequencies = num_frequencies
        freqs = torch.tensor([2. ** i * np.pi for i in range(num_frequencies)])
        self.register_buffer("freqs", freqs)


class RadialBasisEncoding(EncodingLayer):

    @property
    def output_channels(self) -> int:
        return self.num_frequencies

    def forward(self, x: T):
        shape = x.shape[:-1]
        if x.dim() != 2:
            x = x.view(-1, self.domain_dim)
        out = (x[:, None, :] - self.centres[None, :, :]).pow(2).sum(2)
        out = out * self.sigma[None, :] ** 2
        out = torch.exp(-out)
        return out.view(*shape, -1)

    def __init__(self, domain_dim: int, num_frequencies: int, std: int):
        super(RadialBasisEncoding, self).__init__()
        self.domain_dim = domain_dim
        self.num_frequencies = num_frequencies * 2
        centres = torch.rand(self.num_frequencies, domain_dim) * 2 - 1
        sigma = (torch.randn(self.num_frequencies).abs() * std + 1)
        sigma = sigma.sort()[0]
        self.register_buffer("centres", centres)
        self.register_buffer("sigma", sigma)


class RadialBasisGridEncoding(EncodingLayer, abc.ABC):

    @property
    def output_channels(self) -> int:
        return 2 * self.num_frequencies

    def forward(self, x: T):
        shape = x.shape[:-1]
        if x.dim() != 2:
            x = x.view(-1, self.domain_dim)
        x_a = x[:, None, :] + self.offsets[None, :]  # n f d
        x_b = x_a + (1 / self.sigma[None, :, None])  # n f d
        out = torch.stack((x_a, x_b), dim=2)
        out = (out % (2 / self.sigma[None, :, None, None])) * 2 - (2 / self.sigma[None, :, None, None])
        out = out.pow(2).sum(3)  # n f 2
        out = out * self.sigma[None, :, None] ** 2
        out = out.view(-1, self.output_channels)
        out = torch.exp(-out) * 2 - 1
        return out.view(*shape, self.output_channels)

    @abc.abstractmethod
    def init_frequencies(self, std: float) -> T:
        raise NotImplemented

    def __init__(self, domain_dim: int, num_frequencies: int, std: float):
        super(RadialBasisGridEncoding, self).__init__()
        self.domain_dim = domain_dim
        self.num_frequencies = num_frequencies
        sigma = self.init_frequencies(std)
        offsets = (torch.rand(self.num_frequencies, domain_dim) * 2 - 1) % (2 / sigma[:, None])
        sigma = sigma.sort()[0]
        self.register_buffer("offsets", offsets)
        self.register_buffer("sigma", sigma)


class RandomRadialBasisGridEncoding(RadialBasisGridEncoding):

    def init_frequencies(self, std: float) -> T:
        return torch.randn(self.num_frequencies).abs() * std + 1


class UniformRadialBasisGridEncoding(RadialBasisGridEncoding):

    def init_frequencies(self, std: float) -> T:
        frequencies = torch.linspace(0, std * np.sqrt(3), self.num_frequencies)
        frequencies = frequencies + frequencies[1] / 2
        return frequencies


class FFModel(EncodedMlpModel):

    def mlp_forward(self, x: T):
        return self.model(x)

    def get_encoding(self, x: T) -> T:
        return self.encode(x)

    @property
    def encoding_dim(self):
        return self.encode.output_channels

    def __init__(self, opt: ModelParams):
        super(FFModel, self).__init__(opt)
        self.encode = GaussianRandomFourierFeatures(opt.domain_dim, opt.num_frequencies, opt.std)
        self.model = MLP([self.encode.output_channels] + opt.num_layers * [opt.hidden_dim] + [opt.output_channels])


class RFFModel(EncodedMlpModel):

    def mlp_forward(self, x: T):
        return self.model(x)

    def get_encoding(self, x: T) -> T:
        return self.encode(x)

    @property
    def encoding_dim(self):
        return self.encode.output_channels

    def __init__(self, opt: ModelParams):
        super(RFFModel, self).__init__(opt)
        self.encode = GaussianRotatedFourierFeatures(opt.domain_dim, opt.num_frequencies, opt.std)
        self.model = MLP([self.encode.output_channels] + opt.num_layers * [opt.hidden_dim] + [opt.output_channels])


class UFFModel(EncodedMlpModel):

    def mlp_forward(self, x: T):
        return self.model(x)

    def get_encoding(self, x: T) -> T:
        return self.encode(x)

    @property
    def encoding_dim(self):
        return self.encode.output_channels

    def __init__(self, opt: ModelParams):
        super(UFFModel, self).__init__(opt)
        self.encode = UniformFourierFeatures(opt.domain_dim, opt.num_frequencies, opt.std)
        self.model = MLP([self.encode.output_channels] + opt.num_layers * [opt.hidden_dim] + [opt.output_channels])


class PEModel(EncodedMlpModel):

    def mlp_forward(self, x: T):
        return self.model(x)

    def get_encoding(self, x: T) -> T:
        return self.encode(x)

    @property
    def encoding_dim(self):
        return self.encode.output_channels

    def __init__(self, opt: ModelParams):
        super(PEModel, self).__init__(opt)
        self.encode = PositionalEncoding(opt.domain_dim, opt.num_frequencies_pe)
        self.model = MLP([self.encode.output_channels] + opt.num_layers * [opt.hidden_dim] + [opt.output_channels])


class RbfModel(EncodedMlpModel):

    def mlp_forward(self, x: T):
        return self.model(x)

    def get_encoding(self, x: T) -> T:
        return self.encode(x)

    @property
    def encoding_dim(self):
        return self.encode.output_channels

    def __init__(self, opt: ModelParams):
        super(RbfModel, self).__init__(opt)
        self.encode = RadialBasisEncoding(opt.domain_dim, opt.num_frequencies, opt.std_rbf)
        self.model = MLP([self.encode.output_channels] + opt.num_layers * [opt.hidden_dim] + [opt.output_channels])


class RbfgModel(EncodedMlpModel):

    def mlp_forward(self, x: T):
        return self.model(x)

    def get_encoding(self, x: T) -> T:
        return self.encode(x)

    @property
    def encoding_dim(self):
        return self.encode.output_channels

    def __init__(self, opt: ModelParams):
        super(RbfgModel, self).__init__(opt)
        self.encode = UniformRadialBasisGridEncoding(opt.domain_dim, opt.num_frequencies, opt.std_rbf)
        self.model = MLP([self.encode.output_channels] + opt.num_layers * [opt.hidden_dim] + [opt.output_channels])


class ProgressiveModel(EncodedMlpModel, abc.ABC):

    @property
    def is_progressive(self) -> bool:
        return True

    def get_encoding(self, x: T) -> T:
        out = self.encode(x)
        out = torch.cat((x, out), dim=-1)
        return out

    def mlp_forward(self, x: T):
        return self.model(x)

    @property
    def encoding_dim(self):
        return self.encode.output_channels + self.domain_dim

    @functools.lru_cache
    def get_mask(self, alpha: float) -> T:
        mask = torch.zeros(self.encoding_dim)
        if alpha != 0:
            alpha = alpha * self.encode.output_channels + self.domain_dim
            cur_channel = int(alpha // 1)
            mask[:cur_channel] = 1.
            mask[cur_channel] = alpha % 1
        return mask

    def apply_control(self, x: T, *args, **kwargs) -> T:
        if 'override_mask' in kwargs and kwargs['override_mask'] is not None:
            return super(ProgressiveModel, self).apply_control(x, *args, **kwargs)
        if 'alpha' in kwargs and kwargs['alpha'] < 1:
            mask = self.unsqueeze_mask(self.get_mask(kwargs['alpha']), x)
            x = x * mask
            return x
        return x

    def get_encoding_act(self, x, alpha: float):
        weights = list(self.model.parameters())[0]
        act: T = weights.norm(2, dim=0)
        return act

    @staticmethod
    @abc.abstractmethod
    def get_encoding_layer(opt: ModelParams) -> EncodingLayer:
        raise NotImplemented

    def __init__(self, opt: ModelParams):
        super(ProgressiveModel, self).__init__(opt)
        self.encode = self.get_encoding_layer(opt)
        self.model = MLP([self.encoding_dim] + opt.num_layers * [opt.hidden_dim] + [opt.output_channels])


class PFFModel(ProgressiveModel):

    @staticmethod
    def get_encoding_layer(opt: ModelParams):
        return GaussianRandomFourierFeatures(opt.domain_dim, opt.num_frequencies, opt.std)


class PRFFModel(ProgressiveModel):

    @staticmethod
    def get_encoding_layer(opt: ModelParams):
        return GaussianRotatedFourierFeatures(opt.domain_dim, opt.num_frequencies, opt.std)


class PUFFModel(ProgressiveModel):

    @staticmethod
    def get_encoding_layer(opt: ModelParams):
        return UniformFourierFeatures(opt.domain_dim, opt.num_frequencies, opt.std)


class MPFFModel(ProgressiveModel):

    @staticmethod
    def get_encoding_layer(opt: ModelParams):
        return UniformPieceWiseEncoding(opt.domain_dim, opt.num_frequencies, opt.std)


class PPEModel(ProgressiveModel):

    @staticmethod
    def get_encoding_layer(opt: ModelParams):
        return PositionalEncoding(opt.domain_dim, opt.num_frequencies_pe)


class PRBFGModel(ProgressiveModel):

    @staticmethod
    def get_encoding_layer(opt: ModelParams):
        return UniformRadialBasisGridEncoding(opt.domain_dim, opt.num_frequencies, opt.std_rbf)


class PRBFModel(ProgressiveModel):

    @staticmethod
    def get_encoding_layer(opt: ModelParams):
        return RadialBasisEncoding(opt.domain_dim, opt.num_frequencies, opt.std_rbf)


class PieceWiseEncoding(EncodingLayer, abc.ABC):

    @property
    def output_channels(self) -> int:
        return 2 * self.num_frequencies

    def forward(self, x: T):
        shape = x.shape[:-1]
        if x.dim() != 2:
            x = x.view(-1, self.domain_dim)
        out = torch.matmul(x + 1, self.frequencies)
        out = out, out + 1
        out = torch.stack(out, dim=2).view(-1, self.output_channels)
        out = torch.fmod(out, 2) - 1
        mask = out.lt(0)
        out[mask] = 2 * out[mask] + 1
        out[~mask] = 1 - 2 * out[~mask]
        out = out.view(*shape, self.output_channels)
        return out

    @abc.abstractmethod
    def init_frequencies(self, std: float) -> T:
        raise NotImplemented

    def __init__(self, domain_dim: int, num_frequencies: int, std: float):
        super(PieceWiseEncoding, self).__init__()
        self.domain_dim = domain_dim
        self.num_frequencies: int = num_frequencies
        frequencies = self.init_frequencies(std)
        self.register_buffer("frequencies", frequencies)


class GaussianRandomPieceWiseEncoding(PieceWiseEncoding):

    def init_frequencies(self, std: float) -> T:
        frequencies: T = torch.randn(self.domain_dim, self.num_frequencies) * std / (2 * np.pi)
        frequencies = frequencies.abs()
        order = frequencies.norm(2, 0).argsort()
        frequencies = frequencies[:, order]
        return frequencies


class UniformPieceWiseEncoding(PieceWiseEncoding):

    def init_frequencies(self, std: float) -> T:
        b = std * np.sqrt(12) / (2 * np.pi)
        magnitude = torch.linspace(0, b, self.num_frequencies)
        magnitude = magnitude + magnitude[1] / 2
        frequencies: T = torch.randn(self.domain_dim, self.num_frequencies).abs()
        frequencies = nnf.normalize(frequencies, p=2, dim=0) * magnitude[None, :]
        return frequencies


model_dict = {'siren':SirenModel, 'FFN':FFModel, 'UFF':UFFModel, 'PFF':PFFModel,
              'RBF':RbfModel, 'PRBF':PRBFModel, 'RBFG':RbfgModel, 'PRBFG':PRBFGModel,
              'PE':PEModel, 'PPE':PPEModel, 'RFF':RFFModel, 'PRFF':PRFFModel,
              'PUFF':PUFFModel}

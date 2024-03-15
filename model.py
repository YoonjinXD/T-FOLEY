import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn


# --- Helper Utils ---
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
    
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 64):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, l = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1) # (B, 256, L) tuple 3개
        q, k, v = map(lambda t: rearrange(t, 'b (h c) l -> b h c l', h = self.heads), qkv) # (B, 4, 64, L)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k) # (B, 4, L, L)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h l d -> b (h d) l', l=l)
        return self.to_out(out)
    

# --- classifier free guidance functions ---
def uniform(shape, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob


# --- Random Fourier Feature MLP ---
class RFF_MLP_Block(nn.Module):
    def __init__(self, time_emb_dim=512):
        super().__init__()
        self.RFF_freq = nn.Parameter(
            16 * torch.randn([1, 32]), requires_grad=False)
        self.MLP = nn.ModuleList([
            nn.Linear(64, 128),
            nn.Linear(128, 256),
            nn.Linear(256, time_emb_dim),
        ])

    def forward(self, sigma):
        """
        Arguments:
          sigma:
              (shape: [B, 1], dtype: float32)

        Returns:
          x: embedding of sigma
              (shape: [B, 512], dtype: float32)
        """
        x = self._build_RFF_embedding(sigma)
        for layer in self.MLP:
            x = F.relu(layer(x))
        return x

    def _build_RFF_embedding(self, sigma):
        """
        Arguments:
          sigma:
              (shape: [B, 1], dtype: float32)
        Returns:
          table:
              (shape: [B, 64], dtype: float32)
        """
        freqs = self.RFF_freq
        freqs = freqs.to(device=torch.device("cuda"))
        table = 2 * np.pi * sigma * freqs
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table

# --- Conditioning Modules ---
class Film(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_layer = nn.Linear(input_dim, 2 * output_dim)

    def forward(self, sigma_encoding):
        sigma_encoding = self.output_layer(sigma_encoding)
        sigma_encoding = sigma_encoding.unsqueeze(-1)
        gamma, beta = torch.chunk(sigma_encoding, 2, dim=1)
        return gamma, beta
    
class Film_withConds(nn.Module):
    def __init__(self, output_dim, time_emb_dim=512, classes_emb_dim=512):
        super().__init__()
        self.layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(int(time_emb_dim) + int(classes_emb_dim), output_dim * 2)
        ) if exists(time_emb_dim) or exists(classes_emb_dim) else None

    def forward(self, time_emb = None, class_emb = None):
        cond_emb = tuple(filter(exists, (time_emb, class_emb)))
        cond_emb = torch.cat(cond_emb, dim = -1)
        cond_emb = self.layer(cond_emb)
        cond_emb = cond_emb.unsqueeze(-1)
        gamma, beta = cond_emb.chunk(2, dim = 1)
        return gamma, beta
    
class TFilm(nn.Module):
    """
    Arguments:
          block_num: range(1, 88200), dtype: int
          output_dim: dtype: int
        Returns:
          norm_list = [(gamma_b1, beta_b1), (gamma_b2, beta_b2), ..., (gamma_bn, beta_bn)], shape: (block_num, 2), dtype: list(tuples)
    """
    def __init__(self, block_num, input_dim, output_dim):
        super().__init__()
        self.block_num = block_num
        self.num_layers = 8 # default = 2
        self.lstm = nn.LSTM(input_dim, output_dim, num_layers=self.num_layers, batch_first=True, bidirectional=False)
        self.output_dim = output_dim
        self.layer = nn.Linear(output_dim, output_dim*2)

    def forward(self, x):
        block_size = x.shape[-1] // self.block_num
        
        pooling = nn.MaxPool1d(block_size, stride=block_size).to(x.device)
        x = pooling(x.unsqueeze(1)).squeeze(1)
        h0 = torch.randn(self.num_layers, x.shape[0], self.output_dim, device=x.device)
        c0 = torch.randn(self.num_layers, x.shape[0], self.output_dim, device=x.device)
        x, _ = self.lstm(x.unsqueeze(-1), (h0, c0))
        x = self.layer(x)
        x = x.permute(0, 2, 1)
        gamma, beta = x.chunk(2, dim=1)
        
        return gamma, beta
    
class BFilm(nn.Module):
    """
    Arguments:
          block_num: range(1, 88200), dtype: int
          output_dim: dtype: int
        Returns:
          norm_list = [(gamma_b1, beta_b1), (gamma_b2, beta_b2), ..., (gamma_bn, beta_bn)], shape: (block_num, 2), dtype: list(tuples)
    """
    def __init__(self, block_num, input_dim, output_dim):
        super().__init__()
        self.block_num = block_num
        self.layer = nn.Linear(input_dim, output_dim*2)

    def forward(self, x):
        block_size = x.shape[-1] // self.block_num
        
        device = x.device
        pooling = nn.MaxPool1d(block_size, stride=block_size)
        x = x.unsqueeze(1)
        pooled_x = pooling(x).to(device)
        
        if pooled_x.shape[-1] != self.block_num:
            block_size += 1
            padding = (block_size*self.block_num - x.shape[-1]+1)//2
            x = F.interpolate(x, size=x.shape[-1]+2*padding, mode='nearest')
            
            pooling = nn.MaxPool1d(block_size, stride=block_size)
            pooled_x = pooling(x).to(device)
        
        pooled_x = pooled_x.squeeze(1)
        x = self.layer(pooled_x.unsqueeze(-1))
        x = x.permute(0, 2, 1)
        gamma, beta = x.chunk(2, dim = 1)
        
        return gamma, beta

# --- Down&Up-sampling blocks ---
class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        nn.init.zeros_(self.bias)
        

# --- Block Modules ---
class FilmConvBlock(nn.Module):
    '''FiLM + Activation + Conv'''
    def __init__(self, in_channel, out_channel, factor=1):
        super().__init__()
        self.proj = Conv1d(in_channel, out_channel, 3, dilation=1, padding=1)
        # self.norm = nn.GroupNorm(2 if factor < 0 else 1, out_channel)
        self.norm = nn.GroupNorm(8, out_channel)
        self.act = nn.SiLU()
        
        self.factor = factor
        if factor < 0:
            self.conv = nn.ConvTranspose1d(out_channel, out_channel, 3, stride=abs(factor), padding=1, output_padding=abs(factor)-1)
        else:
            self.conv = Conv1d(out_channel, out_channel, 3, stride=factor, padding=1)

    def forward(self, x, gamma, beta):
        x = self.proj(x)
        x = gamma * x + beta
        
        x = self.norm(x)
        x = self.act(x)
        
        x = self.conv(x)
        return x
    
class TFilmConvBlock(nn.Module):
    '''FiLM + Activation + Conv'''
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.proj = Conv1d(in_channel, out_channel, 3, padding=1)
        self.act = nn.SiLU()
        self.conv = Conv1d(out_channel, out_channel, 3, padding=1)

    def forward(self, x, gamma, beta):
        x = self.proj(x)
        
        chunks = list(x.chunk(gamma.shape[-1], dim=-1))
        for i, chunk in enumerate(chunks):
            g = gamma[:,:,i].unsqueeze(-1)
            b = beta[:,:,i].unsqueeze(-1)
            chunks[i] = chunk * g + b
        
        x = torch.cat(chunks, dim=-1)
        x = self.act(x)
        
        x = self.conv(x)
        return x
    
class BFilmConvBlock(nn.Module):
    '''FiLM + Activation + Conv'''
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.proj = Conv1d(in_channel, out_channel, 3, padding=1)
        self.act = nn.SiLU()
        self.conv = Conv1d(out_channel, out_channel, 3, padding=1)

    def forward(self, x, gamma, beta):
        x = self.proj(x)
        
        chunks = list(x.chunk(gamma.shape[-1], dim=-1))
        for i, chunk in enumerate(chunks):
            g = gamma[:,:,i].unsqueeze(-1)
            b = beta[:,:,i].unsqueeze(-1)
            chunks[i] = chunk * g + b
        
        x = torch.cat(chunks, dim=-1)
        x = self.act(x)
        
        x = self.conv(x)
        return x

class GBlock(nn.Module):
    def __init__(self, in_channel, out_channel, factor, block_num, film_type, event_dim):
        super().__init__()
        self.factor = factor
        if self.factor < 0: in_channel = in_channel * 2
        
        self.residual_dense1 = Conv1d(in_channel, out_channel, 1)
        self.factor_convs = nn.ModuleList([
            FilmConvBlock(in_channel, in_channel, factor),
            FilmConvBlock(in_channel, out_channel),
        ])
        self.factor_films = nn.ModuleList([
            Film_withConds(in_channel),
            Film_withConds(out_channel),
        ])
        
        self.residual_dense2 = Conv1d(out_channel, out_channel, 1)
        assert film_type in [None, 'film', 'temporal', 'block']
        if film_type == None:
            self.t_convs = None
            self.t_films = None
        elif film_type == 'film':
            self.t_convs = nn.ModuleList([FilmConvBlock(out_channel, out_channel), FilmConvBlock(out_channel, out_channel)])
            self.t_films = nn.ModuleList([Film(event_dim, out_channel), Film(event_dim, out_channel)])
        elif film_type == 'temporal':
            self.t_convs = nn.ModuleList([TFilmConvBlock(out_channel, out_channel), TFilmConvBlock(out_channel, out_channel)])
            self.t_films = nn.ModuleList([TFilm(block_num, 1, out_channel), TFilm(block_num, 1, out_channel)])
        elif film_type == 'block':
            self.t_convs = nn.ModuleList([BFilmConvBlock(out_channel, out_channel), BFilmConvBlock(out_channel, out_channel)])
            self.t_films = nn.ModuleList([BFilm(block_num, 1, out_channel), BFilm(block_num, 1, out_channel)])

    def forward(self, x, sigma, c, a):
        size = self._output_size(x.shape[-1])
        
        residual = F.interpolate(x, size=size)
        residual = self.residual_dense1(residual)
        for film, layer in zip(self.factor_films, self.factor_convs):
            gamma, beta = film(sigma, c)
            x = layer(x, gamma, beta)
        x = x + residual
        
        if self.t_films != None:
            residual = F.interpolate(x, size=size)
            residual = self.residual_dense2(residual)
            for t_film, layer in zip(self.t_films, self.t_convs):
                gamma, beta = t_film(a)
                x = layer(x, gamma, beta)
            x = x + residual
        return x
    
    def _output_size(self, input_size):
        return input_size * abs(self.factor) if self.factor < 0 else input_size // self.factor


# --- U-Net ---
class UNet(nn.Module):
    def __init__(self, num_classes, params):
        super().__init__()
        print("Model initializing... This can take a few minutes.")

        # Hyperparameter Settings
        sequential = params['sequential']
        assert sequential in ['lstm', 'attn', None], "Choose sequential between \'lstm\' or \'attn\', None."

        dims = params['dims']
        factors =params['factors']
        assert len(dims)-1 == len(factors)
        
        block_nums = params['block_nums']
        time_emb_dim = params['time_emb_dim']
        class_emb_dim = params['class_emb_dim']
        event_dim = params['event_dims'][params['event_type']]
        
        cond_drop_prob = params['cond_prob']
        film_type = params['film_type']
        
        # Pre-conv/emb Layers
        self.conv_1 = Conv1d(1, dims[0], 5, padding=2)
        self.embedding = RFF_MLP_Block(time_emb_dim)
        
        # Up/DownSample Block Layers
        DBlock_list = []
        for in_dim, out_dim, factor, block_num in zip(dims[:-1], dims[1:], factors, block_nums):
            DBlock_list.append(GBlock(in_dim, out_dim, factor, block_num, film_type, event_dim))
        self.downsample = nn.ModuleList(DBlock_list)
        
        UBlock_list = []
        for in_dim, out_dim, factor, block_num in zip(dims[:0:-1], dims[-2::-1], factors[::-1], block_nums[::-1]):
            UBlock_list.append(GBlock(in_dim, out_dim, -1*factor, block_num, film_type, event_dim))
        self.upsample = nn.ModuleList(UBlock_list)
        self.last_conv = Conv1d(dims[0], 1, 3, padding=1)

        # Bottleneck layer
        self.sequential = sequential
        if sequential:
            self.mid_dim = params['mid_dim']
            if sequential == 'lstm':
                self.lstm = nn.LSTM(self.mid_dim, self.mid_dim, num_layers=2, batch_first=True, bidirectional=True)
                self.lstm_mlp = nn.Sequential(
                    nn.Linear(self.mid_dim*2, self.mid_dim),
                    nn.SiLU(),
                    nn.Linear(self.mid_dim, self.mid_dim)
                )
            
            if sequential == 'attn':
                self.mid_attn = Residual(PreNorm(self.mid_dim, Attention(self.mid_dim)))
                
        # Classifier-free guidance
        self.cond_drop_prob = cond_drop_prob
        
        self.classes_emb = nn.Embedding(num_classes, class_emb_dim)
        self.null_classes_emb = nn.Parameter(torch.randn(class_emb_dim))
        self.null_event_emb = nn.Parameter(torch.randn(event_dim))
        
        classes_dim = class_emb_dim * 4
        self.classes_mlp = nn.Sequential(
            nn.Linear(class_emb_dim, classes_dim),
            nn.SiLU(),
            nn.Linear(classes_dim, class_emb_dim)
        ) 

        print("Model successfully initialized!")

    def forward(self, audio, sigma, classes, events, cond_drop_prob=None):
        batch, device = audio.shape[0], audio.device
        x = audio.unsqueeze(1)
        x = self.conv_1(x)
        downsampled = []
        sigma_encoding = self.embedding(sigma)
        
        # Prepare Conditions(class, event)
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        classes_emb = self.classes_emb(classes)
        if cond_drop_prob[0] > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob[0], device = device)
            null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = batch)

            classes_emb = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                classes_emb,
                null_classes_emb
            )
        c = self.classes_mlp(classes_emb)
        
        if cond_drop_prob[1] > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob[1], device = device)
            null_event= repeat(self.null_event_emb, 'd -> b d', b = batch)
            
            events = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                events,
                null_event
            ) if events != None else null_event
            
        # Downsample
        for layer in self.downsample:
            x = layer(x, sigma_encoding, c, events)
            downsampled.append(x)
            
        # Bottleneck
        if self.sequential:
            if self.sequential == 'lstm':
                h0 = torch.randn(4, batch, self.mid_dim, device=device)
                c0 = torch.randn(4, batch, self.mid_dim, device=device)
                x = x.permute(0, 2, 1)
                x, _ = self.lstm(x, (h0, c0))
                x = self.lstm_mlp(x)
                x = x.permute(0, 2, 1)
            
            if self.sequential == 'attn':
                x = self.mid_attn(x)
        
            x = x + downsampled[-1] # residual connection
        
        # Upsample
        for layer, x_dblock in zip(self.upsample, reversed(downsampled)):
            x = torch.cat([x, x_dblock], dim=1)
            x = layer(x, sigma_encoding, c, events)
            
        x = self.last_conv(x)
        x = x.squeeze(1)
        return x
    
    def forward_with_cond_scale(self, audio, sigma, classes, event, cond_scale=1.):
        cond_score = self.forward(audio, sigma, classes, event, cond_drop_prob=[0.0, 0.0])
        if cond_scale == 1: return cond_score
        uncond_score = self.forward(audio, sigma, classes, event, cond_drop_prob=[1.0, 1.0])
        return uncond_score + (cond_score - uncond_score) * cond_scale

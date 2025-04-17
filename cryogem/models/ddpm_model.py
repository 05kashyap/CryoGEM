import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_model import BaseModel
from . import networks
import logging
from einops import rearrange 

logger = logging.getLogger(__name__)

class DDPM(nn.Module):
    def __init__(self, denoise_fn, beta_schedule='linear', timesteps=1000,
                 linear_start=1e-4, linear_end=2e-2):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.timesteps = timesteps
        
        # Define beta schedule
        if beta_schedule == 'linear':
            betas = torch.linspace(linear_start, linear_end, timesteps)
        elif beta_schedule == 'cosine':
            # Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
            steps = timesteps + 1
            s = 0.008
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0, 0.999)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1. - betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.alphas_cumprod))
        
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start, t, noise=None):
        """Calculate loss for denoising using combined L1 and MSE losses"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.denoise_fn(x_noisy, t)
        
        # Calculate MSE loss
        mse_loss = F.mse_loss(predicted_noise, noise)
        
        # Calculate L1 (MAE) loss
        l1_loss = F.l1_loss(predicted_noise, noise)
        
        # Combined loss (equal weighting)
        loss = mse_loss + l1_loss
        
        return loss, mse_loss, l1_loss
    
    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        """Reverse diffusion process - single step"""
        betas_t = self.betas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recip_alphas_t = torch.rsqrt(self.alphas[t]).reshape(-1, 1, 1, 1)
        
        # Equation 11 in the paper
        # Use predicted x_0 for sampling
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.denoise_fn(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.betas[t].reshape(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
        
    @torch.no_grad()
    def p_sample_loop(self, shape, device):
        """Full reverse diffusion process for sampling new images"""
        b = shape[0]
        img = torch.randn(shape, device=device)
        
        for i in reversed(range(0, self.timesteps)):
            img = self.p_sample(
                x=img,
                t=torch.full((b,), i, device=device, dtype=torch.long),
                t_index=i
            )
        return img

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class LSA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()

        mask = torch.eye(dots.shape[-1], device=dots.device, dtype=torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_ratio, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, LSA(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_ratio * dim, dropout=dropout)))
            ]))

    def forward(self, x):
        shape = x.shape
        x = x.flatten(2).transpose(1, 2)

        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        x = x.permute(0, 2, 1).view(shape[0], shape[1], shape[2], shape[3])
        return x


class To_Image(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Patch_Embedding(nn.Module):
    def __init__(self, dim, patch_size, img_channels=3, emb_dim=256):
        super().__init__()
        self.embed = nn.Conv2d(img_channels, dim, kernel_size=patch_size, stride=patch_size)

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                dim
            ),
        )

    def forward(self, x, t):
        x = self.embed(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(4),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x=None, t=None, last=False):
        x = self.up(x) if not last else F.interpolate(x, size=x.shape[2]+16, mode="bilinear", align_corners=True)
        if skip_x is not None:
            x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class EnhancedUNet(nn.Module):
    def __init__(self, input_channels=1, out_channels=1, time_dim=256, dim=1024, depth=1, heads=4, 
                 dim_head=64, mlp_ratio=4, drop_rate=0.):
        super(EnhancedUNet, self).__init__()
        self.time_dim = time_dim
        emb_dim = time_dim
        self.input_channels = input_channels
        # Calculate dimensions for the network scales
        dims = (dim // 32, dim // 16, dim // 8, dim // 4, dim // 2, dim)

        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, dims[0], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, dims[0]),
            nn.GELU(),
        )

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.transformer_blocks = nn.ModuleList()
        self.bottleneck_blocks = nn.ModuleList()

        self.down_blocks.append(Down(dims[0], dims[2], emb_dim))
        self.down_blocks.append(Down(dims[2], dims[3], emb_dim))

        self.transformer_blocks.append(Transformer(dims[2], depth, heads, dim_head, mlp_ratio, drop_rate))
        self.transformer_blocks.append(Transformer(dims[3], depth, heads, dim_head, mlp_ratio, drop_rate))

        self.bottleneck_blocks.append(DoubleConv(dims[3], dims[4]))
        self.bottleneck_blocks.append(DoubleConv(dims[4], dims[4]))
        self.bottleneck_blocks.append(DoubleConv(dims[4], dims[3]))

        self.up_blocks.append(Up(dims[3], dims[2], emb_dim))
        self.up_blocks.append(Up(dims[3], dims[1], emb_dim))
        self.up_blocks.append(Up(dims[1], dims[0], emb_dim))
        self.up_blocks.append(Up(dims[1], dims[0], emb_dim))

        self.last = To_Image(dims[0], out_channels)

    def pos_encoding(self, t, channels):
        """Improved timestep embedding with sinusoidal positions"""
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float().to(t.device) / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        """Forward pass with timestep embedding"""
        # Convert integer timesteps to embeddings
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.initial(x)

        x2 = self.down_blocks[0](x1, t)
        x2 = self.transformer_blocks[0](x2)

        x3 = self.down_blocks[1](x2, t)
        x3 = self.transformer_blocks[1](x3)

        x3 = self.bottleneck_blocks[0](x3)
        x3 = self.bottleneck_blocks[1](x3)
        x3 = self.bottleneck_blocks[2](x3)

        x = self.up_blocks[0](x3, None, t)
        x = self.up_blocks[1](x, x2, t)
        x = self.up_blocks[2](x, None, t)
        x = self.up_blocks[3](x, x1, t)

        x = self.last(x)

        return x

class UNet(nn.Module):
    def __init__(self, input_channels=1, model_channels=128, out_channels=1, 
                 num_res_blocks=2, attention_resolutions=[8, 16], 
                 dropout=0.1, channel_mult=(1, 2, 4, 8), num_heads=4):
        super().__init__()
        
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels)
        )
        
        # Define the input block - make this a module that handles timestep
        self.input_conv = TimestepEmbedSequential(
            nn.Conv2d(input_channels, model_channels, kernel_size=3, padding=1)
        )
        
        # Define encoder blocks (downsampling)
        self.input_blocks = nn.ModuleList([self.input_conv])
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, model_channels, dropout, out_channels=mult * model_channels)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(ch, num_heads=num_heads)
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch)))
                input_block_chans.append(ch)
                ds *= 2
        
        # Define middle blocks
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, model_channels, dropout),
            AttentionBlock(ch, num_heads=num_heads),
            ResBlock(ch, model_channels, dropout)
        )
        
        # Define output blocks (upsampling)
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        model_channels,
                        dropout,
                        out_channels=model_channels * mult
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(ch, num_heads=num_heads)
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
        
        # Final output layer
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x, timesteps):
        # Timestep embedding
        temb = get_timestep_embedding(timesteps, self.time_embed[0].in_features)
        temb = self.time_embed(temb)
        
        # Downsampling
        h = x
        hs = []
        for module in self.input_blocks:
            h = module(h, temb)
            hs.append(h)
        
        # Middle
        h = self.middle_block(h, temb)
        
        # Upsampling
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, temb)
        
        # Final output - doesn't require time embedding
        return self.out(h)


# Add new TimestepEmbedSequential class
class TimestepEmbedSequential(nn.Module):
    """
    A sequential module that passes timestep embeddings to the children that
    accept them as an extra input.
    """
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, emb=None):
        for layer in self.layers:
            if isinstance(layer, (ResBlock, AttentionBlock, Upsample, Downsample)):
                # These layers accept timestep embedding
                x = layer(x, emb)
            else:
                # Standard PyTorch layers don't
                x = layer(x)
        return x


def get_timestep_embedding(timesteps, embedding_dim, max_period=10000):
    """Calculate sinusoidal time embedding"""
    half = embedding_dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(max_period, dtype=torch.float32)) *
        torch.arange(start=0, end=half, dtype=torch.float32) /
        half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad if dimension is odd
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ResBlock(nn.Module):
    """Residual block for U-Net"""
    def __init__(self, in_channels, temb_channels, dropout, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x, temb):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        
        # Add timestep embedding
        h = h + self.temb_proj(temb)[:, :, None, None]
        
        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention block for U-Net"""
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x, temb=None):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # B x heads x (C/heads) x (H*W)
        
        # Compute attention
        scale = (C // self.num_heads) ** -0.5
        weight = (q.transpose(-2, -1) @ k) * scale  # B x heads x (H*W) x (H*W)
        weight = torch.softmax(weight, dim=-1)
        
        # Apply attention
        h = (v @ weight.transpose(-2, -1)).reshape(B, C, H, W)
        h = self.proj(h)
        
        return h + x
    

class Downsample(nn.Module):
    """Downsampling layer using strided convolution"""
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x, temb=None):
        # temb is ignored
        return self.op(x)
    

class Upsample(nn.Module):
    """Upsampling layer using nearest neighbor interpolation + conv"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
    def forward(self, x, temb=None):
        # temb is ignored
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class DDPMModel(BaseModel):
    """DDPM for cryo-EM images"""
    
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='cryogem')
        if is_train:
            parser.add_argument('--timesteps', type=int, default=1000, help='number of diffusion steps')
            parser.add_argument('--beta_schedule', type=str, default='linear', help='noise schedule [linear|cosine]')
            parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
            parser.add_argument('--sample_interval', type=int, default=1000, help='how many iterations to sample')
            # Add new architecture parameters
            parser.add_argument('--dim', type=int, default=4096, help='base dimension for enhanced unet')
            parser.add_argument('--time_dim', type=int, default=1024, help='timestep embedding dimension')
            parser.add_argument('--depth', type=int, default=6, help='transformer depth')
            parser.add_argument('--heads', type=int, default=16, help='transformer heads')
            parser.add_argument('--dim_head', type=int, default=128, help='dimension per head')
            parser.add_argument('--mlp_ratio', type=int, default=8, help='mlp expansion ratio')
            parser.add_argument('--drop_rate', type=float, default=0.1, help='dropout rate')
            
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        
        # Define model components
        self.model_names = ['Diffusion']
        self.timesteps = opt.timesteps if hasattr(opt, 'timesteps') else 1000
        
        # Get architecture parameters (with defaults if not provided)
        dim = getattr(opt, 'dim', 2048)           # Increased from 1024
        time_dim = getattr(opt, 'time_dim', 512)  # Increased from 256
        depth = getattr(opt, 'depth', 2)          # Increased from 1
        heads = getattr(opt, 'heads', 8)          # Increased from 4
        dim_head = getattr(opt, 'dim_head', 96)   # Increased from 64
        mlp_ratio = getattr(opt, 'mlp_ratio', 4)  # Keep same
        drop_rate = getattr(opt, 'drop_rate', 0.1)  # Keep same
        
        
        # Create EnhancedUNet for noise prediction and DDPM for diffusion process
        self.unet = EnhancedUNet(
            input_channels=1,
            out_channels=1,
            time_dim=time_dim,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate
        )
        
        self.netDiffusion = DDPM(
            denoise_fn=self.unet,
            beta_schedule=opt.beta_schedule if hasattr(opt, 'beta_schedule') else 'linear',
            timesteps=self.timesteps
        )
        
        self.netDiffusion.to(self.device)
        
        # Define losses and optimizers
        if self.isTrain:
            self.criterionL2 = torch.nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.netDiffusion.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            
        # Define visualization names
        self.visual_names = ['real_A', 'generated']
        self.generated = None
        
    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.image_paths = input['A_paths']
        
    def forward(self):
        """Generate samples from random noise (used during inference)"""
        # Sample from the diffusion model
        if self.generated is None or self.generated.shape != self.real_A.shape:
            with torch.no_grad():
                self.generated = self.netDiffusion.p_sample_loop(shape=self.real_A.shape, device=self.device)
            
    def backward(self):
        """Calculate loss and backpropagate"""
        # Sample timesteps uniformly
        t = torch.randint(0, self.timesteps, (self.real_A.size(0),), device=self.device).long()
        
        # Calculate diffusion losses
        self.loss, self.mse_loss, self.l1_loss = self.netDiffusion.p_losses(self.real_A, t)
        self.loss.backward()
        
    def optimize_parameters(self):
        """Update model weights"""
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
    
    def get_current_losses(self):
        """Return current losses"""
        return {
            'DDPM': self.loss.item(),
            'MSE': self.mse_loss.item(),
            'L1': self.l1_loss.item()
        }
    
    def generate_samples(self, n_samples=16, size=256):
        """Generate new samples from the diffusion model"""
        shape = (n_samples, 1, size, size)
        with torch.no_grad():
            samples = self.netDiffusion.p_sample_loop(shape=shape, device=self.device)
        return samples
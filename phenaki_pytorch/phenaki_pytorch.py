import copy
import math
from functools import partial, wraps

from typing import Optional, List, Union
from typeguard import typechecked

import torch
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
from torch import nn, einsum
import torchvision

from einops import rearrange, repeat, unpack, pack
from einops.layers.torch import Rearrange

from vector_quantize_pytorch import VectorQuantize

from phenaki_pytorch.t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME

# helpers

def exists(val):
    return val is not None

def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else (val,) * length

# tensor helpers

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()

# decorators

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def remove_vgg(fn):
    @wraps(fn)
    def inner(self, *args, **kwargs):
        has_vgg = hasattr(self, 'vgg')
        if has_vgg:
            vgg = self.vgg
            delattr(self, 'vgg')

        out = fn(self, *args, **kwargs)

        if has_vgg:
            self.vgg = vgg

        return out
    return inner

# classifier free guidance functions

def uniform(shape, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# gan losses

def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()

def hinge_gen_loss(fake):
    return -fake.mean()

def bce_discr_loss(fake, real):
    return (-log(1 - torch.sigmoid(fake)) - log(torch.sigmoid(real))).mean()

def bce_gen_loss(fake):
    return -log(torch.sigmoid(fake)).mean()

def grad_layer_wrt_loss(loss, layer):
    return torch_grad(
        outputs = loss,
        inputs = layer,
        grad_outputs = torch.ones_like(loss),
        retain_graph = True
    )[0].detach()

# tensor helper functions

def log(t, eps = 1e-10):
    return torch.log(t + eps)

def gradient_penalty(images, output, weight = 10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs = output, inputs = images,
                           grad_outputs = torch.ones(output.size(), device = images.device),
                           create_graph = True, retain_graph = True, only_inputs = True)[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((gradients.norm(2, dim = 1) - 1) ** 2).mean()

def l2norm(t):
    return F.normalize(t, dim = -1)

def leaky_relu(p = 0.1):
    return nn.LeakyReLU(p)

def safe_div(numer, denom, eps = 1e-8):
    return numer / (denom + eps)

# sampling helpers

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# alibi positional bias for extrapolation

class AlibiPositionalBias(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)

    def get_bias(self, i, j, device):
        i_arange = torch.arange(j - i, j, device = device)
        j_arange = torch.arange(j, device = device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') - rearrange(i_arange, 'i -> 1 i 1'))
        return bias

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    def forward(self, sim):
        h, i, j, device = *sim.shape[-3:], sim.device

        if exists(self.bias) and self.bias.shape[-1] >= j:
            return self.bias[..., :i, :j]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unalibied))
        self.register_buffer('bias', bias, persistent = False)

        return self.bias

class ContinuousPositionBias(nn.Module):
    """ from https://arxiv.org/abs/2111.09883 """

    def __init__(
        self,
        *,
        dim,
        heads,
        num_dims = 2, # 2 for images, 3 for video
        layers = 2,
        log_dist = True,
        cache_rel_pos = False
    ):
        super().__init__()
        self.num_dims = num_dims
        self.log_dist = log_dist

        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(self.num_dims, dim), leaky_relu()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), leaky_relu()))

        self.net.append(nn.Linear(dim, heads))

        self.cache_rel_pos = cache_rel_pos
        self.register_buffer('rel_pos', None, persistent = False)

    def forward(self, *dimensions, device = torch.device('cpu')):

        if not exists(self.rel_pos) or not self.cache_rel_pos:
            positions = [torch.arange(d, device = device) for d in dimensions]
            grid = torch.stack(torch.meshgrid(*positions, indexing = 'ij'))
            grid = rearrange(grid, 'c ... -> (...) c')
            rel_pos = rearrange(grid, 'i c -> i 1 c') - rearrange(grid, 'j c -> 1 j c')

            if self.log_dist:
                rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)

            self.register_buffer('rel_pos', rel_pos, persistent = False)

        rel_pos = self.rel_pos.float()

        for layer in self.net:
            rel_pos = layer(rel_pos)

        return rearrange(rel_pos, 'i j h -> h i j')

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

def FeedForward(dim, mult = 4):
    inner_dim = int(mult * (2 / 3) * dim)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        GEGLU(),
        nn.Linear(inner_dim, dim, bias = False)
    )

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_context = None,
        dim_head = 64,
        heads = 8,
        causal = False,
        num_null_kv = 0,
        norm_context = True
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        dim_context = default(dim_context, dim)

        if causal:
            self.rel_pos_bias = AlibiPositionalBias(heads = heads)

        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim_context) if norm_context else nn.Identity()

        self.num_null_kv = num_null_kv
        self.null_kv = nn.Parameter(torch.randn(heads, 2 * num_null_kv, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        mask = None,
        context = None,
        attn_bias = None
    ):
        batch, device, dtype = x.shape[0], x.device, x.dtype

        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)

        x = self.norm(x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        q = q * self.scale

        nk, nv = repeat(self.null_kv, 'h (n r) d -> b h n r d', b = batch, r = 2).unbind(dim = -2)

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        i, j = sim.shape[-2:]

        if exists(attn_bias):
            attn_bias = F.pad(attn_bias, (self.num_null_kv, 0), value = 0.)
            sim = sim + attn_bias

        if exists(mask):
            mask = F.pad(mask, (self.num_null_kv, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            sim = sim + self.rel_pos_bias(sim)

            causal_mask = torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_context = None,
        causal = False,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        attn_num_null_kv = 2,
        has_cross_attn = False
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, causal = causal),
                Attention(dim = dim, dim_head = dim_head, dim_context = dim_context, heads = heads, causal = False, num_null_kv = attn_num_null_kv) if has_cross_attn else None,
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm_out = nn.LayerNorm(dim)

    def forward(
        self,
        x,
        attn_bias = None,
        context = None,
        self_attn_mask = None,
        cross_attn_context_mask = None
    ):

        for self_attn, cross_attn, ff in self.layers:
            x = self_attn(x, attn_bias = attn_bias, mask = self_attn_mask) + x

            if exists(cross_attn) and exists(context):
                x = cross_attn(x, context = context, mask = cross_attn_context_mask) + x

            x = ff(x) + x

        return self.norm_out(x)

# resnet blocks

class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        groups = 8
    ):
        super().__init__()
        self.groupnorm = nn.GroupNorm(groups, dim)
        self.activation = leaky_relu()
        self.project = nn.Conv2d(dim, dim_out, 3, padding = 1)

    def forward(self, x, scale_shift = None):
        x = self.groupnorm(x)
        x = self.activation(x)
        return self.project(x)

class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        *,
        groups = 8
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.block = Block(dim, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block(x)
        return h + self.res_conv(x)

# discriminator

class DiscriminatorBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        filters,
        downsample = True
    ):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride = (2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu()
        )

        self.downsample = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
            nn.Conv2d(filters * 4, filters, 1)
        ) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)

        if exists(self.downsample):
            x = self.downsample(x)

        x = (x + res) * (1 / math.sqrt(2))
        return x

class Discriminator(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        channels = 3,
        attn_layers = [],
        max_dim = 512
    ):
        super().__init__()
        num_layers = int(math.log2(min(image_size)) - 1)
        blocks = []

        layer_dims = [channels] + [(dim * 4) * (2 ** i) for i in range(num_layers + 1)]
        layer_dims = [min(layer_dim, max_dim) for layer_dim in layer_dims]
        layer_dims_in_out = tuple(zip(layer_dims[:-1], layer_dims[1:]))

        blocks = []

        for ind, (in_chan, out_chan) in enumerate(layer_dims_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(layer_dims_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample = is_not_last)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

        dim_last = layer_dims[-1]
        latent_dim = 2 * 2 * dim_last

        self.to_logits = nn.Sequential(
            nn.Conv2d(dim_last, dim_last, 3, padding = 1),
            Rearrange('b ... -> b (...)'),
            nn.Linear(latent_dim, 1),
            Rearrange('b 1 -> b')
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return self.to_logits(x)

# c-vivit - 3d ViT with factorized spatial and temporal attention made into an vqgan-vae autoencoder

class CViViT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        codebook_size,
        image_size,
        patch_size,
        temporal_patch_size,
        spatial_depth,
        temporal_depth,
        discr_base_dim = 16,
        dim_head = 64,
        heads = 8,
        channels = 3,
        use_vgg_and_gan = True,
        vgg = None,
        discr_layers = 4,
        use_hinge_loss = True
    ):
        """
        einstein notations:

        b - batch
        c - channels
        t - time
        d - feature dimension
        p1, p2, pt - image patch sizes and then temporal patch size
        """

        super().__init__()

        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size

        self.temporal_patch_size = temporal_patch_size

        self.spatial_rel_pos_bias = ContinuousPositionBias(dim = dim, heads = heads)

        assert (self.image_size[0] % patch_height) == 0 and (self.image_size[1] % patch_width) == 0

        self.to_patch_emb_first_frame = nn.Sequential(
            Rearrange('b c 1 (h p1) (w p2) -> b 1 h w (c p1 p2)', p1 = patch_height, p2 = patch_width),
            nn.Linear(channels * patch_width * patch_height, dim)
        )

        self.to_patch_emb = nn.Sequential(
            Rearrange('b c (t pt) (h p1) (w p2) -> b t h w (c pt p1 p2)', p1 = patch_height, p2 = patch_width, pt = temporal_patch_size),
            nn.Linear(channels * patch_width * patch_height * temporal_patch_size, dim)
        )

        self.enc_spatial_transformer = Transformer(dim = dim, depth = spatial_depth, dim_head = dim_head, heads = heads)
        self.enc_temporal_transformer = Transformer(dim = dim, depth = temporal_depth, dim_head = dim_head, heads = heads, causal = True)

        self.vq = VectorQuantize(dim = dim, codebook_size = codebook_size, use_cosine_sim = True)

        self.dec_spatial_transformer = Transformer(dim = dim, depth = spatial_depth, dim_head = dim_head, heads = heads)
        self.dec_temporal_transformer = Transformer(dim = dim, depth = temporal_depth, dim_head = dim_head, heads = heads, causal = True)

        self.to_pixels_first_frame = nn.Sequential(
            nn.Linear(dim, channels * patch_width * patch_height),
            Rearrange('b 1 h w (c p1 p2) -> b c 1 (h p1) (w p2)', p1 = patch_height, p2 = patch_width)
        )

        self.to_pixels = nn.Sequential(
            nn.Linear(dim, channels * patch_width * patch_height * temporal_patch_size),
            Rearrange('b t h w (c pt p1 p2) -> b c (t pt) (h p1) (w p2)', p1 = patch_height, p2 = patch_width, pt = temporal_patch_size),
        )

        # turn off GAN and perceptual loss if grayscale

        self.vgg = None
        self.discr = None
        self.use_vgg_and_gan = use_vgg_and_gan

        if not use_vgg_and_gan:
            return

        # preceptual loss

        if exists(vgg):
            self.vgg = vgg
        else:
            self.vgg = torchvision.models.vgg16(pretrained = True)
            self.vgg.classifier = nn.Sequential(*self.vgg.classifier[:-2])

        # gan related losses

        layer_mults = list(map(lambda t: 2 ** t, range(discr_layers)))
        layer_dims = [dim * mult for mult in layer_mults]
        dims = (dim, *layer_dims)

        self.discr = Discriminator(dim = discr_base_dim, channels = channels, image_size = self.image_size)

        self.discr_loss = hinge_discr_loss if use_hinge_loss else bce_discr_loss
        self.gen_loss = hinge_gen_loss if use_hinge_loss else bce_gen_loss

    def calculate_video_token_mask(self, videos, video_frame_mask):
        *_, h, w = videos.shape
        ph, pw = self.patch_size

        assert torch.all(((video_frame_mask.sum(dim = -1) - 1) % self.temporal_patch_size) == 0), 'number of frames must be divisible by temporal patch size, subtracting off the first frame'
        first_frame_mask, rest_frame_mask = video_frame_mask[:, :1], video_frame_mask[:, 1:]
        rest_vq_mask = rearrange(rest_frame_mask, 'b (f p) -> b f p', p = self.temporal_patch_size)
        video_mask = torch.cat((first_frame_mask, rest_vq_mask.any(dim = -1)), dim = -1)
        return repeat(video_mask, 'b f -> b (f hw)', hw = (h // ph) * (w // pw))

    def get_video_patch_shape(self, num_frames, include_first_frame = True):
        patch_frames = 0

        if include_first_frame:
            num_frames -= 1
            patch_frames += 1

        patch_frames += (num_frames // self.temporal_patch_size)

        return (patch_frames, *self.patch_height_width)

    @property
    def image_num_tokens(self):
        return int(self.image_size[0] / self.patch_size[0]) * int(self.image_size[1] / self.patch_size[1])

    def frames_per_num_tokens(self, num_tokens):
        tokens_per_frame = self.image_num_tokens

        assert (num_tokens % tokens_per_frame) == 0, f'number of tokens must be divisible by number of tokens per frame {tokens_per_frame}'
        assert (num_tokens > 0)

        pseudo_frames = num_tokens // tokens_per_frames
        return (pseudo_frames - 1) * self.temporal_patch_size + 1

    def num_tokens_per_frames(self, num_frames, include_first_frame = True):
        image_num_tokens = self.image_num_tokens

        total_tokens = 0

        if include_first_frame:
            num_frames -= 1
            total_tokens += image_num_tokens

        assert (num_frames % self.temporal_patch_size) == 0

        return total_tokens + int(num_frames / self.temporal_patch_size) * image_num_tokens

    def copy_for_eval(self):
        device = next(self.parameters()).device
        vae_copy = copy.deepcopy(self.cpu())

        if vae_copy.use_vgg_and_gan:
            del vae_copy.discr
            del vae_copy.vgg

        vae_copy.eval()
        return vae_copy.to(device)

    @remove_vgg
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    @remove_vgg
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    def decode_from_codebook_indices(self, indices):
        codes = self.vq.codebook[indices]
        return self.decode(codes)

    @property
    def patch_height_width(self):
        return self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]

    def encode(
        self,
        tokens
    ):
        b = tokens.shape[0]
        h, w = self.patch_height_width

        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')

        attn_bias = self.spatial_rel_pos_bias(h, w, device = tokens.device)

        tokens = self.enc_spatial_transformer(tokens, attn_bias = attn_bias)

        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b = b, h = h , w = w)

        # encode - temporal

        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')

        tokens = self.enc_temporal_transformer(tokens)

        tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b = b, h = h, w = w)

        return tokens

    def decode(
        self,
        tokens
    ):
        b = tokens.shape[0]
        h, w = self.patch_height_width

        if tokens.ndim == 3:
            tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = h, w = w)

        # decode - temporal

        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')

        tokens = self.dec_temporal_transformer(tokens)

        tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b = b, h = h, w = w)

        # decode - spatial

        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')

        attn_bias = self.spatial_rel_pos_bias(h, w, device = tokens.device)

        tokens = self.dec_spatial_transformer(tokens, attn_bias = attn_bias)

        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b = b, h = h , w = w)

        # to pixels

        first_frame_token, rest_frames_tokens = tokens[:, :1], tokens[:, 1:]

        first_frame = self.to_pixels_first_frame(first_frame_token)

        rest_frames = self.to_pixels(rest_frames_tokens)

        recon_video = torch.cat((first_frame, rest_frames), dim = 2)

        return recon_video

    def forward(
        self,
        video,
        mask = None,
        return_recons = False,
        return_recons_only = False,
        return_discr_loss = False,
        apply_grad_penalty = True,
        return_only_codebook_ids = False
    ):
        assert video.ndim in {4, 5}

        is_image = video.ndim == 4

        if is_image:
            video = rearrange(video, 'b c h w -> b c 1 h w')
            assert not exists(mask)

        b, c, f, *image_dims = video.shape

        assert tuple(image_dims) == self.image_size
        assert not exists(mask) or mask.shape[-1] == f

        first_frame, rest_frames = video[:, :, :1], video[:, :, 1:]

        # derive patches

        first_frame_tokens = self.to_patch_emb_first_frame(first_frame)
        rest_frames_tokens = self.to_patch_emb(rest_frames)

        tokens = torch.cat((first_frame_tokens, rest_frames_tokens), dim = 1)

        # save height and width in

        shape = tokens.shape
        *_, h, w, _ = shape

        # encode - spatial

        tokens = self.encode(tokens)

        # quantize

        tokens, packed_fhw_shape = pack([tokens], 'b * d')

        vq_mask = None
        if exists(mask):
            vq_mask = self.calculate_video_token_mask(video, mask)

        tokens, indices, commit_loss = self.vq(tokens, mask = vq_mask)

        if return_only_codebook_ids:
            indices, = unpack(indices, packed_fhw_shape, 'b *')
            return indices

        tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = h, w = w)

        recon_video = self.decode(tokens)

        returned_recon = rearrange(recon_video, 'b c 1 h w -> b c h w') if is_image else recon_video.clone()

        if return_recons_only:
            return returned_recon

        if exists(mask):
            # variable lengthed video / images training
            recon_loss = F.mse_loss(video, recon_video, reduction = 'none')
            recon_loss = recon_loss[repeat(mask, 'b t -> b c t', c = c)]
            recon_loss = recon_loss.mean()
        else:
            recon_loss = F.mse_loss(video, recon_video)

        # whether to return discriminator loss

        if return_discr_loss:
            assert exists(self.discr), 'discriminator must exist to train it'

            # use first frame for now

            video = video[:, :, 0]
            recon_video = recon_video[:, :, 0]

            recon_video = recon_video.detach()
            video.requires_grad_()

            recon_video_discr_logits, video_discr_logits = map(self.discr, (recon_video, video))

            discr_loss = self.discr_loss(recon_video_discr_logits, video_discr_logits)

            if apply_grad_penalty:
                gp = gradient_penalty(video, video_discr_logits)
                loss = discr_loss + gp

            if return_recons:
                return loss, returned_recon

            return loss

        # early return if training on grayscale

        if not self.use_vgg_and_gan:
            if return_recons:
                return recon_loss, returned_recon

            return recon_loss

        # perceptual loss

        # use first frame for now - todo, randomly sample or set some maximum frames to help with mem

        input_vgg_input = video[:, :, 0]
        recon_vgg_input = recon_video[:, :, 0]

        # handle grayscale for vgg

        if video.shape[1] == 1:
            input_vgg_input, recon_vgg_input = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c = 3), (img_vgg_input, fmap_vgg_input))

        input_vgg_feats = self.vgg(input_vgg_input)
        recon_vgg_feats = self.vgg(recon_vgg_input)

        perceptual_loss = F.mse_loss(input_vgg_feats, recon_vgg_feats)

        # generator loss

        gen_loss = self.gen_loss(self.discr(recon_vgg_input))

        # calculate adaptive weight

        last_dec_layer = self.to_pixels[0].weight

        norm_grad_wrt_gen_loss = grad_layer_wrt_loss(gen_loss, last_dec_layer).norm(p = 2)
        norm_grad_wrt_perceptual_loss = grad_layer_wrt_loss(perceptual_loss, last_dec_layer).norm(p = 2)

        adaptive_weight = safe_div(norm_grad_wrt_perceptual_loss, norm_grad_wrt_gen_loss)
        adaptive_weight.clamp_(max = 1e4)

        # combine losses

        loss = recon_loss + perceptual_loss + commit_loss + adaptive_weight * gen_loss

        if return_recons:
            return loss, returned_recon

        return loss

# mask git

class MaskGit(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        max_seq_len,
        gradient_shrink_alpha = 0.1,
        heads = 8,
        dim_head = 64,
        **kwargs
    ):
        super().__init__()
        self.mask_id = num_tokens

        self.token_emb = nn.Embedding(num_tokens + 1, dim) # last token is used as mask_id
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.gradient_shrink_alpha = gradient_shrink_alpha  # used with great success in cogview and GLM 130B attention net

        self.continuous_pos_bias = ContinuousPositionBias(dim = dim_head, heads = heads, num_dims = 3)

        self.transformer = Transformer(
            dim = dim,
            attn_num_null_kv = 2,
            has_cross_attn = True,
            dim_head = dim_head,
            heads = heads,
            **kwargs
        )

        self.to_logits = nn.Linear(dim, num_tokens)

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 3,
        **kwargs
    ):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        cond_drop_prob = 0.,
        text_mask = None,
        video_mask = None,
        video_patch_shape = None,
        **kwargs
    ):
        b, n, device = *x.shape, x.device

        if not exists(text_mask):
            text_mask = torch.ones((b, n), device = device, dtype = torch.bool)

        rel_pos_bias = None
        if exists(video_patch_shape):
            rel_pos_bias = self.continuous_pos_bias(*video_patch_shape, device = device)

        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((b,), 1 - cond_drop_prob, device = device)
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask

        x = self.token_emb(x)
        x = self.pos_emb(torch.arange(n, device = device)) + x

        x = x * self.gradient_shrink_alpha + x.detach() * (1 - self.gradient_shrink_alpha)

        x = self.transformer(
            x,
            attn_bias = rel_pos_bias,
            self_attn_mask = video_mask,
            cross_attn_context_mask = text_mask,
            **kwargs
        )

        return self.to_logits(x)

class MaskGitTrainWrapper(nn.Module):
    def __init__(
        self,
        maskgit,
        *,
        steps
    ):
        super().__init__()
        self.maskgit = maskgit
        self.mask_id = maskgit.mask_id

        self.steps = steps

    def forward(self, x, video_mask = None, **kwargs):
        batch, seq, device = *x.shape, x.device

        self.maskgit.train()

        rand_step = torch.randint(0, self.steps, (1,), device = device)
        mask_token_prob = torch.cos(rand_step * math.pi * 0.5 / self.steps) # cosine schedule was best

        if not exists(video_mask):
            video_mask = torch.ones((batch, seq), device = device).bool()

        mask_token_mask = get_mask_subset_with_prob(video_mask, mask_token_prob)

        masked_input = torch.where(mask_token_mask, self.mask_id, x)

        logits = self.maskgit(masked_input, video_mask = video_mask, **kwargs)

        loss = F.cross_entropy(logits[mask_token_mask], x[mask_token_mask])
        return loss

# token critic

class TokenCritic(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        max_seq_len,
        has_cross_attn = False,
        **kwargs
    ):
        super().__init__()
        self.has_cross_attn = has_cross_attn

        self.mask_id = num_tokens

        self.token_emb = nn.Embedding(num_tokens + 1, dim) # last token is used as mask_id
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.transformer = Transformer(
            dim = dim,
            has_cross_attn = has_cross_attn,
            **kwargs
        )

        self.to_logits = nn.Sequential(
            nn.Linear(dim, 1),
            Rearrange('... 1 -> ...')
        )

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 3,
        **kwargs
    ):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        text_mask = None,
        cond_drop_prob = None,
        context = None,
        **kwargs
    ):
        b, n, device = *x.shape, x.device

        if not exists(text_mask):
            text_mask = torch.ones((b, n), device = device, dtype = torch.bool)

        if exists(context) and cond_drop_prob > 0:
            keep_mask = prob_mask_like((b,), 1 - cond_drop_prob, device = device)
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask

        x = self.token_emb(x)
        x = self.pos_emb(torch.arange(n, device = device)) + x

        x = self.transformer(
            x,
            context = context,
            cross_attn_context_mask = text_mask,
            **kwargs
        )

        return self.to_logits(x)

@typechecked
class CriticTrainer(nn.Module):
    def __init__(
        self,
        *,
        maskgit: MaskGit,
        critic: TokenCritic,
        temperature = 0.,
        t5_name = DEFAULT_T5_NAME,
        text_embed_dim = None,
        cond_drop_prob = 0.25,
        max_text_len = 128
    ):
        super().__init__()
        self.maskgit = maskgit
        self.mask_id = maskgit.mask_id

        self.critic = critic
        self.temperature = temperature

        # text conditioning

        text_embed_dim = default(text_embed_dim, get_encoded_dim(t5_name))
        self.encode_texts = partial(t5_encode_text, name = t5_name)
        self.text_embed_dim = text_embed_dim
        self.max_text_len = max_text_len

        assert cond_drop_prob > 0.
        self.cond_drop_prob = cond_drop_prob # classifier free guidance for transformers - @crowsonkb

    def forward(
        self,
        x,
        texts: Optional[List[str]] = None,
        text_embeds = None,
        cond_drop_prob = None,
        **kwargs
    ):
        batch, seq, device = *x.shape, x.device

        self.critic.train()

        # get time and number of tokens to mask

        rand_time = uniform((1,), device)
        num_tokens_mask = (seq * torch.cos(rand_time * math.pi * 0.5)).round().long().clamp(min = 1) # cosine schedule was best

        _, indices = torch.randn((batch, seq), device = device).topk(num_tokens_mask.item(), dim = -1)

        mask = torch.zeros((batch, seq), device = device).scatter(1, indices, 1.).bool()

        # mask the input into maskgit

        masked_input = torch.where(mask, self.mask_id, x)

        # get text conditioning

        if not exists(text_embeds):
            with torch.no_grad():
                text_embeds = self.encode_texts(texts, output_device = device)

        text_mask = torch.any(text_embeds != 0, dim = -1) # save the researcher from having to think about mask, by assuming if all of the feature dimension is 0, it is masked out

        # condition dropout for Katherine's (@crowsonkb) version of classifier free guidance for transformers

        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # predict masked tokens

        with torch.no_grad():
            self.maskgit.eval()

            logits = self.maskgit(
                masked_input,
                context = text_embeds,
                text_mask = text_mask,
                cond_drop_prob = cond_drop_prob,
                **kwargs
            )

        # sample the predicted masked tokens

        if self.temperature <= 0:
            pred_video_ids = logits.argmax(dim = -1)
        else:
            pred_video_ids = gumbel_sample(logits, temperature = self.temperature)

        # derive critic input

        critic_input = torch.where(mask, pred_video_ids, x)

        # critic may or may not need text conditioning

        critic_kwargs = dict()
        if self.critic.has_cross_attn:
            critic_kwargs = dict(
                context = text_embeds,
                text_mask = text_mask,
                cond_drop_prob = cond_drop_prob
            )

        pred_fake_or_real_logits = self.critic(
            critic_input,
            **critic_kwargs
        )

        critic_loss = F.binary_cross_entropy_with_logits(
            pred_fake_or_real_logits,
            mask.float()
        )

        return critic_loss

# main class

@typechecked
class Phenaki(nn.Module):
    def __init__(
        self,
        *,
        maskgit: MaskGit,
        cvivit: CViViT = None,
        critic: TokenCritic = None,
        steps = 18,                         # 18 is the ideal steps with token critic
        t5_name = DEFAULT_T5_NAME,
        sample_temperature = 0.,
        text_embed_dim = None,
        cond_drop_prob = 0.25,
        max_text_len = 128
    ):
        super().__init__()

        self.cvivit = cvivit.copy_for_eval()

        self.maskgit = maskgit
        self.mask_id = maskgit.mask_id
        self.maskgit_trainer = MaskGitTrainWrapper(maskgit, steps = steps)

        # sampling

        if exists(critic):
            critic = critic.eval()

        self.critic = critic
        self.steps = steps
        self.sample_temperature = sample_temperature

        # text conditioning

        text_embed_dim = default(text_embed_dim, get_encoded_dim(t5_name))
        self.encode_texts = partial(t5_encode_text, name = t5_name)
        self.text_embed_dim = text_embed_dim
        self.max_text_len = max_text_len

        assert cond_drop_prob > 0.
        self.cond_drop_prob = cond_drop_prob # classifier free guidance for transformers - @crowsonkb

    def sample_image(
        self,
        *,
        text,
        cond_scale = 3.,
        starting_temperature = 0.9,
        noise_K = 1.
    ):
        single_framed_video = self.sample(
            text = text,
            num_frames = 1,
            cond_scale = cond_scale,
            starting_temperature = starting_temperature,
            noise_K = noise_K
        )

        return rearrange(single_framed_video, 'b c 1 h w -> b c h w')

    @eval_decorator
    @torch.no_grad()
    def sample(
        self,
        *,
        text,
        num_frames,
        prime_frames = None,
        cond_scale = 3.,
        starting_temperature = 0.9,
        noise_K = 1. # hyperparameter for noising of critic score in section 3.2 of token-critic paper, need to find correct value
    ):
        device = next(self.parameters()).device

        # derive the priming token ids, to be prepended to the input being demasked by mask-git at each round

        has_prime = exists(prime_frames)
        prime_token_ids = None
        prime_token_length = 0
        prime_num_frames = 0

        if has_prime:
            with torch.no_grad():
                prime_token_ids = self.cvivit(prime_frames, return_only_codebook_ids = True)
                patch_shape = prime_token_ids.shape[1:]
                prime_token_ids = rearrange(prime_token_ids, 'b ... -> b (...)')

            prime_token_length = prime_token_ids.shape[-1]
            prime_num_frames = prime_frames.shape[2]

        num_tokens = self.cvivit.num_tokens_per_frames(num_frames, include_first_frame = not exists(prime_frames))

        # get text embeds and mask

        with torch.no_grad():
            text_embeds = self.encode_texts([text], output_device = device)
            text_mask = torch.any(text_embeds != 0, dim = -1)

        # derive video patch shape

        patch_shape = self.cvivit.get_video_patch_shape(num_frames + prime_num_frames, include_first_frame = True)

        # get video token ids

        shape = (1, num_tokens)

        video_token_ids = torch.full(shape, self.mask_id, device = device)
        mask = torch.ones(shape, device = device, dtype = torch.bool)

        scores = None # keeping track of the confidence or critic scores, determining what should be masked at the next iteration

        for step in range(self.steps):
            is_first_step = step == 0
            is_last_step = step == (self.steps - 1)

            steps_til_x0 = self.steps - (step + 1)

            if not is_first_step and exists(scores):
                time = torch.full((1,), step / self.steps, device = device)
                num_tokens_mask = (num_tokens * torch.cos(time * math.pi * 0.5)).round().long().clamp(min = 1)

                _, indices = scores.topk(num_tokens_mask.item(), dim = -1)
                mask = torch.zeros(shape, device = device).scatter(1, indices, 1).bool()

            video_token_ids = torch.where(mask, self.mask_id, video_token_ids)

            input_token_ids = video_token_ids if not has_prime else torch.cat((prime_token_ids, video_token_ids), dim = -1)

            logits = self.maskgit.forward_with_cond_scale(
                input_token_ids,
                video_patch_shape = patch_shape,
                context = text_embeds,
                text_mask = text_mask,
                cond_scale = cond_scale
            )

            if has_prime:
                logits = logits[:, prime_token_length:]

            temperature = starting_temperature * (steps_til_x0 / self.steps)
            pred_video_ids = gumbel_sample(logits, temperature = temperature)

            video_token_ids = torch.where(mask, pred_video_ids, video_token_ids)

            if not is_last_step:
                if exists(self.critic):
                    critic_kwargs = dict()

                    if self.critic.has_cross_attn:
                        critic_kwargs = dict(
                            context = text_embeds,
                            text_mask = text_mask,
                            cond_csale = cond_scale
                        )

                    with torch.no_grad():
                        scores = self.critic(
                            video_token_ids,
                            **critic_kwargs
                        )

                    noise = noise_K * (uniform(scores.shape, device) - 0.5) * (steps_til_x0 / self.steps)
                    scores = scores + noise
                else:
                    scores = logits.gather(2, rearrange(pred_video_ids, '... -> ... 1'))
                    scores = 1 - rearrange(scores, '... 1 -> ...')
                    scores = torch.where(mask, scores, -1e4)

        if has_prime:
            video_token_ids = torch.cat((prime_token_ids, video_token_ids), dim = -1)

        video = self.cvivit.decode_from_codebook_indices(video_token_ids)

        if has_prime:
            video = video[:, :, prime_num_frames:]

        return video

    def forward(
        self,
        videos = None,
        texts: Optional[List[str]] = None,
        video_codebook_ids = None,
        video_frame_mask = None,
        text_embeds = None,
        cond_drop_prob = None
    ):
        assert exists(videos) ^ exists(video_codebook_ids), 'either raw video or '
        assert not (exists(videos) and not exists(self.cvivit)), 'cvivit must be provided if one wants to encode the videos live during training'

        assert exists(text_embeds) ^ exists(texts), 'either raw text of text embeds must be given'
        assert not (exists(text_embeds) and text_embeds.shape[-1] != self.text_embed_dim), 'text embedding dimension is not correct'

        if not exists(video_codebook_ids):
            assert videos.ndim in {4, 5}

            if videos.ndim == 4:
                videos = rearrange(videos, 'b c h w -> b c 1 h w')

            with torch.no_grad():
                self.cvivit.eval()
                video_codebook_ids = self.cvivit(videos, return_only_codebook_ids = True)
                patch_shape = video_codebook_ids.shape[1:]
                video_codebook_ids = rearrange(video_codebook_ids, 'b ... -> b (...)')

        if not exists(text_embeds):
            with torch.no_grad():
                text_embeds = self.encode_texts(texts, output_device = video_codebook_ids.device)

        batch, device = text_embeds.shape[0], text_embeds.device

        text_mask = torch.any(text_embeds != 0, dim = -1) # save the researcher from having to think about mask, by assuming if all of the feature dimension is 0, it is masked out

        # condition dropout for Katherine's (@crowsonkb) version of classifier free guidance for transformers

        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # calculate video frame mask

        video_mask = None
        if exists(video_frame_mask):
            video_mask = self.cvivit.calculate_video_token_mask(
                videos,
                video_frame_mask = video_frame_mask
            )

        # train maskgit with text condition

        loss = self.maskgit_trainer(
            video_codebook_ids,
            video_patch_shape = patch_shape,
            cond_drop_prob = cond_drop_prob,
            video_mask = video_mask,
            text_mask = text_mask,
            context = text_embeds
        )

        return loss

# make video function

@typechecked
def make_video(
    phenaki: Phenaki,
    texts: List[str],
    num_frames,
    prime_lengths
):
    num_scenes = len(texts)
    num_frames = cast_tuple(num_frames, num_scenes)

    prime_lengths = cast_tuple(prime_lengths, num_scenes - 1)
    prime_lengths = (*prime_lengths, 0) # last scene needs no priming

    entire_video = []
    video_prime = None
    scenes = []

    for text, scene_num_frames, next_scene_prime_length in zip(texts, num_frames, prime_lengths):
        video = phenaki.sample(text = text, prime_frames = video_prime, num_frames = scene_num_frames)
        scenes.append(video)

        video_prime = video[:, :, -next_scene_prime_length:]

    return torch.cat(scenes, dim = 2), scenes

import copy
import math
from functools import partial, wraps
from typing import Optional, List, Union

import torch
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
from torch import nn, einsum
import torchvision

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from vector_quantize_pytorch import VectorQuantize

from phenaki_pytorch.t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else (val,) * length

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
    return nn.LeakyReLU(0.1)

def safe_div(numer, denom, eps = 1e-8):
    return numer / (denom + eps)

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t + eps)

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

    def __init__(self, *, dim, heads, layers = 2):
        super().__init__()
        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(2, dim), leaky_relu()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), leaky_relu()))

        self.net.append(nn.Linear(dim, heads))
        self.register_buffer('rel_pos', None, persistent = False)

    def forward(self, n, device):
        fmap_size = int(math.sqrt(n))

        if not exists(self.rel_pos):
            pos = torch.arange(fmap_size, device = device)
            grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
            grid = rearrange(grid, 'c i j -> (i j) c')
            rel_pos = rearrange(grid, 'i c -> i 1 c') - rearrange(grid, 'j c -> 1 j c')
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
        num_null_kv = 0
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
            sim = sim + attn_bias

        if exists(mask):
            mask = F.pad(mask, (j - mask.shape[-1], 0), value = True)
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
        mask = None
    ):

        for self_attn, cross_attn, ff in self.layers:
            x = self_attn(x, attn_bias = attn_bias) + x

            if exists(cross_attn) and exists(context):
                x = cross_attn(x, context = context, mask = None) + x

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

class Discriminator(nn.Module):
    def __init__(
        self,
        dims,
        channels = 3,
        groups = 8,
        init_kernel_size = 5,
        cross_embed_kernel_sizes = (3, 7, 15)
    ):
        super().__init__()
        init_dim, *_, final_dim = dims
        dim_pairs = zip(dims[:-1], dims[1:])

        self.layers = nn.ModuleList([nn.Sequential(
            nn.Conv2d(channels, init_dim, 5, padding = 2),
            leaky_relu()
        )])

        for dim_in, dim_out in dim_pairs:
            self.layers.append(nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 4, stride = 2, padding = 1),
                leaky_relu(),
                nn.GroupNorm(groups, dim_out),
                ResnetBlock(dim_out, dim_out),
            ))

        self.to_logits = nn.Sequential( # return 5 x 5, for PatchGAN-esque training
            nn.Conv2d(final_dim, final_dim, 1),
            leaky_relu(),
            nn.Conv2d(final_dim, 1, 4)
        )

    def forward(self, x):
        for net in self.layers:
            x = net(x)

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

        self.image_size = image_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size

        self.spatial_rel_pos_bias = ContinuousPositionBias(dim = dim, heads = heads)

        assert (self.image_size % self.patch_size) == 0

        self.to_patch_emb_first_frame = nn.Sequential(
            Rearrange('b c 1 (h p1) (w p2) -> b 1 h w (c p1 p2)', p1 = patch_size, p2 = patch_size),
            nn.Linear(channels * patch_size * patch_size, dim)
        )

        self.to_patch_emb = nn.Sequential(
            Rearrange('b c (t pt) (h p1) (w p2) -> b t h w (c pt p1 p2)', p1 = patch_size, p2 = patch_size, pt = temporal_patch_size),
            nn.Linear(channels * patch_size * patch_size * temporal_patch_size, dim)
        )

        self.enc_spatial_transformer = Transformer(dim = dim, depth = spatial_depth, dim_head = dim_head, heads = heads)
        self.enc_temporal_transformer = Transformer(dim = dim, depth = temporal_depth, dim_head = dim_head, heads = heads, causal = True)

        self.vq = VectorQuantize(dim = dim, codebook_size = codebook_size, use_cosine_sim = True)

        self.dec_spatial_transformer = Transformer(dim = dim, depth = spatial_depth, dim_head = dim_head, heads = heads)
        self.dec_temporal_transformer = Transformer(dim = dim, depth = temporal_depth, dim_head = dim_head, heads = heads, causal = True)

        self.to_pixels_first_frame = nn.Sequential(
            nn.Linear(dim, channels * patch_size * patch_size),
            Rearrange('b 1 h w (c p1 p2) -> b c 1 (h p1) (w p2)', p1 = patch_size, p2 = patch_size)
        )

        self.to_pixels = nn.Sequential(
            nn.Linear(dim, channels * patch_size * patch_size * temporal_patch_size),
            Rearrange('b t h w (c pt p1 p2) -> b c (t pt) (h p1) (w p2)', p1 = patch_size, p2 = patch_size, pt = temporal_patch_size),
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

        self.discr = Discriminator(dims = dims, channels = channels)

        self.discr_loss = hinge_discr_loss if use_hinge_loss else bce_discr_loss
        self.gen_loss = hinge_gen_loss if use_hinge_loss else bce_gen_loss

    def frames_per_num_tokens(self, num_tokens):
        tokens_per_frame = int(self.image_size / self.patch_size) ** 2
        assert (num_tokens % tokens_per_frame) == 0, f'number of tokens must be divisible by number of tokens per frame {tokens_per_frame}'
        assert (num_tokens > 0)

        pseudo_frames = num_tokens // tokens_per_frames
        return (pseudo_frames - 1) * self.temporal_patch_size + 1

    def num_tokens_per_frames(self, num_frames, include_first_frame = True):
        image_num_tokens = int(self.image_size / self.patch_size) ** 2

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

    def encode(
        self,
        tokens
    ):
        b = tokens.shape[0]
        h = w = (self.image_size // self.patch_size)

        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')

        attn_bias = self.spatial_rel_pos_bias(tokens.shape[-2], device = tokens.device)

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
        h = w = (self.image_size // self.patch_size)

        if tokens.ndim == 3:
            tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = h, w = w)

        # decode - temporal

        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')

        tokens = self.dec_temporal_transformer(tokens)

        tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b = b, h = h, w = w)

        # decode - spatial

        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')

        attn_bias = self.spatial_rel_pos_bias(tokens.shape[-2], device = tokens.device)

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
        return_recons = False,
        return_discr_loss = False,
        apply_grad_penalty = True,
        return_only_codebook_ids = False
    ):
        assert video.ndim == 5
        b, c, f, *_ = video.shape
        assert video.shape[-1] == self.image_size and video.shape[-2] == self.image_size

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

        tokens = rearrange(tokens, 'b t h w d -> b (t h w) d')

        tokens, indices, commit_loss = self.vq(tokens)

        if return_only_codebook_ids:
            return indices

        tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = h, w = w)

        recon_video = self.decode(tokens)

        recon_loss = F.mse_loss(video, recon_video)

        # whether to return discriminator loss

        if return_discr_loss:
            assert exists(self.discr), 'discriminator must exist to train it'

            recons = recon_video.clone()

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
                return loss, recons

            return loss

        # early return if training on grayscale

        if not self.use_vgg_and_gan:
            if return_recons:
                return recon_loss, recon_video

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
            return loss, recon_video

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
        **kwargs
    ):
        super().__init__()
        self.mask_id = num_tokens

        self.token_emb = nn.Embedding(num_tokens + 1, dim) # last token is used as mask_id
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.gradient_shrink_alpha = gradient_shrink_alpha  # used with great success in cogview and GLM 130B attention net

        self.transformer = Transformer(
            dim = dim,
            attn_num_null_kv = 2,
            has_cross_attn = True,
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

    def forward(self, x, cond_drop_prob = 0., text_mask = None, **kwargs):
        b, n, device = *x.shape, x.device

        if not exists(text_mask):
            text_mask = torch.ones((b, n), device = device, dtype = torch.bool)

        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((b,), 1 - cond_drop_prob, device = device)
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask

        x = self.token_emb(x)
        x = self.pos_emb(torch.arange(n, device = device)) + x

        x = x * self.gradient_shrink_alpha + x.detach() * (1 - self.gradient_shrink_alpha)

        x = self.transformer(x, **kwargs)

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

    def forward(self, x, **kwargs):
        batch, seq, device = *x.shape, x.device

        self.maskgit.train()

        rand_step = torch.randint(0, self.steps, (1,), device = device)
        num_tokens_mask = (seq * torch.cos(rand_step * math.pi * 0.5 / self.steps)).round().long().clamp(min = 1) # cosine schedule was best

        _, indices = torch.randn((batch, seq), device = device).topk(num_tokens_mask.item(), dim = -1)
        mask = torch.zeros((batch, seq), device = device).scatter(1, indices, 1.).bool()

        masked_input = torch.where(mask, self.mask_id, x)

        logits = self.maskgit(masked_input, **kwargs)

        loss = F.cross_entropy(logits[mask], x[mask])
        return loss

# token critic

class TokenCritic(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        max_seq_len,
        **kwargs
    ):
        super().__init__()
        self.mask_id = num_tokens

        self.token_emb = nn.Embedding(num_tokens + 1, dim) # last token is used as mask_id
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.transformer = Transformer(
            dim = dim,
            **kwargs
        )

        self.to_logits = nn.Sequential(
            nn.Linear(dim, 1),
            Rearrange('... 1 -> ...')
        )


    def forward(self, x, **kwargs):
        n, device = x.shape[1], x.device

        x = self.token_emb(x)
        x = self.pos_emb(torch.arange(n, device = device)) + x

        x = self.transformer(x, **kwargs)

        return self.to_logits(x)

class CriticTrainer(nn.Module):
    def __init__(
        self,
        *,
        maskgit,
        critic,
        temperature = 0.
    ):
        super().__init__()
        self.maskgit = maskgit
        self.mask_id = maskgit.mask_id

        self.critic = critic
        self.temperature = temperature

    def forward(self, x, **kwargs):
        batch, seq, device = *x.shape, x.device

        self.critic.train()

        rand_time = uniform((1,), device)
        num_tokens_mask = (seq * torch.cos(rand_time * math.pi * 0.5)).round().long().clamp(min = 1) # cosine schedule was best

        _, indices = torch.randn((batch, seq), device = device).topk(num_tokens_mask.item(), dim = -1)

        mask = torch.zeros((batch, seq), device = device).scatter(1, indices, 1.).bool()

        # mask the input into maskgit

        masked_input = torch.where(mask, self.mask_id, x)

        # predict masked tokens

        with torch.no_grad():
            self.maskgit.eval()
            logits = self.maskgit(masked_input, **kwargs)

        # sample the predicted masked tokens

        if self.temperature <= 0:
            pred_video_ids = logits.argmax(dim = -1)
        else:
            pred_video_ids = gumbel_sample(logits, temperature = self.temperature)

        # derive critic input

        critic_input = torch.where(mask, pred_video_ids, x)

        pred_fake_or_real_logits = self.critic(critic_input)

        critic_loss = F.binary_cross_entropy_with_logits(
            pred_fake_or_real_logits,
            mask.float()
        )

        return critic_loss

# main class

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

            prime_token_length = prime_token_ids.shape[-1]
            prime_num_frames = prime_frames.shape[2]

        num_tokens = self.cvivit.num_tokens_per_frames(num_frames, include_first_frame = not exists(prime_frames))

        # get text embeds and mask

        with torch.no_grad():
            text_embeds = self.encode_texts([text], output_device = device)
            text_mask = torch.any(text_embeds != 0, dim = -1)

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
                context = text_embeds,
                mask = text_mask,
                cond_scale = cond_scale
            )

            if has_prime:
                logits = logits[:, prime_token_length:]

            temperature = starting_temperature * (steps_til_x0 / self.steps)
            pred_video_ids = gumbel_sample(logits, temperature = temperature)

            video_token_ids = torch.where(mask, pred_video_ids, video_token_ids)

            if not is_last_step:
                if exists(self.critic):
                    scores = self.critic(video_token_ids)

                    noise = K * (uniform(scores.shape, device) - 0.5) * (steps_til_x0 / self.steps)
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
        text_embeds = None,
        cond_drop_prob = None
    ):
        assert exists(videos) ^ exists(video_codebook_ids), 'either raw video or '
        assert not (exists(videos) and not exists(self.cvivit)), 'cvivit must be provided if one wants to encode the videos live during training'

        assert exists(text_embeds) ^ exists(texts), 'either raw text of text embeds must be given'
        assert not (exists(text_embeds) and text_embeds.shape[-1] != self.text_embed_dim), 'text embedding dimension is not correct'

        if not exists(video_codebook_ids):
            with torch.no_grad():
                self.cvivit.eval()
                video_codebook_ids = self.cvivit(videos, return_only_codebook_ids = True)

        if not exists(text_embeds):
            with torch.no_grad():
                text_embeds = self.encode_texts(texts, output_device = video_codebook_ids.device)

        batch, device = text_embeds.shape[0], text_embeds.device

        text_mask = torch.any(text_embeds != 0, dim = -1) # save the researcher from having to think about mask, by assuming if all of the feature dimension is 0, it is masked out

        # condition dropout for Katherine's (@crowsonkb) version of classifier free guidance for transformers

        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # train maskgit with text condition

        loss = self.maskgit_trainer(
            video_codebook_ids,
            cond_drop_prob = cond_drop_prob,
            mask = text_mask,
            context = text_embeds
        )

        return loss

# make video function

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

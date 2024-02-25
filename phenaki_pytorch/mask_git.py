import torch
from torch import nn, einsum
from einops import rearrange, repeat, pack, unpack

from .attention import Attention, Transformer, ContinuousPositionBias
from .utils import exists, prob_mask_like

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
        unconditional = False,
        attn_dropout = 0.,
        ff_dropout = 0.,
        **kwargs
    ):
        super().__init__()
        self.dim = dim

        self.mask_id = num_tokens
        self.unconditional = unconditional

        self.token_emb = nn.Embedding(num_tokens + 1, dim) # last token is used as mask_id

        self.max_seq_len = max_seq_len
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.gradient_shrink_alpha = gradient_shrink_alpha  # used with great success in cogview and GLM 130B attention net

        self.continuous_pos_bias = ContinuousPositionBias(dim = dim_head, heads = heads, num_dims = 3)

        self.transformer = Transformer(
            dim = dim,
            attn_num_null_kv = 2,
            has_cross_attn = not self.unconditional,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            peg = True,
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
        return_embeds = False,
        **kwargs
    ):
        assert x.ndim in {2, 4}, 'video token ids must be of shape (batch, seq) or (batch, frame, height, width)'

        if x.ndim == 4:
            video_patch_shape = x.shape[1:]
            x = rearrange(x, 'b ... -> b (...)')

        b, n, device = *x.shape, x.device

        if not exists(text_mask):
            text_mask = torch.ones((b, n), device = device, dtype = torch.bool)

        assert exists(video_patch_shape), 'video patch shape must be given'

        rel_pos_bias = self.continuous_pos_bias(*video_patch_shape, device = device)

        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((b,), 1 - cond_drop_prob, device = device)
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask

        video_shape = (b, *video_patch_shape)

        x = self.token_emb(x)

        assert n <= self.max_seq_len, f'the video token sequence length you are passing in ({n}) is greater than the `max_seq_len` ({self.max_seq_len}) set on your `MaskGit`'
        x = self.pos_emb(torch.arange(n, device = device)) + x

        x = x * self.gradient_shrink_alpha + x.detach() * (1 - self.gradient_shrink_alpha)

        x = self.transformer(
            x,
            video_shape = video_shape,
            attn_bias = rel_pos_bias,
            self_attn_mask = video_mask,
            cross_attn_context_mask = text_mask,
            **kwargs
        )

        if return_embeds:
            return x

        return self.to_logits(x)
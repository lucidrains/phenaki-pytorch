import math
import functools
from functools import partial, wraps

from typing import Optional, List, Union
from beartype import beartype

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from phenaki_pytorch.t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME

from phenaki_pytorch.cvivit import CViViT
from phenaki_pytorch.attention import Attention, Transformer, ContinuousPositionBias

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else (val,) * length

def reduce_mult(arr):
    return functools.reduce(lambda x, y: x * y, arr)

def divisible_by(numer, denom):
    return (numer % denom) == 0

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

# tensor helper functions

def log(t, eps = 1e-10):
    return torch.log(t + eps)

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
        **kwargs
    ):
        super().__init__()
        self.mask_id = num_tokens
        self.unconditional = unconditional

        self.token_emb = nn.Embedding(num_tokens + 1, dim) # last token is used as mask_id
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.gradient_shrink_alpha = gradient_shrink_alpha  # used with great success in cogview and GLM 130B attention net

        self.continuous_pos_bias = ContinuousPositionBias(dim = dim_head, heads = heads, num_dims = 3)

        self.transformer = Transformer(
            dim = dim,
            attn_num_null_kv = 2,
            has_cross_attn = not self.unconditional,
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
        x, packed_shape = pack([x], 'b *')

        batch, seq, device = *x.shape, x.device

        self.maskgit.train()

        rand_step = torch.randint(0, self.steps, (1,), device = device)
        mask_token_prob = torch.cos(rand_step * math.pi * 0.5 / self.steps) # cosine schedule was best

        if not exists(video_mask):
            video_mask = torch.ones((batch, seq), device = device).bool()

        mask_token_mask = get_mask_subset_with_prob(video_mask, mask_token_prob)

        masked_input = torch.where(mask_token_mask, self.mask_id, x)

        masked_input, = unpack(masked_input, packed_shape, 'b *')

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
        x = rearrange(x, 'b ... -> b (...)')

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

@beartype
class CriticTrainer(nn.Module):
    def __init__(
        self,
        *,
        maskgit: MaskGit,
        critic: TokenCritic,
        cvivit: Optional[CViViT] = None,
        temperature = 0.,
        t5_name = DEFAULT_T5_NAME,
        text_embed_dim = None,
        cond_drop_prob = 0.25,
        max_text_len = 128
    ):
        super().__init__()

        if exists(cvivit):
            self.cvivit = cvivit.copy_for_eval()

        self.maskgit = maskgit
        self.unconditional = maskgit.unconditional

        self.mask_id = maskgit.mask_id

        self.critic = critic
        self.temperature = temperature

        assert not (maskgit.unconditional and critic.has_cross_attn), 'critic does not need cross attention if maskgit is unconditional'

        # text conditioning

        text_embed_dim = default(text_embed_dim, get_encoded_dim(t5_name))
        self.encode_texts = partial(t5_encode_text, name = t5_name)
        self.text_embed_dim = text_embed_dim
        self.max_text_len = max_text_len

        assert cond_drop_prob > 0.
        self.cond_drop_prob = cond_drop_prob # classifier free guidance for transformers - @crowsonkb

    def forward(
        self,
        *,
        videos = None,
        video_codebook_ids = None,
        texts: Optional[List[str]] = None,
        text_embeds = None,
        cond_drop_prob = None,
        **kwargs
    ):
        # derive codebook ids

        if not exists(video_codebook_ids):
            assert exists(self.cvivit)

            self.cvivit.eval()
            with torch.no_grad():
                video_codebook_ids = self.cvivit(videos, return_only_codebook_ids = True)

        assert video_codebook_ids.ndim == 4

        x, codebook_packed_shape = pack([video_codebook_ids], 'b *')

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

        text_mask = None
        cond_drop_prob = 0

        if not self.unconditional:
            if not exists(text_embeds):
                with torch.no_grad():
                    text_embeds = self.encode_texts(texts, output_device = device)

            text_mask = torch.any(text_embeds != 0, dim = -1) # save the researcher from having to think about mask, by assuming if all of the feature dimension is 0, it is masked out

            # condition dropout for Katherine's (@crowsonkb) version of classifier free guidance for transformers

            cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # get the input back to the original video token shape

        masked_input, = unpack(masked_input, codebook_packed_shape, 'b *')

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

        critic_input, = unpack(critic_input, codebook_packed_shape, 'b *')

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

@beartype
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
        max_text_len = 128,
        critic_noise_anneal_schedule = 'decay'
    ):
        super().__init__()

        self.cvivit = cvivit.copy_for_eval()

        self.maskgit = maskgit
        self.unconditional = maskgit.unconditional

        self.mask_id = maskgit.mask_id
        self.maskgit_trainer = MaskGitTrainWrapper(maskgit, steps = steps)

        # sampling

        if exists(critic):
            critic = critic.eval()

        self.critic = critic
        self.critic_noise_anneal_schedule = critic_noise_anneal_schedule

        self.steps = steps
        self.sample_temperature = sample_temperature

        # text conditioning

        text_embed_dim = default(text_embed_dim, get_encoded_dim(t5_name))
        self.encode_texts = partial(t5_encode_text, name = t5_name)
        self.text_embed_dim = text_embed_dim
        self.max_text_len = max_text_len

        assert cond_drop_prob > 0.
        self.cond_drop_prob = cond_drop_prob # classifier free guidance for transformers - @crowsonkb

    def sample_images(
        self,
        *,
        texts: Union[List[str], str] = None,
        batch_size = 1,
        cond_scale = 3.,
        starting_temperature = 0.9,
        noise_K = 1.
    ):
        single_framed_video = self.sample(
            texts = texts,
            num_frames = 1,
            cond_scale = cond_scale,
            starting_temperature = starting_temperature,
            noise_K = noise_K
        )

        return rearrange(single_framed_video, '... c 1 h w -> ... c h w')

    @eval_decorator
    @torch.no_grad()
    def sample(
        self,
        *,
        num_frames,
        texts: Union[List[str], str] = None,
        prime_frames = None,
        batch_size = 1,
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

        text_embeds = text_mask = None

        if exists(texts):
            if isinstance(texts, str):
                texts = [texts]

            with torch.no_grad():
                text_embeds = self.encode_texts(texts, output_device = device)
                text_mask = torch.any(text_embeds != 0, dim = -1)

            batch_size = len(texts)

        # derive video patch shape

        patch_shape = self.cvivit.get_video_patch_shape(num_frames + prime_num_frames, include_first_frame = True)

        # get video token ids

        shape = (batch_size, num_tokens)

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

                    # different types of annealing

                    if self.critic_noise_anneal_schedule == 'fixed':
                        noise_multiplier = 1.
                    elif self.critic_noise_anneal_schedule == 'decay':
                        noise_multiplier = steps_til_x0 / self.steps
                    elif self.critic_noise_anneal_schedule == 'increase':
                        noise_multiplier = (step + 1) / self.steps
                    else:
                        raise ValueError(f'invalid critic noise anneal schedule name')

                    # noise to add to critic scores

                    noise = noise_K * (uniform(scores.shape, device) - 0.5) * noise_multiplier
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
        *,
        texts: Optional[List[str]] = None,
        video_codebook_ids = None,
        video_frame_mask = None,
        text_embeds = None,
        cond_drop_prob = None
    ):
        assert exists(videos) ^ exists(video_codebook_ids), 'either raw video or '
        assert not (exists(videos) and not exists(self.cvivit)), 'cvivit must be provided if one wants to encode the videos live during training'
        assert (exists(text_embeds) ^ exists(texts)) ^ self.unconditional, 'either raw text of text embeds must be given, and if unconditional, none should be given'

        assert not (exists(text_embeds) and text_embeds.shape[-1] != self.text_embed_dim), 'text embedding dimension is not correct'

        if not exists(video_codebook_ids):
            assert videos.ndim in {4, 5}

            if videos.ndim == 4:
                videos = rearrange(videos, 'b c h w -> b c 1 h w')

            with torch.no_grad():
                self.cvivit.eval()
                video_codebook_ids = self.cvivit(videos, return_only_codebook_ids = True)

        # derive text embeddings, mask, conditional dropout

        text_mask = None
        cond_drop_prob = 0

        if not self.unconditional:
            if not exists(text_embeds):
                with torch.no_grad():
                    text_embeds = self.encode_texts(texts, output_device = video_codebook_ids.device)

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
            cond_drop_prob = cond_drop_prob,
            video_mask = video_mask,
            text_mask = text_mask,
            context = text_embeds
        )

        return loss

# make video function

@beartype
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
        video = phenaki.sample(texts = text, prime_frames = video_prime, num_frames = scene_num_frames)
        scenes.append(video)

        video_prime = video[:, :, -next_scene_prime_length:]

    return torch.cat(scenes, dim = 2), scenes

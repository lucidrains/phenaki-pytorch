<img src="./phenaki.png" width="450px"></img>

## <a href="https://en.wikipedia.org/wiki/Phenakistiscope">Phenaki</a> - Pytorch (wip)

Implementation of <a href="https://phenaki.video/">Phenaki Video</a>, which uses <a href="https://arxiv.org/abs/2202.04200">Mask GIT</a> to produce text guided videos of up to 2 minutes in length, in Pytorch. It will also combine another technique involving a <a href="https://arxiv.org/abs/2209.04439">token critic</a> for potentially even better generations

Please join <a href="https://discord.gg/xBPBXfcFHd"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a> if you are interested in replicating this work in the open

<a href="https://www.youtube.com/watch?v=RYLomvaPWa4">AI Coffeebreak explanation</a>

## Appreciation

- <a href="https://stability.ai/">Stability.ai</a> for the generous sponsorship to work on cutting edge artificial intelligence research

- <a href="https://huggingface.co/">🤗 Huggingface</a> for their amazing transformers and accelerate library

- <a href="https://github.com/gmegh">Guillem</a> for his ongoing contributions

- You? If you are a great machine learning engineer and / or researcher, feel free to contribute to the frontier of open source generative AI

## Install

```bash
$ pip install phenaki-pytorch
```

## Usage

C-ViViT

```python
import torch
from phenaki_pytorch import CViViT, CViViTTrainer

cvivit = CViViT(
    dim = 512,
    codebook_size = 5000,
    image_size = 256,
    patch_size = 32,
    temporal_patch_size = 2,
    spatial_depth = 4,
    temporal_depth = 4,
    dim_head = 64,
    heads = 8
).cuda()

trainer = CViViTTrainer(
    cvivit,
    folder = '/path/to/images/or/videos',
    batch_size = 4,
    grad_accum_every = 4,
    train_on_images = False,  # you can train on images first, before fine tuning on video, for sample efficiency
    use_ema = False,          # recommended to be turned on (keeps exponential moving averaged cvivit) unless if you don't have enough resources
    num_train_steps = 10000
)

trainer.train()               # reconstructions and checkpoints will be saved periodically to ./results

```

Phenaki

```python
import torch
from phenaki_pytorch import CViViT, MaskGit, Phenaki

cvivit = CViViT(
    dim = 512,
    codebook_size = 5000,
    image_size = (256, 128),  # video with rectangular screen allowed
    patch_size = 32,
    temporal_patch_size = 2,
    spatial_depth = 4,
    temporal_depth = 4,
    dim_head = 64,
    heads = 8
)

cvivit.load('/path/to/trained/cvivit.pt')

maskgit = MaskGit(
    num_tokens = 5000,
    max_seq_len = 1024,
    dim = 512,
    dim_context = 768,
    depth = 6,
)

phenaki = Phenaki(
    cvivit = cvivit,
    maskgit = maskgit
).cuda()

videos = torch.randn(3, 3, 17, 256, 128).cuda() # (batch, channels, frames, height, width)
mask = torch.ones((3, 17)).bool().cuda() # [optional] (batch, frames) - allows for co-training videos of different lengths as well as video and images in the same batch

texts = [
    'a whale breaching from afar',
    'young girl blowing out candles on her birthday cake',
    'fireworks with blue and green sparkles'
]

loss = phenaki(videos, texts = texts, video_frame_mask = mask)
loss.backward()

# do the above for many steps, then ...

video = phenaki.sample(texts = 'a squirrel examines an acorn', num_frames = 17, cond_scale = 5.) # (1, 3, 17, 256, 128)

# so in the paper, they do not really achieve 2 minutes of coherent video
# at each new scene with new text conditioning, they condition on the previous K frames
# you can easily achieve this with this framework as so

video_prime = video[:, :, -3:] # (1, 3, 3, 256, 128) # say K = 3

video_next = phenaki.sample(texts = 'a cat watches the squirrel from afar', prime_frames = video_prime, num_frames = 14) # (1, 3, 14, 256, 128)

# the total video

entire_video = torch.cat((video, video_next), dim = 2) # (1, 3, 17 + 14, 256, 128)

# and so on...
```

Or just import the `make_video` function

```python
# ... above code

from phenaki_pytorch import make_video

entire_video, scenes = make_video(phenaki, texts = [
    'a squirrel examines an acorn buried in the snow',
    'a cat watches the squirrel from a frosted window sill',
    'zoom out to show the entire living room, with the cat residing by the window sill'
], num_frames = (17, 14, 14), prime_lengths = (5, 5))

entire_video.shape # (1, 3, 17 + 14 + 14 = 45, 256, 256)

# scenes - List[Tensor[3]] - video segment of each scene
```

That's it!

## Token Critic

A <a href="https://arxiv.org/abs/2209.04439">new paper</a> suggests that instead of relying on the predicted probabilities of each token as a measure of confidence, one can train an extra critic to decide what to iteratively mask during sampling. You can optionally train this critic for potentially better generations as shown below

```python
import torch
from phenaki_pytorch import CViViT, MaskGit, TokenCritic, PhenakiCritic

cvivit = CViViT(
    dim = 512,
    codebook_size = 5000,
    image_size = (256, 128),
    patch_size = 32,
    temporal_patch_size = 2,
    spatial_depth = 4,
    temporal_depth = 4,
    dim_head = 64,
    heads = 8
)

maskgit = MaskGit(
    num_tokens = 5000,
    max_seq_len = 1024,
    dim = 512,
    dim_context = 768,
    depth = 6,
)

critic = TokenCritic(
    num_tokens = 5000,
    max_seq_len = 1024,
    dim = 512,
    dim_context = 768,
    depth = 6
)

critic_trainer = PhenakiCritic(
    maskgit = maskgit,
    critic = critic,
    cvivit = cvivit
).cuda()

texts = [
    'a whale breaching from afar',
    'young girl blowing out candles on her birthday cake',
    'fireworks with blue and green sparkles'
]

videos = torch.randn(3, 3, 3, 256, 128).cuda() # (batch, channels, frames, height, width)

loss = critic_trainer(videos = videos, texts = texts)
loss.backward()
```

Then just pass the critic to `Phenaki`

```python

phenaki = Phenaki(
    cvivit = cvivit,
    maskgit = maskgit,
    critic = critic
).cuda()

```

Now your generations should be greatly improved (but who knows, since this is only a month old research)

## Phenaki Trainer

This repository will also endeavor to allow the researcher to train on text-to-image and then text-to-video. Similarly, for unconditional training, the researcher should be able to first train on images and then fine tune on video. Below is an example for text-to-video


```python
import torch
from torch.utils.data import Dataset
from phenaki_pytorch import CViViT, MaskGit, Phenaki, PhenakiTrainer

cvivit = CViViT(
    dim = 512,
    codebook_size = 5000,
    image_size = 256,
    patch_size = 32,
    temporal_patch_size = 2,
    spatial_depth = 4,
    temporal_depth = 4,
    dim_head = 64,
    heads = 8
)

cvivit.load('/path/to/trained/cvivit.pt')

maskgit = MaskGit(
    num_tokens = 5000,
    max_seq_len = 1024,
    dim = 512,
    dim_context = 768,
    depth = 6,
    unconditional = False
)

phenaki = Phenaki(
    cvivit = cvivit,
    maskgit = maskgit
).cuda()

# mock text video dataset
# you will have to extend your own, and return the (<video tensor>, <caption>) tuple

class MockTextVideoDataset(Dataset):
    def __init__(
        self,
        length = 100,
        image_size = 256,
        num_frames = 17
    ):
        super().__init__()
        self.num_frames = num_frames
        self.image_size = image_size
        self.len = length

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        video = torch.randn(3, self.num_frames, self.image_size, self.image_size)
        caption = 'video caption'
        return video, caption

dataset = MockTextVideoDataset()

# pass in the dataset

trainer = PhenakiTrainer(
    phenaki = phenaki,
    batch_size = 4,
    grad_accum_every = 4,
    train_on_images = False, # if your mock dataset above return (images, caption) pairs, set this to True
    dataset = dataset,       # pass in your dataset here
    sample_texts_file_path = '/path/to/captions.txt' # each caption should be on a new line, during sampling, will be randomly drawn
)

trainer.train()
```

Token critic training is similarly

```python
import torch
from torch.utils.data import Dataset
from phenaki_pytorch import CViViT, MaskGit, Phenaki, PhenakiTrainer

cvivit = CViViT(
    dim = 512,
    codebook_size = 5000,
    image_size = 256,
    patch_size = 32,
    temporal_patch_size = 2,
    spatial_depth = 4,
    temporal_depth = 4,
    dim_head = 64,
    heads = 8
)

cvivit.load('/path/to/trained/cvivit.pt')

maskgit = MaskGit(
    num_tokens = 5000,
    max_seq_len = 1024,
    dim = 512,
    dim_context = 768,
    depth = 6,
    unconditional = False
)

critic = TokenCritic(
    num_tokens = 5000,
    max_seq_len = 1024,
    dim = 512,
    dim_context = 768,
    depth = 6
)

phenaki_critic = PhenakiCritic(
    maskgit = maskgit,
    critic = critic,
    cvivit = cvivit
).cuda()

# mock text video dataset
# you will have to extend your own, and return the (<video tensor>, <caption>) tuple

class MockTextVideoDataset(Dataset):
    def __init__(
        self,
        length = 100,
        image_size = 256,
        num_frames = 17
    ):
        super().__init__()
        self.num_frames = num_frames
        self.image_size = image_size
        self.len = length

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        video = torch.randn(3, self.num_frames, self.image_size, self.image_size)
        caption = 'video caption'
        return video, caption

dataset = MockTextVideoDataset()

# pass in the dataset

trainer = PhenakiCriticTrainer(
    phenaki_critic = phenaki_critic,
    batch_size = 4,
    grad_accum_every = 4,
    train_on_images = False, # if your mock dataset above return (images, caption) pairs, set this to True
    dataset = dataset        # pass in your dataset here
)

trainer.train()
```

Unconditional is as follows

ex. unconditional images and video training

```python
import torch
from phenaki_pytorch import CViViT, MaskGit, Phenaki, PhenakiTrainer

cvivit = CViViT(
    dim = 512,
    codebook_size = 5000,
    image_size = 256,
    patch_size = 32,
    temporal_patch_size = 2,
    spatial_depth = 4,
    temporal_depth = 4,
    dim_head = 64,
    heads = 8
)

cvivit.load('/path/to/trained/cvivit.pt')

maskgit = MaskGit(
    num_tokens = 5000,
    max_seq_len = 1024,
    dim = 512,
    dim_context = 768,
    depth = 6,
    unconditional = False
)

phenaki = Phenaki(
    cvivit = cvivit,
    maskgit = maskgit
).cuda()

# pass in the folder to images or video

trainer = PhenakiTrainer(
    phenaki = phenaki,
    batch_size = 4,
    grad_accum_every = 4,
    train_on_images = True,                # for sake of example, bottom is folder of images
    dataset = '/path/to/images/or/video'
)

trainer.train()
```

## Todo

- [x] pass mask probability into maskgit and auto-mask and get cross entropy loss
- [x] cross attention + get t5 embeddings code from imagen-pytorch and get classifier free guidance wired up
- [x] wire up full vqgan-vae for c-vivit, just take what is in parti-pytorch already, but make sure to use a stylegan discriminator as said in paper
- [x] complete token critic training code
- [x] complete first pass of maskgit scheduled sampling + token critic (optionally without if researcher does not want to do extra training)
- [x] inference code that allows for sliding time + conditioning on K past frames
- [x] alibi pos bias for temporal attention
- [x] give spatial attention the most powerful positional bias
- [x] make sure to use stylegan-esque discriminator
- [x] 3d relative positional bias for maskgit
- [x] make sure maskgit can also support training of images, and make sure it works on local machine
- [x] also build option for token critic to be conditioned with the text
- [x] should be able to train for text to image generation first
- [x] make sure critic trainer can take in cvivit and automatically pass in video patch shape for relative positional bias - make sure critic also gets optimal relative positional bias
- [x] training code for cvivit
- [x] move cvivit into own file
- [x] unconditional generations (both video and images)
- [x] wire up accelerate for multi-gpu training for both c-vivit and maskgit
- [x] add depthwise-convs to cvivit for position generating
- [x] some basic video manipulation code, allow for sampled tensor to be saved as gif
- [x] basic critic training code
- [x] add position generating dsconv to maskgit too
- [x] outfit customizable self attention blocks to stylegan discriminator
- [x] add all top of the line research for stabilizing transformers training

- [ ] get some basic critic sampling code, show comparison of with and without critic
- [ ] bring in concatenative token shift (temporal dimension)
- [ ] add a DDPM upsampler, either port from imagen-pytorch or just rewrite a simple version here
- [ ] take care of masking in maskgit
- [ ] test maskgit + critic alone on oxford flowers dataset
- [ ] support rectangular sized videos
- [ ] add flash attention as an option for all transformers and cite @tridao

## Citations

```bibtex
@article{Villegas2022PhenakiVL,
    title   = {Phenaki: Variable Length Video Generation From Open Domain Textual Description},
    author  = {Ruben Villegas and Mohammad Babaeizadeh and Pieter-Jan Kindermans and Hernan Moraldo and Han Zhang and Mohammad Taghi Saffar and Santiago Castro and Julius Kunze and D. Erhan},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2210.02399}
}
```

```bibtex
@article{Chang2022MaskGITMG,
    title   = {MaskGIT: Masked Generative Image Transformer},
    author  = {Huiwen Chang and Han Zhang and Lu Jiang and Ce Liu and William T. Freeman},
    journal = {2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year    = {2022},
    pages   = {11305-11315}
}
```

```bibtex
@article{Lezama2022ImprovedMI,
    title   = {Improved Masked Image Generation with Token-Critic},
    author  = {Jos{\'e} Lezama and Huiwen Chang and Lu Jiang and Irfan Essa},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2209.04439}
}
```

```bibtex
@misc{ding2021cogview,
    title   = {CogView: Mastering Text-to-Image Generation via Transformers},
    author  = {Ming Ding and Zhuoyi Yang and Wenyi Hong and Wendi Zheng and Chang Zhou and Da Yin and Junyang Lin and Xu Zou and Zhou Shao and Hongxia Yang and Jie Tang},
    year    = {2021},
    eprint  = {2105.13290},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{shazeer2020glu,
    title   = {GLU Variants Improve Transformer},
    author  = {Noam Shazeer},
    year    = {2020},
    url     = {https://arxiv.org/abs/2002.05202}
}
```

```bibtex
@misc{press2021ALiBi,
    title   = {Train Short, Test Long: Attention with Linear Biases Enable Input Length Extrapolation},
    author  = {Ofir Press and Noah A. Smith and Mike Lewis},
    year    = {2021},
    url     = {https://ofir.io/train_short_test_long.pdf}
}
```

```bibtex
@article{Liu2022SwinTV,
    title   = {Swin Transformer V2: Scaling Up Capacity and Resolution},
    author  = {Ze Liu and Han Hu and Yutong Lin and Zhuliang Yao and Zhenda Xie and Yixuan Wei and Jia Ning and Yue Cao and Zheng Zhang and Li Dong and Furu Wei and Baining Guo},
    journal = {2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year    = {2022},
    pages   = {11999-12009}
}
```

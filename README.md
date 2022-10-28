<img src="./phenaki.png" width="450px"></img>

## <a href="https://en.wikipedia.org/wiki/Phenakistiscope">Phenaki</a> - Pytorch (wip)

Implementation of <a href="https://phenaki.video/">Phenaki Video</a>, which uses <a href="https://arxiv.org/abs/2202.04200">Mask GIT</a> to produce text guided videos of up to 2 minutes in length, in Pytorch. It will also combine another technique involving a <a href="https://arxiv.org/abs/2209.04439">token critic</a> for potentially even better generations

## Install

```bash
$ pip install phenaki-pytorch
```

## Usage

C-ViViT

```python
import torch
from phenaki_pytorch import CViViT

cvivit = CViViT(
    dim = 512,
    codebook_size = 5000,
    patch_size = 32,
    temporal_patch_size = 2,
    spatial_depth = 4,
    temporal_depth = 4,
    dim_head = 64,
    heads = 8
).cuda()

video = torch.randn(1, 3, 17, 256, 256).cuda() # (batch, channels, frames + 1 leading frame, image height, image width)

loss = cvivit(video)
loss.backward()
```

Training the Token Critic, which vastly improves the generation results

```python
import torch
from phenaki_pytorch import CViViT, MaskGit, TokenCritic, CriticTrainer

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

critic_trainer = CriticTrainer(
    maskgit = maskgit,
    critic = critic
)

video_codes = torch.randint(0, 5000, (4, 1024))

loss = critic_trainer(video_codes)
loss.backward()
```

Phenaki

```python
import torch
from phenaki_pytorch import CViViT, MaskGit, Phenaki

maskgit = MaskGit(
    num_tokens = 5000,
    max_seq_len = 1024,
    dim = 512,
    dim_context = 768,
    depth = 6,
)

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

phenaki = Phenaki(
    cvivit = cvivit,
    maskgit = maskgit
).cuda()

videos = torch.randn(3, 3, 17, 256, 256).cuda() # (batch, channels, frames, height, width)

texts = [
    'a whale breaching from afar',
    'young girl blowing out candles on her birthday cake',
    'fireworks with blue and green sparkles'
]

loss = phenaki(videos, texts)
loss.backward()

# do the above for many steps, then ...

video = phenaki.sample(text = 'a squirrel examines an acorn', num_frames = 17, cond_scale = 5.) # (1, 3, 17, 256, 256)

# so in the paper, they do not really achieve 2 minutes of coherent video, more research needed there
# they condition on the previous K frames, with new text conditoining
# you can easily achieve this with this framework as so

video_prime = video[:, :, -3:] # (1, 3, 3, 256, 256) # say K = 3

video_next = phenaki.sample(text = 'a cat watches the squirrel from afar', prime_frames = video_prime, num_frames = 14) # (1, 3, 14, 256, 256)

# the total video

entire_video = torch.cat((video, video_next), dim = 2) # (1, 3, 17 + 14, 256, 256)

# and so on...
```

- [ ] todo, add a master sampler class that allows one to pass in all the text, how long each scene lasts, and stitch together the entire video

## Appreciation

- <a href="https://stability.ai/">Stability.ai</a> for the generous sponsorship to work on cutting edge artificial intelligence research

- <a href="https://huggingface.co/">🤗 Huggingface</a> for their amazing transformers and accelerate library

## Todo

- [x] pass mask probability into maskgit and auto-mask and get cross entropy loss
- [x] cross attention + get t5 embeddings code from imagen-pytorch and get classifier free guidance wired up
- [x] wire up full vqgan-vae for c-vivit, just take what is in parti-pytorch already, but make sure to use a stylegan discriminator as said in paper
- [x] complete token critic training code
- [x] complete first pass of maskgit scheduled sampling + token critic (optionally without if researcher does not want to do extra training)

- [ ] inference code that allows for sliding time + conditioning on K past frames
- [ ] wire up best positional embeddings for all attention
- [ ] wire up accelerate for multi-gpu training for both c-vivit and maskgit
- [ ] some basic video manipulation code, allow for sampled tensor to be saved as gif
- [ ] make sure maskgit can also support training of images, and make sure it works on local machine
- [ ] training code for cvivit
- [ ] make sure to use stylegan-esque discriminator
- [ ] also build option for token critic to be conditioned with the text
- [ ] add all top of the line research for stabilizing transformers training
- [ ] could the critic in turn be used to improve maskgit further with extra adversarial loss?
- [ ] test maskgit + critic alone on oxford flowers dataset
- [ ] bring in the <a href="https://github.com/lucidrains/mega-pytorch">learned multi-headed EMA</a> across time

## Citations

```bibtex
@inproceedings{
anonymous2023phenaki,
    title   = {Phenaki: Variable Length Video Generation from Open Domain Textual Descriptions},
    author  = {Anonymous},
    booktitle = {Submitted to The Eleventh International Conference on Learning Representations },
    year    = {2023},
    url     = {https://openreview.net/forum?id=vOEXS39nOF},
    note    = {under review}
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

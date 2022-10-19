<img src="./phenaki.png" width="450px"></img>

## <a href="https://en.wikipedia.org/wiki/Phenakistiscope">Phenaki</a> - Pytorch (wip)

Implementation of <a href="https://phenaki.video/">Phenaki Video</a>, which uses <a href="https://arxiv.org/abs/2202.04200">Mask GIT</a> to produce text guided videos of up to 2 minutes in length, in Pytorch. It will also combine another <a href="https://arxiv.org/abs/2209.04439">adversarial technique</a> for potentially even better generations

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
)

video = torch.randn(1, 3, 17, 256, 256) # (batch, channels, frames + 1 leading frame, image height, image width)

loss = cvivit(video, return_loss = True)
loss.backward()
```

MaskGit

```python
import torch
from phenaki_pytorch import MaskGit

model = MaskGit(
    num_tokens = 5000,
    max_seq_len = 1024,
    dim = 512,
    depth = 6,
)

x = torch.randint(0, 5000, (1, 1024))
logits = model(x)
```

## Appreciation

- <a href="https://stability.ai/">Stability.ai</a> for the generous sponsorship to work on cutting edge artificial intelligence research

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

import math
import copy
from pathlib import Path
from random import random, choices
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

from beartype import beartype
from typing import Optional, List, Iterable

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset

from torch.optim import Adam

from torchvision import transforms as T
from torchvision.utils import make_grid, save_image

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm

from phenaki_pytorch.optimizer import get_optimizer
from accelerate import Accelerator

from phenaki_pytorch.phenaki_pytorch import Phenaki

from phenaki_pytorch.data import ImageDataset, VideoDataset, video_tensor_to_gif, DataLoader

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def elements_to_device_if_tensor(arr, device):
    output = []
    for el in arr:
        if isinstance(el, torch.Tensor):
            el = el.to(device)
        output.append(el)
    return output

def split_iterable(it, split_size):
    accum = []
    for ind in range(math.ceil(len(it) / split_size)):
        start_index = ind * split_size
        accum.append(it[start_index: (start_index + split_size)])
    return accum

def split(t, split_size = None):
    if not exists(split_size):
        return t

    if isinstance(t, torch.Tensor):
        return t.split(split_size, dim = 0)

    if isinstance(t, Iterable):
        return split_iterable(t, split_size)

    return TypeError

def find_first(cond, arr):
    for el in arr:
        if cond(el):
            return el
    return None

def split_args_and_kwargs(*args, batch_size = None, split_size = None, **kwargs):
    all_args = (*args, *kwargs.values())
    len_all_args = len(all_args)

    if not exists(batch_size):
        first_tensor = find_first(lambda t: isinstance(t, torch.Tensor), all_args)
        assert exists(first_tensor)
        batch_size = len(first_tensor)

    split_size = default(split_size, batch_size)
    num_chunks = math.ceil(batch_size / split_size)

    dict_len = len(kwargs)
    dict_keys = kwargs.keys()
    split_kwargs_index = len_all_args - dict_len

    split_all_args = [split(arg, split_size = split_size) if exists(arg) and isinstance(arg, (torch.Tensor, Iterable)) else ((arg,) * num_chunks) for arg in all_args]
    chunk_sizes = tuple(map(len, split_all_args[0]))

    for (chunk_size, *chunked_all_args) in tuple(zip(chunk_sizes, *split_all_args)):
        chunked_args, chunked_kwargs_values = chunked_all_args[:split_kwargs_index], chunked_all_args[split_kwargs_index:]
        chunked_kwargs = dict(tuple(zip(dict_keys, chunked_kwargs_values)))
        chunk_size_frac = chunk_size / batch_size
        yield chunk_size_frac, (chunked_args, chunked_kwargs)

def simple_slugify(text, max_length = 255):
    return text.replace('-', '_').replace(',', '').replace(' ', '_').replace('|', '--').strip('-_')[:max_length]

# trainer class

@beartype
class PhenakiTrainer(object):
    def __init__(
        self,
        phenaki: Phenaki,
        *,
        folder = None,
        train_on_images = False,
        batch_size = 16,
        grad_accum_every = 1,
        num_frames = 17,
        sample_num_frames = None,
        train_lr = 1e-4,
        train_num_steps = 100000,
        max_grad_norm = None,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        wd = 0,
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        fp16 = False,
        split_batches = True,
        convert_image_to = None,
        sample_texts: Optional[List[str]] = None,
        dataset = None,
        dataset_fields = ('videos', 'texts', 'video_frame_masks')
    ):
        super().__init__()
        maskgit = phenaki.maskgit
        cvivit = phenaki.cvivit

        assert exists(cvivit)

        assert maskgit.unconditional or exists(sample_texts), 'if maskgit is to be trained text conditioned, `sample_texts` List[str] must be given'

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        self.model = phenaki

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.sample_texts = sample_texts
        self.unconditional = maskgit.unconditional

        self.save_and_sample_every = save_and_sample_every

        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps
        self.image_size = phenaki.cvivit.image_size

        # dataset and dataloader

        dataset_klass = ImageDataset if train_on_images else VideoDataset

        self.sample_num_frames = default(sample_num_frames, num_frames)
        self.train_on_images = train_on_images

        if dataset:
            self.ds = dataset
        elif train_on_images:
            assert exists(folder)
            self.ds = ImageDataset(folder, self.image_size)
        else:
            assert exists(folder)
            self.ds = VideoDataset(folder, self.image_size, num_frames = num_frames)

        dl = DataLoader(self.ds, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        self.dataset_fields = dataset_fields

        # optimizer

        self.opt = get_optimizer(maskgit.parameters(), lr = train_lr, wd = wd, betas = adam_betas)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents = True, exist_ok = True)

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train_step(self):
        accelerator = self.accelerator
        device = self.device

        total_loss = 0.

        for _ in range(self.grad_accum_every):
            data = next(self.dl)
            data = elements_to_device_if_tensor(data, device)
            data_kwargs = dict(zip(self.dataset_fields, data))

            assert not (self.train_on_images and data_kwargs['videos'].ndim != 4), 'you have it set to train on images, but the dataset is not returning tensors of 4 dimensions (batch, channels, height, width)'

            with self.accelerator.autocast():
                loss = self.model(**data_kwargs)
                loss = loss / self.grad_accum_every
                total_loss += loss.item()

            self.accelerator.backward(loss)

        if exists(self.max_grad_norm):
            accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        accelerator.wait_for_everyone()

        self.opt.step()
        self.opt.zero_grad()

        accelerator.wait_for_everyone()

        if self.is_main and self.step % self.save_and_sample_every == 0:
            self.model.eval()
            milestone = self.step // self.save_and_sample_every

            # whether to pass in texts or not

            sample_kwargs = dict()

            if not self.unconditional:
                texts = choices(self.sample_texts, k = self.num_samples)
            else:
                texts = (None,) * self.num_samples

            sample_kwargs = {'texts': texts}

            # method to call

            method_name = 'sample_images' if self.train_on_images else 'sample'
            sample_method = getattr(self.model, method_name)

            # evaluate in groups, splitting the kwargs appropriately

            with torch.no_grad():
                groups = num_to_groups(self.num_samples, self.batch_size)
                args_kwargs_iter = split_args_and_kwargs(batch_size = self.num_samples, split_size = self.batch_size, **sample_kwargs)

                all_sampled = []
                for group_batch_size, (_, (_, kwargs)) in zip(groups, args_kwargs_iter):
                    _kwargs = kwargs if not self.unconditional else dict()
                    sampled = sample_method(num_frames = self.sample_num_frames, batch_size = group_batch_size, **_kwargs)
                    all_sampled.append(sampled)

            # save video and images differently

            if not self.train_on_images:
                sampled_videos = torch.cat(all_sampled, dim = 0)
                milestone_folder = self.results_folder / f'videos.{milestone}'
                milestone_folder.mkdir(parents = True, exist_ok = True)

                for ind, (video_tensor, video_caption) in enumerate(zip(sampled_videos.unbind(dim = 0), texts)):
                    slugged_video_caption = simple_slugify(video_caption) if exists(video_caption) else str(ind)
                    video_tensor_to_gif(video_tensor, str(milestone_folder / f'{slugged_video_caption}.gif'))
            else:
                nrows = int(math.sqrt(self.num_samples))

                sampled_images = sampled_videos.detach().cpu().float().clamp(0., 1.)
                grid = make_grid(sampled_images, nrow = nrows, normalize = True, value_range = (0, 1))

                save_image(grid, str(self.results_folder / f'{milestone}.png'))

            # save checkpoints

            self.save(milestone)

        self.step += 1
        return total_loss

    def train(self):

        with tqdm(
            initial = self.step,
            total = self.train_num_steps,
            disable = not self.is_main
        ) as pbar:

            while self.step < self.train_num_steps:
                loss = self.train_step()

                pbar.set_description(f'loss: {loss:.4f}')
                pbar.update(1)

        self.print('training complete')

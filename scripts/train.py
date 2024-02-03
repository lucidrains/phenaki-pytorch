import os
import argparse
from omegaconf import OmegaConf
import torch
from phenaki_pytorch import CViViT, CViViTTrainer, MaskGit, Phenaki



def main(config):
    cvivit = CViViT(
        dim = 512,
        codebook_size = 65536,
        image_size = (256, 256),
        patch_size = 32,
        temporal_patch_size = 2,
        spatial_depth = 4,
        temporal_depth = 4,
        dim_head = 64,
        heads = 8
    ).cuda()
    if config['vqvae_from_pretrained'] is None:
        trainer = CViViTTrainer(
            cvivit,
            folder = config['data_folder'],
            batch_size = config['batch_size'],
            grad_accum_every = config['grad_accum_every'],
            train_on_images = config['train_on_images'],  # you can train on images first, before fine tuning on video, for sample efficiency
            use_ema = config['use_ema'],          # recommended to be turned on (keeps exponential moving averaged cvivit) unless if you don't have enough resources
            num_train_steps = config['num_train_steps']
        )
        trainer.train()               # reconstructions and checkpoints will be saved periodically to ./results
    else:
        model_path = os.path.expanduser(f"~/.cache/Appimate")
        cvivit.load(model_path)

    """
        Train the Phenaki Model.
    """
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


    videos = torch.randn(3, 3, 17, 256, 256)#.cuda() # (batch, channels, frames, height, width)
    mask = torch.ones((3, 17)).bool()#.cuda() # [optional] (batch, frames) - allows for co-training videos of different lengths as well as video and images in the same batch
    texts = [
        'a whale breaching from afar',
        'young girl blowing out candles on her birthday cake',
        'fireworks with blue and green sparkles'
    ]
    for epoch in range(0, config['num_epochs']):
        loss = phenaki(videos, texts = texts, video_frame_mask = mask)
        loss.backward()
        # do the above for many steps, then ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../configs/main_config.yaml")
    parser.add_argument("--use_wandb", type=bool, default=False)
    parser.add_argument("--experiment_num", type=int, default=1, required=True)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)
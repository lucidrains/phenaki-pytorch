import torch
from phenaki_pytorch import CViViT, MaskGit, Phenaki

def main(cfg):
    cvivit = CViViT(
        dim = 512,
        codebook_size = 65536,
        image_size = (256, 128),  # video with rectangular screen allowed
        patch_size = 32,
        temporal_patch_size = 2,
        spatial_depth = 4,
        temporal_depth = 4,
        dim_head = 64,
        heads = 8
    )
    # cvivit.load('/path/to/trained/cvivit.pt')
    cvivit.load(cfg.path_to_cvivit)
    phenaki = Phenaki(
        cvivit = cvivit,
    ).cuda()
    video = phenaki.sample(texts = cfg.prompt, num_frames = 17, cond_scale = 5.) # (1, 3, 17, 256, 128)
    video_prime = video[:, :, -3:] # (1, 3, 3, 256, 128) # say K = 3
    video_next = phenaki.sample(texts = 'a cat watches the squirrel from afar', prime_frames = video_prime, num_frames = 14) # (1, 3, 14, 256, 128)
    entire_video = torch.cat((video, video_next), dim = 2) # (1, 3, 17 + 14, 256, 128)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='A simple distributed inference job')
    parser.add_argument('--prompt', default="", type=str, help='The text that model should condition from.')
    parser.add_argument('--path_to_cvivit', default=None, type=str, help='The name of the pretrained model')
    # parser.add_argument('--path_to_phenaki', default=None, type=str, help='The name of the pretrained model')
    args = parser.parse_args()
    main(args)
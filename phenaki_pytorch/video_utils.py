import torch
from torchvision import transforms as T

import cv2
import numpy as np
from einops import rearrange

# handle reading and writing gif

CHANNELS_TO_MODE = {
    1 : 'L',
    3 : 'RGB',
    4 : 'RGBA'
}

def seek_all_images(img, channels = 3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1

# tensor of shape (channels, frames, height, width) -> gif

def video_tensor_to_gif(
    tensor,
    path,
    duration = 120,
    loop = 0,
    optimize = True
):
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images

# gif -> (channels, frame, height, width) tensor

def gif_to_tensor(
    path,
    channels = 3,
    transform = T.ToTensor()
):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels = channels)))
    return torch.stack(tensors, dim = 1)

# handle reading and writing mp4

def video_to_tensor(
    path,
    num_frames = -1,
    crop = False,
    crop_size = 256
):
    '''
    Generate Pytorch tensor from .mp4 video

    Parameters
    ----------
    path : str
        Path of the video to be imported

    num_frames : int, optional
        Number of frames to be stored in the output tensor
        
    crop: Bool, optional
        Crop video to be size (256, 256)

    Returns
    -------
    tensor
        Video tensor with shape (1, C, F, H, W)
        C - channels
        F - number of frames
        H - video height
        W - video width
    '''

    # Import the video and cut it into frames.
    video = cv2.VideoCapture(path)

    frames = []
    check = True

    while check:
        check, frame = video.read()
        if crop:
            frame = crop_center(frame, crop_size, crop_size)

        if check:
            frames.append(rearrange(frame, '... -> 1 ...'))

    frames = np.array(np.concatenate(frames[:-1], axis = 0))  # convert list of frames to numpy array
    frames = rearrange(frames, 'f h w c -> 1 c f h w')

    frames_torch = torch.tensor(frames).float()
    
    return frames_torch[:, :, :num_frames, :, :]

def tensor_to_video(
    tensor,
    path,
    fps = 25,
    video_format = 'MP4V'
):
    '''
    Generate .mp4 from Pytorch video tensor

    Parameters
    ----------
    tensor : tensor
        Pytorch video tensor
        
    path : str
        Path of the video to be saved

    fps : int, optional
        Frames per second for the saved video

    Returns
    -------
    video
        
    '''
    # Import the video and cut it into frames.
    tensor = tensor.cpu()
    
    num_frames, height, width = tensor.shape[-3:]
    
    fourcc = cv2.VideoWriter_fourcc(*video_format) # Changes in this line can allow for different video formats.
    video = cv2.VideoWriter(path, fourcc, fps, (width, height))

    frames = []

    for idx in range(num_frames):
        numpy_frame = tensor[0, :, idx, :, :].numpy()
        numpy_frame = np.uint8(rearrange(numpy_frame, 'c h w -> h w c'))
        video.write(numpy_frame)
    
    video.release()

    cv2.destroyAllWindows()
    
    return video

def crop_center(
    img,
    cropx,
    cropy
):
    '''
    Crop image

    Parameters
    ----------
    img : vector
        
    cropx : int
        Length of the final image in the x direction.

    cropy : int
        Length of the final image in the y direction.

    Returns
    -------
    cropped_image : vector
        
    '''
    try:
        y,x,c = img.shape
        startx = x // 2 - cropx // 2
        starty = y // 2 - cropy // 2
        return img[starty:starty+cropy, startx:startx+cropx, :]
    except: 
        pass

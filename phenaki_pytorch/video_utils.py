import torch
from torchvision import transforms as T

import cv2
import numpy as np
from einops import rearrange

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
    check = True

    for idx in range(num_frames):
        frame = np.uint8(np.transpose(tensor[0, :, idx, :, :].numpy().T, (1, 0, 2))) # refactor this with einops
        video.write(frame)
    
    video.release()

    # Closes all the frames
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
        startx = x//2 - cropx//2
        starty = y//2 - cropy//2    
        return img[starty:starty+cropy, startx:startx+cropx, :]
    except: 
        pass

import torch
import glob
import numpy as np
import cv2
from torchvision import transforms as T

def fromVideoToTensor(path, num_frames=-1, crop = False):
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
            frame = crop_center(frame, 256, 256)
        frames.append(np.expand_dims(frame, axis=0))

    frames = np.array(np.concatenate(frames[:-1],axis=0))  # convert list of frames to numpy array
    frames_torch = torch.tensor(np.expand_dims(np.transpose(frames, (3,0,1,2)), axis=0 )).float()
    
    return frames_torch[:, :, :num_frames, :, :]

def fromTensorToVideo(tensor, path, fps = 25):
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
    
    frame_width = int(tensor.shape[4])
    frame_height = int(tensor.shape[3])
    
    fourcc = cv2.VideoWriter_fourcc(*'MP4V') # Changes in this line can allow for different video formats.
    video = cv2.VideoWriter(path, fourcc, fps, (frame_width, frame_height))

    frames = []
    check = True

    for idx in range(tensor.shape[2]):
        frame = np.uint8(np.transpose(tensor[0, :, idx, :, :].numpy().T, (1, 0, 2)))
        video.write(frame)
    
    video.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    
    return video

def crop_center(img, cropx, cropy):
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

"""
Assorted functions to make your life easier.
"""
import numpy as np
import torch
from skimage.io import imread

def load_main_movie(movie):
    """
    Load in the specified clip (1, 2, or 3) from the KITTI dataset
    that the PredNet was trained on, in the PredNet's NTCHW form.

    NTCHW = (size of batch, time steps, channels, height, width)
    """
    assert movie in {1, 2, 3}
    # they're named differently on disk
    movie_map = {1 : 3, 2 : 6, 3 : 10}

    frames = torch.zeros((1, 20, 3, 128, 160))
    for i in range(20):
        frame = imread(f'main_movies/mov{movie_map[movie]}_f{i}.png')
        frames[0, i] = torch.tensor(frame.transpose(2, 0, 1))

    return frames

def load_alt_movie(movie):
    """
    Load in the specified movie that PredNet was not trained on
    (movie 1, 2, or 3) in the PredNet's NTCHW form.
    """
    assert movie in {1, 2, 3}
    # they're named differently on disk
    movie_map = {1 : 15, 2 : 28, 3 : 46}

    frames = torch.zeros((1, 50, 3, 200, 200))
    for i in range(50):
        frame = imread(f'alternative_movies/mov{movie_map[movie]}_f{i}.png')
        frames[0, i] = torch.tensor(frame.transpose(2, 0, 1))

    return frames

def load_gratings(movie):
    """
    Loads in the specified sine-wave grating movie
    in the PredNet's NTCHW form. Movies 1-4 do not change
    direction, while movies 5-8 do.
    """
    assert movie in range(1, 9)
    # they're saved by orientation and direction change
    movie_map = {1 : 0,
            2 : 90,
            3 : 180,
            4 : 270,
            5 : 'c15',
            6 : 'c30',
            7 : 'c45',
            8 : 'c60'
            }

    frames = torch.zeros((1, 20, 3, 128, 128))
    for i in range(20):
        frame = imread(f'gratings/grating_{movie_map[movie]}_{i}.png')
        frames[0, i] = torch.tensor(frame.transpose(2, 0, 1))

    return frames

def extrapolate_prednet(network, starting_sequence, num_frames):
    """
    Use the PredNet to predict multiple following frames by
    giving it its input as 

    Arguments:
        network: the PredNet object, with weights loaded, to use

        starting_sequence: the tensor of ground-truth frames for the
        network to see before it starts extrapolating

        num_frames: the number of frames to extrapolate out

    Returns:
        a tensor containing the predicted frames
    """
    assert network.output_mode == 'prediction' # need this to get the frames

    sequence = starting_sequence
    nt = sequence.shape[1]
    predicted_frames = torch.zeros(sequence.shape[0], num_frames,
            *sequence.shape[2:])

    for frame in range(num_frames):
        with torch.no_grad():
            prediction = network(sequence)
        predicted_frames[:, frame] = prediction[:, nt-1]

        next_sequence = torch.zeros(sequence.shape)
        next_sequence[:, :nt-1] = sequence[:, 1:]
        next_sequence[:, nt-1] = prediction[:, nt-1]

        sequence = next_sequence

    return predicted_frames


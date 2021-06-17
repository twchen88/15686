"""
Some examples of how to use the PredNet,
and PCA in Python.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.io import imsave
from sklearn.decomposition import PCA
from prednet import PredNet
from helper_functions import *

# these are parameters setting the size of the network
# you can't change them unless you train a new one
A_channels = (3, 48, 96, 192)
R_channels = (3, 48, 96, 192)

# create the PredNet model
# specifying different output modes will change what it returns
# you can look at prednet.py to see the options (prediction, H, R, and E)
model = PredNet(R_channels, A_channels, output_mode='H')
# you can change the output mode later too
model.output_mode = 'prediction'
# for now, it has randomly initialized weights

# load in the trained weights
# you need these to get sensible results
weights = torch.load('models/prednet_1.pt', map_location='cpu')
# and put them into the model
model.load_state_dict(weights)

# now you can run the model on some movies
# for instance, the first movie from the KITTI dataset
movie = load_main_movie(1)

# the inputs have to follow the NxTxCxHxW shape
# N is the batch size, the number of videos to run in parallel
# larger batches speed things up but take more memory
# T is the number of time steps, C channels, H and W height and width
print(f'Shape of the input movie is: {movie.shape}')

# with the movie loaded in, we can run the model on it
# torch.no_grad speeds things up and saves memory by not computing gradients
# which we don't need since we're not training
with torch.no_grad():
    predictions = model(movie)

# this gives the prediction of the following frame for each frame in the movie
# so the first frame is not predicted (as there's nothing to predict it from)
# and the last frame in the predictions does not have a "ground-truth" value
print(f'Shape of the predicted frames is: {predictions.shape}')

# save the last predicted frame -- need to convert to numpy and shape it HxWxC
imsave('example_prediction.png', predictions[0,-1].detach().numpy().transpose(1, 2, 0))
# if you're dealing with lots of frames, you might find the
# torchvision function "torchvision.utils.make_grid" helpful

# try a different output mode
model.output_mode = 'E' # error at each layer
# the modes E, H, and R return lists of lists
# the outer list is over layers
# the inner list is over time steps (corresponding to each frame of the movie)
# and the elements are tensors of shape NxCxHxW
with torch.no_grad():
    errors = model(movie)
print(f'Number of layers in the model: {len(errors)}')
print(f'Number of timesteps: {len(errors[0])}')
print(f'Shape of error at layer 1 of the model: {errors[0][0].shape}')

# if you want to feed the model's predictions back into it
# to predict multiple frames ahead
# you can look at the "extrapolate_prednet" function in the helper file
# and use it outright or modify it as you see fit

##########################################################
##########################################################
##########################################################

# now try PCA
# the scikit-learn documentation is great but here's an example
# make some random 20-dimensional data
data = np.random.rand(300, 20)

# and run PCA
pca = PCA(n_components=2)
pca.fit(data)
transformed = pca.transform(data)

# look how well it did -- not very well on random data, of course
print(f'Explained variance of first two PCs: {pca.explained_variance_ratio_}')

# and plot the results
# of course, they're just random noise
plt.figure()
# plot each half in a different color, because you can
plt.scatter(transformed[:150, 0], transformed[:150, 1], color='black',
        label='first half of data')
plt.scatter(transformed[150:, 0], transformed[150:, 1], color='red',
        label='second half of data')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('Example PCA plot of random data')
plt.legend() # to show the labels
plt.savefig('pca_example.png')
plt.show()

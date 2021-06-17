import numpy as np
import torch
import matplotlib.pyplot as plt

from skimage.io import imsave
from sklearn.decomposition import PCA
from prednet import PredNet
from helper_functions import *
import torchvision

A_channels = (3, 48, 96, 192)
R_channels = (3, 48, 96, 192)

model = PredNet(R_channels, A_channels, output_mode='H')
model.output_mode = 'prediction'
weights = torch.load('models/prednet_1.pt', map_location='cpu')
model.load_state_dict(weights)

movie1 = load_gratings(1)
movie2 = load_gratings(2)
movie3 = load_gratings(3)
movie4 = load_gratings(4)

with torch.no_grad():
    predictions1 = model(movie1)
    predictions2 = model(movie2)
    predictions3 = model(movie3)
    predictions4 = model(movie4)

truth1 = movie1[0]
grid1 = torchvision.utils.make_grid(truth1, nrow = 5)
imsave('part2_1_truth1.png', grid1.detach().numpy().transpose(1, 2, 0))

truth2 = movie2[0]
grid2 = torchvision.utils.make_grid(truth2, nrow = 5)
imsave('part2_1_truth2.png', grid2.detach().numpy().transpose(1, 2, 0))

truth3 = movie3[0]
grid3 = torchvision.utils.make_grid(truth3, nrow = 5)
imsave('part2_1_truth3.png', grid3.detach().numpy().transpose(1, 2, 0))

truth4 = movie4[0]
grid4 = torchvision.utils.make_grid(truth4, nrow = 5)
imsave('part2_1_truth4.png', grid4.detach().numpy().transpose(1, 2, 0))

pred1 = predictions1[0]
grid5 = torchvision.utils.make_grid(pred1, nrow = 5)
imsave('part2_1_pred1.png', grid5.detach().numpy().transpose(1, 2, 0))

pred2 = predictions2[0]
grid6 = torchvision.utils.make_grid(pred2, nrow = 5)
imsave('part2_1_pred2.png', grid6.detach().numpy().transpose(1, 2, 0))

pred3 = predictions3[0]
grid7 = torchvision.utils.make_grid(pred3, nrow = 5)
imsave('part2_1_pred3.png', grid7.detach().numpy().transpose(1, 2, 0))

pred4 = predictions4[0]
grid8 = torchvision.utils.make_grid(pred4, nrow = 5)
imsave('part2_1_pred4.png', grid8.detach().numpy().transpose(1, 2, 0))
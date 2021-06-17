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
movie = load_main_movie(1)

output = extrapolate_prednet(model, movie[:, 0:10], 10)

truth = movie[0, 10:20]
grid1 = torchvision.utils.make_grid(truth, nrow = 5)
imsave('part1_5_truth.png', grid1.detach().numpy().transpose(1, 2, 0))

grid2 = torchvision.utils.make_grid(output[0], nrow = 5)
imsave('part1_5_prediction.png', grid2.detach().numpy().transpose(1, 2, 0))

error = []
frames = range(10)
length = 128 * 160

for i in frames:
    diff1 = (output[0][i] - truth[i]).detach().numpy().transpose(1, 2, 0)
    diff_squared1 = diff1 ** 2
    diff_squared_sum1 = sum(sum(sum(diff_squared1)))/3

    error.append(diff_squared_sum1/length)

print(error)

plt.figure(1)
plt.plot(frames, error)
plt.xlabel('frame number')
plt.ylabel('MSE')
plt.title('MSE vs Frame Curve')
plt.savefig('part1_5_error.png')
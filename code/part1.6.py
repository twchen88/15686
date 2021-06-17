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

movie = load_alt_movie(1)
with torch.no_grad():
    predictions = model(movie)

truth = movie[0, 0:10]
grid1 = torchvision.utils.make_grid(truth, nrow = 5)
imsave('part1_6_truth.png', grid1.detach().numpy().transpose(1, 2, 0))
pred = predictions[0, 0:10]
grid2 = torchvision.utils.make_grid(pred, nrow = 5)
imsave('part1_6_pred.png', grid2.detach().numpy().transpose(1, 2, 0))

model.output_mode = 'E'
with torch.no_grad():
    e = model(movie)

error_output = []
error_manual = []
frames = range(1, 20)

length = 128 * 160

for i in frames:
    E1 = sum(e[0][i]).detach().numpy().transpose(1, 2, 0)
    E1 = E1 ** 2
    E1 = sum(sum(sum(E1)))/3
    diff1 = (predictions[0][i-1] - movie[0][i]).detach().numpy().transpose(1, 2, 0)
    diff_squared1 = diff1 ** 2
    diff_squared_sum1 = sum(sum(sum(diff_squared1)))/3

    error_output.append(E1/length)
    error_manual.append(diff_squared_sum1/length)

print(error_manual)
print(error_output)

plt.figure(1)
plt.plot(frames, error_manual)
plt.xlabel('frame number')
plt.ylabel('MSE')
plt.title('MSE vs Frame Curve')
plt.savefig('part1_6_error.png')
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
model.output_mode = 'E'

'''
prednet 1 first
'''
weights = torch.load('models/prednet_1.pt', map_location='cpu')
model.load_state_dict(weights)

movie1 = load_main_movie(1)
movie2 = load_main_movie(2)
movie3 = load_main_movie(3)

with torch.no_grad():
    e1 = model(movie1)
    e2 = model(movie2)
    e3 = model(movie3)

model.output_mode = 'prediction'
with torch.no_grad():
    predictions1 = model(movie1)
    predictions2 = model(movie2)
    predictions3 = model(movie3)

error_output = []
error_manual = []
frames = range(1, 11)

length = 128 * 160

for i in frames:
    E1 = sum(e1[0][i]).detach().numpy().transpose(1, 2, 0)
    E1 = E1 ** 2
    E1 = sum(sum(sum(E1)))/3
    diff1 = (predictions1[0][i-1] - movie1[0][i]).detach().numpy().transpose(1, 2, 0)
    diff_squared1 = diff1 ** 2
    diff_squared_sum1 = sum(sum(sum(diff_squared1)))/3

    E2 = sum(e2[0][i]).detach().numpy().transpose(1, 2, 0)
    E2 = E2 ** 2
    E2 = sum(sum(sum(E2)))/3
    diff2 = (predictions2[0][i-1] - movie2[0][i]).detach().numpy().transpose(1, 2, 0)
    diff_squared2 = diff2 ** 2
    diff_squared_sum2 = sum(sum(sum(diff_squared2)))/3

    E3 = sum(e3[0][i]).detach().numpy().transpose(1, 2, 0)
    E3 = E3 ** 2
    E3 = sum(sum(sum(E3)))/3
    diff3 = (predictions3[0][i-1] - movie3[0][i]).detach().numpy().transpose(1, 2, 0)
    diff_squared3 = diff3 ** 2
    diff_squared_sum3 = sum(sum(sum(diff_squared3)))/3

    error_output.append((E1 + E2 + E3)/(length*3))
    error_manual.append((diff_squared_sum1 + diff_squared_sum2 + diff_squared_sum3)/(length*3))

print(error_manual)
print(error_output)

plt.figure(1)
plt.plot(frames, error_manual)
plt.xlabel('frame number')
plt.ylabel('MSE')
plt.title('MSE vs Frame Curve')
plt.savefig('part1_2_prednet1.png')

model = PredNet(R_channels, A_channels, output_mode='H')
model.output_mode = 'E'

'''
prednet 2
'''
weights = torch.load('models/prednet_2.pt', map_location='cpu')
model.load_state_dict(weights)

with torch.no_grad():
    e1 = model(movie1)
    e2 = model(movie2)
    e3 = model(movie3)

model.output_mode = 'prediction'
with torch.no_grad():
    predictions1 = model(movie1)
    predictions2 = model(movie2)
    predictions3 = model(movie3)

error_output = []
error_manual = []
frames = range(1, 11)

length = 128 * 160

for i in frames:
    E1 = sum(e1[0][i]).detach().numpy().transpose(1, 2, 0)
    E1 = E1 ** 2
    E1 = sum(sum(sum(E1)))/3
    diff1 = (predictions1[0][i-1] - movie1[0][i]).detach().numpy().transpose(1, 2, 0)
    diff_squared1 = diff1 ** 2
    diff_squared_sum1 = sum(sum(sum(diff_squared1)))/3

    E2 = sum(e2[0][i]).detach().numpy().transpose(1, 2, 0)
    E2 = E2 ** 2
    E2 = sum(sum(sum(E2)))/3
    diff2 = (predictions2[0][i-1] - movie2[0][i]).detach().numpy().transpose(1, 2, 0)
    diff_squared2 = diff2 ** 2
    diff_squared_sum2 = sum(sum(sum(diff_squared2)))/3

    E3 = sum(e3[0][i]).detach().numpy().transpose(1, 2, 0)
    E3 = E3 ** 2
    E3 = sum(sum(sum(E3)))/3
    diff3 = (predictions3[0][i-1] - movie3[0][i]).detach().numpy().transpose(1, 2, 0)
    diff_squared3 = diff3 ** 2
    diff_squared_sum3 = sum(sum(sum(diff_squared3)))/3

    error_output.append((E1 + E2 + E3)/(3 *length))
    error_manual.append((diff_squared_sum1 + diff_squared_sum2 + diff_squared_sum3)/(3 * length))

print(error_manual)
print(error_output)

plt.figure(2)
plt.plot(frames, error_manual)
plt.xlabel('frame number')
plt.ylabel('MSE')
plt.title('MSE vs Frame Curve')
plt.savefig('part1_2_prednet2.png')
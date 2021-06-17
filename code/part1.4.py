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

error_layer_1_pred1 = []
error_layer_2_pred1 = []
error_layer_3_pred1 = []
error_layer_4_pred1 = []
frames = range(1, 11)

length = 128 * 160

for i in frames:
    E1 = sum(e1[0][i]).detach().numpy().transpose(1, 2, 0)
    E1 = E1 ** 2
    E1 = sum(sum(sum(E1)))/3

    E2 = sum(e2[0][i]).detach().numpy().transpose(1, 2, 0)
    E2 = E2 ** 2
    E2 = sum(sum(sum(E2)))/3

    E3 = sum(e3[0][i]).detach().numpy().transpose(1, 2, 0)
    E3 = E3 ** 2
    E3 = sum(sum(sum(E3)))/3

    error_layer_1_pred1.append((E1 + E2 + E3)/(length * 3))

for i in frames:
    E1 = sum(e1[1][i]).detach().numpy().transpose(1, 2, 0)
    E1 = E1 ** 2
    E1 = sum(sum(sum(E1)))/3

    E2 = sum(e2[1][i]).detach().numpy().transpose(1, 2, 0)
    E2 = E2 ** 2
    E2 = sum(sum(sum(E2)))/3

    E3 = sum(e3[1][i]).detach().numpy().transpose(1, 2, 0)
    E3 = E3 ** 2
    E3 = sum(sum(sum(E3)))/3

    error_layer_2_pred1.append((E1 + E2 + E3)/(length * 3))

for i in frames:
    E1 = sum(e1[2][i]).detach().numpy().transpose(1, 2, 0)
    E1 = E1 ** 2
    E1 = sum(sum(sum(E1)))/3

    E2 = sum(e2[2][i]).detach().numpy().transpose(1, 2, 0)
    E2 = E2 ** 2
    E2 = sum(sum(sum(E2)))/3

    E3 = sum(e3[2][i]).detach().numpy().transpose(1, 2, 0)
    E3 = E3 ** 2
    E3 = sum(sum(sum(E3)))/3

    error_layer_3_pred1.append((E1 + E2 + E3)/(length * 3))

for i in frames:
    E1 = sum(e1[3][i]).detach().numpy().transpose(1, 2, 0)
    E1 = E1 ** 2
    E1 = sum(sum(sum(E1)))/3

    E2 = sum(e2[3][i]).detach().numpy().transpose(1, 2, 0)
    E2 = E2 ** 2
    E2 = sum(sum(sum(E2)))/3

    E3 = sum(e3[3][i]).detach().numpy().transpose(1, 2, 0)
    E3 = E3 ** 2
    E3 = sum(sum(sum(E3)))/3

    error_layer_4_pred1.append((E1 + E2 + E3)/ (length * 3))


'''
prednet 2
'''
model = PredNet(R_channels, A_channels, output_mode='H')
model.output_mode = 'E'
weights = torch.load('models/prednet_2.pt', map_location='cpu')
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

error_layer_1_pred2 = []
error_layer_2_pred2 = []
error_layer_3_pred2 = []
error_layer_4_pred2 = []
frames = range(1, 11)

length = 128 * 160

for i in frames:
    E1 = sum(e1[0][i]).detach().numpy().transpose(1, 2, 0)
    E1 = E1 ** 2
    E1 = sum(sum(sum(E1)))/3

    E2 = sum(e2[0][i]).detach().numpy().transpose(1, 2, 0)
    E2 = E2 ** 2
    E2 = sum(sum(sum(E2)))/3

    E3 = sum(e3[0][i]).detach().numpy().transpose(1, 2, 0)
    E3 = E3 ** 2
    E3 = sum(sum(sum(E3)))/3

    error_layer_1_pred2.append((E1 + E2 + E3)/(length * 3))

for i in frames:
    E1 = sum(e1[1][i]).detach().numpy().transpose(1, 2, 0)
    E1 = E1 ** 2
    E1 = sum(sum(sum(E1)))/3

    E2 = sum(e2[1][i]).detach().numpy().transpose(1, 2, 0)
    E2 = E2 ** 2
    E2 = sum(sum(sum(E2)))/3

    E3 = sum(e3[1][i]).detach().numpy().transpose(1, 2, 0)
    E3 = E3 ** 2
    E3 = sum(sum(sum(E3)))/3

    error_layer_2_pred2.append((E1 + E2 + E3)/ (length * 3))

for i in frames:
    E1 = sum(e1[2][i]).detach().numpy().transpose(1, 2, 0)
    E1 = E1 ** 2
    E1 = sum(sum(sum(E1)))/3

    E2 = sum(e2[2][i]).detach().numpy().transpose(1, 2, 0)
    E2 = E2 ** 2
    E2 = sum(sum(sum(E2)))/3

    E3 = sum(e3[2][i]).detach().numpy().transpose(1, 2, 0)
    E3 = E3 ** 2
    E3 = sum(sum(sum(E3)))/3

    error_layer_3_pred2.append((E1 + E2 + E3)/(length * 3))

for i in frames:
    E1 = sum(e1[3][i]).detach().numpy().transpose(1, 2, 0)
    E1 = E1 ** 2
    E1 = sum(sum(sum(E1)))/3

    E2 = sum(e2[3][i]).detach().numpy().transpose(1, 2, 0)
    E2 = E2 ** 2
    E2 = sum(sum(sum(E2)))/3

    E3 = sum(e3[3][i]).detach().numpy().transpose(1, 2, 0)
    E3 = E3 ** 2
    E3 = sum(sum(sum(E3)))/3

    error_layer_4_pred2.append((E1 + E2 + E3)/(3 * length))


plt.figure(1)
plt.plot(frames, error_layer_1_pred1, label='prednet 1')
plt.plot(frames, error_layer_1_pred2, label='prednet 2')
plt.legend()
plt.xlabel('frame number')
plt.ylabel('MSE')
plt.title('MSE vs Frame Curve for Layers 1')
plt.savefig('part1_4_layer1.png')

plt.figure(2)
plt.plot(frames, error_layer_2_pred1, label='prednet 1')
plt.plot(frames, error_layer_2_pred2, label='prednet 2')
plt.legend()
plt.xlabel('frame number')
plt.ylabel('MSE')
plt.title('MSE vs Frame Curve for Layers 2')
plt.savefig('part1_4_layer2.png')

plt.figure(3)
plt.plot(frames, error_layer_3_pred1, label='prednet 1')
plt.plot(frames, error_layer_3_pred2, label='prednet 2')
plt.legend()
plt.xlabel('frame number')
plt.ylabel('MSE')
plt.title('MSE vs Frame Curve for Layers 3')
plt.savefig('part1_4_layer3.png')

plt.figure(4)
plt.plot(frames, error_layer_4_pred1, label='prednet 1')
plt.plot(frames, error_layer_4_pred2, label='prednet 2')
plt.legend()
plt.xlabel('frame number')
plt.ylabel('MSE')
plt.title('MSE vs Frame Curve for Layers 4')
plt.savefig('part1_4_layer4.png')
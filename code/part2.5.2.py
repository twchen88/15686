import numpy as np
import torch
import matplotlib.pyplot as plt

from skimage.io import imsave
from sklearn.decomposition import PCA
from prednet_2_5 import PredNet
from helper_functions import *
import torchvision

A_channels = (3, 48, 96, 192)
R_channels = (3, 48, 96, 192)

model = PredNet(R_channels, A_channels, output_mode='R')
weights = torch.load('models/prednet_1.pt', map_location='cpu')
model.load_state_dict(weights)

movie1 = load_gratings(1)
movie2 = load_gratings(2)
movie3 = load_gratings(3)
movie4 = load_gratings(4)

movies = torch.cat([movie1, movie2, movie3, movie4], dim=0)

with torch.no_grad():
    neurons = model(movies)
print(neurons[0][0].shape) # 4, 20, 4, 3, 128, 128
print(neurons[1][0].shape) # 48, 64, 64
print(neurons[2][0].shape) # 96, 32, 32
print(neurons[3][0].shape) # 192, 16, 16

for layer in range(4):
    size = (neurons[layer][0].shape)[2]
    data = (neurons[layer][0])[:, :, size//2, size//2]
    for i in range(1, len(neurons[0])):
        data = torch.cat([data, (neurons[layer][i])[:, :, size//2, size//2]], dim = 0)
    print(data.shape)
    pca = PCA(n_components=2)
    pca.fit(data)
    transformed = pca.transform(data)

    print(f'Explained variance of first two PCs: {pca.explained_variance_ratio_}')

    plt.figure()
    plt.plot(transformed[0:80:4, 0], transformed[0:80:4, 1], color='black',
        label='first grating')
    plt.plot(transformed[1:80:4, 0], transformed[1:80:4, 1], color='red',
            label='second grating')
    plt.plot(transformed[2:80:4, 0], transformed[2:80:4, 1], color='green',
            label='third grating')
    plt.plot(transformed[3:80:4, 0], transformed[3:80:4, 1], color='blue',
            label='fourth grating')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title(f'R Neuron Layer {layer + 1} PCA Plot')
    plt.legend() # to show the labels
    plt.savefig(f'part2_5_R_layer{layer + 1}.png')
    # plt.show()


### H neurons ###

model.output_mode = 'H'

with torch.no_grad():
    neurons = model(movies)
print(neurons[0][0].shape) # 4, 20, 4, 3, 128, 128
print(neurons[1][0].shape) # 48, 64, 64
print(neurons[2][0].shape) # 96, 32, 32
print(neurons[3][0].shape) # 192, 16, 16

for layer in range(4):
    size = (neurons[layer][0].shape)[2]
    data = (neurons[layer][0])[:, :, size//2, size//2]
    for i in range(1, len(neurons[0])):
        data = torch.cat([data, (neurons[layer][i])[:, :, size//2, size//2]], dim = 0)
    print(data.shape)
    pca = PCA(n_components=2)
    pca.fit(data)
    transformed = pca.transform(data)

    print(f'Explained variance of first two PCs: {pca.explained_variance_ratio_}')

    plt.figure()
    plt.plot(transformed[0:80:4, 0], transformed[0:80:4, 1], color='black',
        label='first grating')
    plt.plot(transformed[1:80:4, 0], transformed[1:80:4, 1], color='red',
            label='second grating')
    plt.plot(transformed[2:80:4, 0], transformed[2:80:4, 1], color='green',
            label='third grating')
    plt.plot(transformed[3:80:4, 0], transformed[3:80:4, 1], color='blue',
            label='fourth grating')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title(f'H neuron Layer {layer + 1} PCA Plot')
    plt.legend() # to show the labels
    plt.savefig(f'part2_5_H_layer{layer + 1}.png')
    # plt.show()
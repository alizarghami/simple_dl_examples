#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 20:07:46 2020

@author: ali
"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.datasets import MNIST

import numpy as np
import matplotlib.pyplot as plt


train_data = MNIST('./data', train=True, transform=transforms.ToTensor(), download=False)
test_data = MNIST('./data', train=False, transform=transforms.ToTensor(), download=False)

train_loader = DataLoader(train_data, batch_size=20, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)


class MyAutoEncoder(nn.Module):
    def __init__(self):
        super(MyAutoEncoder, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        
        self.pool = nn.MaxPool2d(2,2)
        self.up = nn.Upsample(scale_factor=2)
        
        self.conv3 = nn.Conv2d(20, 10, 5, padding=4)
        self.conv4 = nn.Conv2d(10, 1, 5, padding=4)
        
    def forward(self, x):
        def encode(x):
            x = f.relu(self.conv1(x))
            x = self.pool(x)
            x = f.relu(self.conv2(x))
            x = self.pool(x)
            return x
        
        def decode(x):
            x = self.up(x)
            x = f.relu(self.conv3(x))
            x = self.up(x)
            x = f.relu(self.conv4(x))
            return x
        
        x = encode(x)
        x = decode(x)
        return x
        
model = MyAutoEncoder()

# Test the model on a data before training
sample, label = next(iter(test_loader))
image = np.squeeze(sample.numpy())

output = model.forward(sample)
res = np.squeeze(output.data.numpy())

fig1 = plt.figure('Before training')
ax1 = fig1.add_subplot(1,2,1)
ax1.imshow(image)
ax2 = fig1.add_subplot(1,2,2)
ax2.imshow(res)


# Training the network
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epoch = 20
running_loss = 0
loss_over_time =[]
for epoch in range(n_epoch):
    for batch_i, (X_train, y_train) in enumerate(train_loader):
        optimizer.zero_grad()
        
        output = model.forward(X_train)
        
        loss = loss_function(output, X_train)
        loss.backward()

        optimizer.step()
        
        running_loss += loss.item()
        if batch_i % 1000 == 999:
            avg_loss = running_loss/1000
            loss_over_time.append(avg_loss)
            # record and print the avg loss over the 1000 batches
            print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, avg_loss))
            running_loss = 0.0


# Test the model on a data after training
output = model.forward(sample)
res = np.squeeze(output.data.numpy())


fig2 = plt.figure('Loss over time')
plt.plot(loss_over_time)
plt.xlabel('1000\'s of batches')
plt.ylabel('loss')
plt.ylim(0, 0.05) # consistent scale
plt.show()


fig3 = plt.figure('After training')
ax1 = fig3.add_subplot(1,2,1)
ax1.imshow(image)
ax2 = fig3.add_subplot(1,2,2)
ax2.imshow(res)


# Saving the model
version = 2
model_dir = 'saved_models/'
model_name = 'mnist_conv_autoencoder_v{}.pt'.format(version)

# after training, save your model parameters in the dir 'saved_models'
# when you're ready, un-comment the line below
torch.save(model.state_dict(), model_dir+model_name)
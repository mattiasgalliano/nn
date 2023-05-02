"""
The canonical example of a function that can't be
learned with a simple linear model is XOR
"""
import numpy as np
import pandas as pd

from nn.train import train
from nn.nn import NeuralNet
from nn.layers import Linear, Tanh, Sigmoid, Relu

fname = 'fashion.csv'

labels, ims = [], []

df = pd.read_csv(fname, header=None)

labels = df.iloc[:, 0]
ims = df.iloc[:, 1:]

labels, ims = np.array(labels), np.array(ims)
labels = np.expand_dims(labels, axis=1)


inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

net = NeuralNet([
    Linear(input_size=784, output_size=256),
    Tanh(),
    Linear(input_size=256, output_size=1)
])

train(net, ims, labels)

for x, y in zip(ims, labels):
    predicted = net.forward(x)
    print(x, predicted, y)

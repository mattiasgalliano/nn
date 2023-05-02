"""
Fashion MNIST example
"""
import numpy as np
import pandas as pd

from nn.train import train
from nn.nn import NeuralNet
from nn.layers import Linear, Tanh, Relu, Sigmoid


fname = 'fashion.csv'

df = pd.read_csv(fname, header=None)

labels = df.iloc[:, 0]
ims = df.iloc[:, 1:]


labels, ims = np.array(labels), np.array(ims)
labels = np.expand_dims(labels, axis=1)

train_ims, test_ims = ims, ims
train_labels, test_labels = labels, labels


net = NeuralNet([
    Linear(input_size=784, output_size=256),
    Sigmoid(),
    Linear(input_size=256, output_size=1)
])

train(net, ims, labels)

for x, y in zip(ims, labels):
    predicted = net.forward(x)
    print(predicted, y)


from nn.tensor import Tensor
from nn.nn import NeuralNet
from nn.loss import Loss, MSE
from nn.optim import Optimizer, SGD
from nn.data import DataIterator, BatchIterator

def train(net: NeuralNet,
        inputs: Tensor,
        targets: Tensor,
        num_epochs: int = 1000,
        iterator: DataIterator = BatchIterator(),
        loss: Loss = MSE(),
        optimizer: Optimizer = SGD(lr = 1e-5)) -> None:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            predicted = net.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        print(epoch, epoch_loss)

"""
A loss function measures how good our predictions are,
we can use this to adjust the parameters of our network
"""
import numpy as np

from nn.tensor import Tensor

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):
    """
    MSE is mean squared error, although we're
    just going to do total squared error
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)

class BCE(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        # print(predicted, actual)
        # print(np.log(predicted) * -actual)
        # print(((1 - actual)*(np.log(1 - predicted))))
        # print(-actual * np.log(predicted) - ((1 - actual)*(np.log(1 - predicted))))
        return np.sum(-actual * np.log(predicted) - ((1 - actual)*(np.log(1 - predicted))))
    
    def grad(self, predicted: Tensor, actual: Tensor) -> float:
        return (-(actual / predicted) + ((1 - actual) / (1 - predicted)))

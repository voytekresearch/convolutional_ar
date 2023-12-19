"""Tests for neural network."""
import numpy as np
import torch
from torch import nn
from convolutional_ar.reshape import ar_reshape
from convolutional_ar.nn import AR, train_ar

def test_ar_forward():
    x_ar, _ = ar_reshape(np.random.rand(100, 100), 10)
    model = AR(order=10)
    output = model.forward(x_ar)
    assert output.shape[0] == len(x_ar)  # Ensure the output shape is correct

def test_train_ar():
    x_ar, y_ar = ar_reshape(np.random.rand(100, 100), 10)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD
    n_epochs = 10
    batch_size = 5
    output = train_ar(x_ar, y_ar, loss_fn, optimizer, lr=0.01, n_epochs=n_epochs, batch_size=batch_size)
    assert output.shape == (n_epochs,)  # Ensure the output shape is correct

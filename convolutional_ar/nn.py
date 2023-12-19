"""Autoregressive neutral network."""

from typing import Callable, Optional
import numpy as np
import torch
from torch import nn


class AR(nn.Module):
    """Neutral network for solving AR."""

    def __init__(self, order: int):
        super().__init__()
        # Initalize at small random values close to zero
        self.w = nn.Parameter(torch.zeros(order) + torch.rand(order) / 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w


def train_ar(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    loss_fn: Callable,
    optimizer: Callable,
    lr: float,
    n_epochs: int,
    batch_size: int,
    save_dir: Optional[str] = None,
    save_index: Optional[int] = None,
    progress: Optional[Callable] = None,
) -> np.ndarray:
    """Train the AR model.

    Parameters
    ----------
    X_train : 1d array
        Past values along the circle.
    y_train : 1d array
        Future values along the circle.
    loss_fn : function
        Evaluates loss e.g. MSE, L1, Huber, etc.
    optimzier : torch.optim.Optimizer
        Imported from torch.optim.* and unintialize,
        e.g. torch.optim.Adam or torch.optim.SGD.
    lr : float
        Learning rate.
    n_epochs : int
        Number of times to pass through dataset.
    batch_size : int
        Number of observations per gradient step.
        Batches are randomly shuffled.
    save_dir : str, optional, default: None
        Path to save model.
    save_index : int, optional, default: None
        Used in the output file name.
    progress : Callable, optional, default: None
        Wraps epoch iterable and prints iteration,
        e.g. tqdm.tqdm or tqdm.notebook.tqm

    Returns
    -------
    ar_coefs : 1d array
        Autoregressive coefficients.
    """
    model = AR(len(X_train[0]))
    opt = optimizer(model.parameters(), lr=lr)

    e_iter = range(n_epochs)
    if progress:
        e_iter = progress(e_iter, total=n_epochs)

    for _ in e_iter:
        permutation = torch.randperm(len(X_train))

        for i_batch in range(0, len(X_train), batch_size):
            indices = permutation[i_batch : i_batch + batch_size]
            X_batch, y_batch = X_train[indices], y_train[indices]

            out = model(X_batch)
            loss = loss_fn(out, y_batch)
            loss.backward()

            opt.step()
            opt.zero_grad()

    # Report final loss
    if progress is not None:
        y_pred_train = model(X_train)
        loss_train = float(loss_fn(y_pred_train, y_train))
    if progress is not None and save_index is not None:
        print(f"{save_index} final loss: {loss_train}")
    elif progress is not None:
        print(f"final loss: {loss_train}")

    # Save model
    if save_dir is not None and save_index is not None:
        torch.save(model, f"{save_dir}/model_{str(save_index).zfill(4)}.pkl")
    elif save_dir is not None:
        torch.save(model, f"{save_dir}/model.pkl")

    # Return ar coefficients
    ar_coefs = model.w.detach().numpy()

    return ar_coefs

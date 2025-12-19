"""Convolutional Autoregressive Model."""

import sys
from typing import Callable, Optional

import numpy as np

import torch
from torch import nn


class ConvAR:
    """Convolutional Autoregressive Model.

    Attributes
    ----------
    weight_matrix_ : 2d or 3d tensor
        Learned weights in matrix form (e.g. with duplicates).
    weight_vector_ : 2d tensor
        Learned weights as the unique of the weight matrix, sorted by distance.
    """

    def __init__(
        self,
        radius: int,
        ndim: Optional[int] = 2,
        loss_fn: Optional[Callable] = None,
        loss_thresh: Optional[float] = None,
        optim: Optional[type] = None,
        lr: Optional[float] = 1e-3,
        lr_scheduler: Optional[type] = None,
        n_epochs: Optional[int] = 1000,
        device=None,
        verbose: Optional[int] = 10,
        init_weight_matrix: Optional[torch.tensor]=None
    ):
        """Initialize.

        Parameters
        ----------
        radius : int
            Size of window from the center pixel, excluding the center pixel.
            Windows length is (2 * radius) + 1
        ndim : int, optional, default: 2
            Dimensionality of input. 2 for images. 3 for volumetric.
        loss_fn : func, optional, default: None
            Loss function, e.g. torch.nn.MSELoss().
        loss_thresh : float, optional, default: None
            Loss threshold to exit optimization loop.
        optim : type : optional, default: None
            Unitialized optimizer, e.g. torch.optim.Adam.
        lr : float, optional, default: 1e-3
            Learning rate.
        lr_scheduler : type : optional, default: None
            Function, e.g. torch.optim.lr_scheduler.LinearLR, that accepts an
            optimizer as input. Additional arguments to the scheduler should be
            set using a partial or lambda function.
        device : str, optional, default: None
            None for cpu. "cuda" or "mps" for gpu.
        n_epochs : int, optional, default: 1000
            Number of epochs.
        verbose : int, optional, default: 10
            Number of epochs between loss reports.
        init_weight_matrix : torch.tensor, optional, default: None
            Initalize weight matrix. Useful for resuming optimization.
        """
        # Window
        self.radius = radius
        self.ndim = ndim
        self.window_size = int(2 * radius + 1)
        self.ctr = (self.window_size - 1) // 2

        # Optimization
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.n_epochs = n_epochs

        self.loss_fn = nn.MSELoss() if loss_fn is None else loss_fn
        self.loss_thresh = -torch.inf if loss_thresh is None else loss_thresh
        self.optim = torch.optim.Adam if optim is None else optim

        self.device = device
        self.verbose = verbose

        # Results
        self.model = None
        if init_weight_matrix is not None:
            self._init_weight_matrix = init_weight_matrix.clone()
        else:
            self._init_weight_matrix = None
        self.weight_matrix_ = None
        self.weight_vector_ = None

    def fit(
        self,
        X: torch.tensor,
        progress: Optional[Callable] = None,
    ):
        """Fit image(s).

        Parameters
        ----------
        X : 3d or 4d tensor
            Greyscale image data. Should have shape of either:
            (n_observations, voxels_i, voxels_j, voxels_k) or (voxels_i, voxels_j, voxels_k)
        progress : func
            Wraps iterations over X, typically a lambda with tqdm.tqdm or tqdm.notebook.tqdm.
        """

        if (self.ndim == 3 and X.ndim == 3) or (self.ndim == 2 and X.ndim == 2):
            X = X.reshape(1, *X.shape)

        self.weight_matrix_ = torch.zeros((len(X), *[self.window_size]*self.ndim))
        self.weight_vector_ = torch.zeros((len(X), self.window_size))

        if progress is None:
            iterable = range(len(X))
        else:
            iterable = progress(range(len(X)))

        if self.device is not None and self._init_weight_matrix is not None:
            self._init_weight_matrix = self._init_weight_matrix.to(self.device)

        if self.model is None:
            self.model = ConvARBase(self.radius, ndim=self.ndim, weight_matrix=self._init_weight_matrix)
        else:
            self.model.reset_weights()

        if self.device is not None:
            X = X.to(self.device)
            self.model = self.model.to(self.device)
            self.model.weight_scale = self.model.weight_scale.to(self.device)
            self.model.weight_matrix = self.model.weight_matrix.to(self.device)

        radius_slice = [slice(self.radius, -self.radius)] * self.ndim

        for i_x in iterable:

            # Target values (center of windows)
            y_ctr = X[i_x, *radius_slice].reshape(-1, 1)

            # Reset weights
            if i_x != 0:
                self.model.reset_weights()

            # Initialize optimizer
            optim = self.optim(self.model.parameters(), lr=self.lr)
            if self.lr_scheduler is not None:
                scheduler = self.lr_scheduler(optim)

            # Descent
            for i_epoch in range(self.n_epochs):

                y_pred = self.model(X[i_x])

                loss = self.loss_fn(y_pred, y_ctr)
                loss.backward()

                optim.step()
                optim.zero_grad()

                if self.lr_scheduler is not None:
                    scheduler.step()

                # Reporting
                if self.verbose is None:
                    continue

                if self.verbose is not None and (
                    i_epoch % self.verbose == 0 or i_epoch == self.n_epochs - 1
                ):
                    # Print progress
                    sys.stdout.write(
                        f"\rmodel {i_x}, epoch {i_epoch}, loss {float(loss.detach())}"
                    )
                    sys.stdout.flush()

                if float(loss.detach()) < self.loss_thresh:
                    # Exit loop when loss is below threshold after n iterations
                    if self.verbose is not None:
                        sys.stdout.write(
                            f"\rmodel {i_x}, epoch {i_epoch}, loss {float(loss.detach())}"
                        )
                        print("")
                    break

                if self.verbose and i_epoch == self.n_epochs - 1:
                    print("")

            # Get weights out of model
            self.weight_matrix_[i_x] = (
                self.model.weight_matrix.detach().clone() * self.model.weight_scale
            )

        # Get weight vectors from weight matrix using masks
        self.weight_vector_ = torch.zeros((len(X), self.model.n_unique))
        for i_x in range(len(X)):
            for i_m in range(self.model.n_unique):
                self.weight_vector_[i_x][i_m] = self.weight_matrix_[i_x][
                    self.model.masks[i_m]
                ][0]

        self.y_pred = y_pred


class ConvARBase(nn.Module):
    """Torch model."""

    def __init__(
        self,
        radius: int,
        ndim : Optional[int] = 2,
        weight_vector: Optional[torch.tensor] = None,
        weight_matrix: Optional[torch.tensor] = None,
    ):
        """
        Parameters
        ----------
        radius : int
            Size of window from the center pixel, excluding the center pixel.
            Windows length is (2 * radius) + 1
        ndim : int, optional, default: 2
            Dimensionality of input. 2 for images. 3 for volumetric.
        weight_vector : 1d or 2d tensor
            Inital weights. 1d if input is 2d. 2d if input is in 3d.
            Theses will be mapped into weight_matrix, e.g. for a 3x3 window:
                weight_vector = torch.tensor([w0, w1])
                weight_matix = torch.tensor([
                    [w1, w0, w1],
                    [w0,  0, w0],
                    [w1, w0, w1]
                ])
        weight_matrix : 1d or 2d tensor
            Inital weights. Theses will be mapped into weight_vector.
            This matrix should be symmetric with zero in the middle.
                weight_matix = torch.tensor([
                    [w1, w0, w1],
                    [w0,  0, w0],
                    [w1, w0, w1]
                ])
        """
        super().__init__()

        # Set forward func
        self.ndim = ndim

        if self.ndim == 2:
            self.conv = torch.conv2d
        else:
            self.conv = torch.conv3d

        # Compute distances from center
        self.radius = radius
        self.window_size = int(2 * radius + 1)

        shape = tuple(self.window_size for _ in range(ndim))
        center = torch.tensor(shape) // 2
        indices = torch.meshgrid([torch.arange(size) for size in shape], indexing="ij")
        indices = torch.stack(indices, dim=-1).float()
        self.distances = torch.linalg.norm(indices - center, dim=-1)

        # Sort distances from center
        self.unique_distances = torch.unique(self.distances)[1:]
        self.n_unique = len(self.unique_distances)

        # Counts (number of pixels) per distance
        self.counts = torch.zeros(len(self.unique_distances))

        for id, d in enumerate(self.unique_distances):
            self.counts[id] = (self.distances == d).sum()

        # Masks that map to distance group
        #   (e.g. all pixels with distance from center == i)
        self.masks = torch.zeros(
            (len(self.unique_distances), *shape),
            dtype=bool
        )
        for i, d in enumerate(self.unique_distances):
            mask = self.distances == d
            self.masks[i] = mask

        # Inital weights
        if weight_vector is not None and weight_matrix is not None:
            # Vector and matrix passed
            raise ValueError("Pass eighter weight_vector or weight_matrix, not both.")
        elif weight_vector is None and weight_matrix is None:
            # Neither passed, randomize weights
            self.weight_vector = torch.rand(self.n_unique)
            self.weight_matrix =  vector_to_matrix(self.weight_vector, self.masks)
        elif weight_vector is not None:
            # Vector passed
            self.weight_vector = weight_vector.clone()
            self.weight_matrix =  vector_to_matrix(self.weight_vector.clone(), self.masks)
        elif weight_matrix is not None:
            # Matrix pass
            self.weight_matrix = weight_matrix.clone()
            self.weight_vector = matrix_to_vector(self.weight_matrix.clone(), self.masks, self.n_unique)

        # Matrix is differentiable
        self.weight_matrix = nn.Parameter(self.weight_matrix)

        # Ensure gradients are fixed at equal distances from center
        self.weight_matrix.register_hook(
           lambda grad: self.equalize_grad(grad, self.masks)
        )

        # Store inital state
        self._weight_vector_orig = self.weight_vector.clone()
        self._weight_matrix_orig = self.weight_matrix.clone()

        # Scale weights based on how many pixels map to distance i
        self.weight_scale = torch.zeros(shape)
        for i in range(self.n_unique):
            # Scale with 1 / number of points at same distance from center of kernel
            self.weight_scale[self.masks[i]] = 1 / self.counts[i]


    def reset_weights(self):
        """Reset model weights.

        Notes
        -----
        Useful for avoiding the computation in init when applied to multiple tensors.
        """
        self.weight_vector = self._weight_vector_orig.clone()
        self.weight_matrix.data = self._weight_matrix_orig.clone()

    @staticmethod
    def equalize_grad(grad: torch.tensor, masks: list[torch.tensor]):
        """Hook to equalize gradient.

        Parameters
        ----------
        grad : 2d tensor
            Gradient from autograd.
        masks : list of 2d tensor
            Uses to index weights that should be equal.

        Notes
        -----
        Given the following weight matrix:

        weight_matrix = torch.tensor([
                    [w1, w0, w1],
                    [w0,  0, w0],
                    [w1, w0, w1]
        ])

        This hook forces the gradients for w1 and w0 to be the same by taking the mean.
        """
        for m in masks:
            grad[m] = grad[m].mean()
        return grad

    def forward(self, X: torch.tensor):
        """Define forward pass.

        Parameters
        ----------
        X : torch.Tensor
            Image.
        """
        return self.conv(
            X.view(1, 1, *X.shape),
            self.weight_matrix.view(1, 1, *self.weight_matrix.shape) * \
                self.weight_scale.view(1, 1, *self.weight_scale.shape)
        )[0, 0].view(-1, 1)


def vector_to_matrix(weight_vector, masks):
    """Convert weights from vector to matrix."""
    weight_matrix = torch.zeros(*masks[0].shape)

    for i_m in range(len(weight_vector)):
        weight_matrix[masks[i_m]] = weight_vector[i_m]

    return weight_matrix


def matrix_to_vector(weight_matrix, masks, n_unique):
    """Convert weights frmo matrix to vector."""
    weight_vector = torch.zeros(n_unique)

    for i_m in range(n_unique):
        weight_vector[i_m] = weight_matrix[
            masks[i_m]
        ][0]

    return weight_vector
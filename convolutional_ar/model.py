"""Convolutional Autoregressive Model."""

import sys
from typing import Callable, Optional
import torch


class ConvolutionalAR:
    """Convolutional Autoregressive Model.

    Attributes
    ----------
    weight_matrix_ : 2d tensor
        Learned weights in matrix form (e.g. with duplicates).
    weight_vector_ : 1d tensor
        Learned weights as the unique of the weight matrix, sorted by distance.
    """

    def __init__(
        self,
        window_size: int,
        loss_fn: Optional[Callable] = None,
        loss_thresh: Optional[float] = None,
        optim: Optional[type] = None,
        lr: Optional[float] = 1e-3,
        n_epochs: Optional[int] = 1000,
        adaptive_weights: Optional[bool] = False,
        verbose: Optional[int] = 10,
    ):
        """Initialize.

        Parameters
        ----------
        window_size : int
            Size of window length that is to be convolved. Must be odd so that
            there is a single center pixel.
        loss_fn : func, optional, default: None
        """
        # Window
        self.window_size = window_size
        if window_size % 2 != 1:
            raise ValueError("window_size should be odd.")
        self.ctr = (self.window_size - 1) // 2

        # Optimization
        self.lr = lr
        self.adaptive_weights = adaptive_weights
        self._last_label = None
        self.n_epochs = n_epochs

        self.loss_fn = torch.nn.MSELoss() if loss_fn is None else loss_fn
        self.loss_thresh = -torch.inf if loss_thresh is None else loss_thresh
        self.optim = torch.optim.Adam if optim is None else optim

        self.verbose = verbose

        # Results
        self.weight_matrix_ = None
        self.weight_vector_ = None

    def fit(
        self,
        X: torch.tensor,
        y: Optional[torch.tensor] = None,
        progress: Optional[Callable] = None,
    ):
        """Fit image(s).

        Parameters
        ----------
        X : 2d or 3d tensor
            Greyscale image data. Should have shape of either:
            (n_observations, pixel_rows, pixel_columns) or (pixel_rows, pixel_columns)
        y : 1d tensor
            Only needed if adapative_weights is True and X is 3d. These are binary labels.
            Used to initialize weights within groups adaptively for faster convergence.
        progress : func
            Wraps iterations over X, typically with tqdm.tqdm or tqdm.notebook.tqdm.
        """

        if X.ndim == 2:
            X = X.reshape(1, *X.shape)

        self.weight_matrix_ = torch.zeros((len(X), self.window_size, self.window_size))
        self.weight_vector_ = torch.zeros((len(X), self.window_size))

        if progress is None:
            iterable = range(len(X))
        else:
            iterable = progress(range(len(X)))

        self.model = SolveConvolutionalAR(self.window_size)

        for i_x in iterable:
            # Todo: multiprocessing for this loop

            # Sliding window view
            x_windowed = (
                X[i_x].unfold(0, self.window_size, 1).unfold(1, self.window_size, 1)
            )
            x_windowed = x_windowed.reshape(-1, self.window_size, self.window_size)

            # Target values (center of windows)
            y_ctr = x_windowed[:, self.ctr, self.ctr].reshape(-1, 1)

            # Reset weights
            if (
                i_x != 0
                and not self.adaptive_weights
                and (last_label is None or last_label != y[i_x])
            ):
                self.model.reset_weights()

            # Initialize optimizer
            optim = self.optim(self.model.parameters(), lr=self.lr)

            # Descent
            for i_epoch in range(self.n_epochs):
                y_pred = self.model(x_windowed)

                loss = self.loss_fn(y_pred, y_ctr)
                loss.backward()

                optim.step()
                optim.zero_grad()

                # Reporting
                if self.verbose is not None and (
                    i_epoch % self.verbose == 0 or i_epoch == self.n_epochs - 1
                ):
                    # Print progress
                    sys.stdout.write(
                        f"\rmodel {i_x}, epoch {i_epoch}, loss {float(loss)}"
                    )
                    sys.stdout.flush()

                if float(loss) < self.loss_thresh:
                    # Exit loop when loss is below threshold
                    if self.verbose is not None:
                        sys.stdout.write(
                            f"\rmodel {i_x}, epoch {i_epoch}, loss {float(loss)}"
                        )
                        print("")
                    break

                if self.verbose and i_epoch == self.n_epochs - 1:
                    print("")

            # Get weights out of model
            self.weight_matrix_[i_x] = (
                self.model.weight_matrix.detach().clone() * self.model.weight_scale
            )

            last_label = y[i_x] if y is not None else None

        # Get weight vectors from weight matrix using masks
        self.weight_vector_ = torch.zeros((len(X), self.model.n_unique))
        for i_x in range(len(X)):
            for i_m in range(len(self.model.masks)):
                self.weight_vector_[i_x] = self.weight_matrix_[i_x][
                    self.model.masks[i_m]
                ][0]


class SolveConvolutionalAR(torch.nn.Module):
    """Torch model."""

    def __init__(
        self,
        window_size: int,
        weight_vector: Optional[torch.tensor] = None,
        weight_matrix: Optional[torch.tensor] = None,
    ):
        """
        Parameters
        ----------
        window_size : int
            Size of window length. Total elements in window is: window_size**2.
        weight_vector : 1d tensor
            Inital weights. Theses will be mapped into weight_matrix, e.g. for a 3x3 window:
                weight_vector = torch.tensor([w0, w])
                weight_matix = torch.tensor([
                    [w1, w0, w1],
                    [w0,  0, w0],
                    [w1, w0, w1]
                ])
        weight_matrix : 2d tensor
            Inital weights. Theses will be mapped into weight_vector.
            This matrix should be symmetric with zero in the middle.
                weight_matix = torch.tensor([
                    [w1, w0, w1],
                    [w0,  0, w0],
                    [w1, w0, w1]
                ])
        """
        super().__init__()

        # Compute distances from center
        if window_size % 2 != 1:
            raise ValueError("window_size should be odd.")

        self.window_size = window_size

        # A faster algorithm for this exists
        self.distances = torch.hypot(
            *torch.meshgrid(
                torch.arange(self.window_size) - ((self.window_size - 1) / 2),
                torch.arange(self.window_size) - ((self.window_size - 1) / 2),
            )
        )

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
            (len(self.unique_distances), window_size, window_size), dtype=bool
        )
        for i, d in enumerate(self.unique_distances):
            mask = self.distances == d
            self.masks[i] = mask

        # Inital weights
        if weight_vector == None:
            self._random_weights = True
            self.weight_vector = torch.exp(-self.unique_distances.clone() / 0.5) * 5
        else:
            self.weight_vector = weight_vector

        self._weight_vector_orig = self.weight_vector.clone()

        # Scale weights based on how many pixels map to distance i
        self.weight_scale = torch.zeros((window_size, window_size))
        for i in range(self.n_unique):
            # Scale with 1 / number of points at same distance from center of kernel
            self.weight_scale[self.masks[i]] = 1 / self.counts[i]

        # Create a matrix of weights
        if weight_matrix is not None:
            self.weight_matrix = weight_matrix
        else:
            self.weight_matrix = torch.zeros((self.window_size, self.window_size))
            for i in range(self.n_unique):
                # Matrix of weights (e.g. w mapped to a 2d gaussian kernel)
                self.weight_matrix[self.masks[i]] = self.weight_vector[i]

        # Make differentiable
        self.weight_matrix = torch.nn.Parameter(self.weight_matrix)

        # Ensure gradients are fixed at equal distances from center
        self.weight_matrix.register_hook(
            lambda grad: self.equalize_grad(grad, self.masks)
        )

        self._weight_matrix_orig = self.weight_matrix.clone()

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

        weight_matix = torch.tensor([
                    [w1, w0, w1],
                    [w0,  0, w0],
                    [w1, w0, w1]
        ])

        This hook forces the gradients for w1 and w0 to be the same by taking the mean.
        """
        for m in masks:
            grad[m] = grad[m].mean()
        return grad

    def forward(self, x: torch.tensor):
        """Define forward pass.

        Parameters
        ----------
        x : 3d torch.tensor
            Windowed image with shape [n_windows, window_size, window_size].
        """
        return x.reshape(len(x), -1) @ (
            self.weight_matrix.view(-1, 1) * self.weight_scale.view(-1, 1)
        )

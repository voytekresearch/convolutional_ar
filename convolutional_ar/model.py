"""Convolutional Autoregressive Model."""

import sys
from typing import Callable, Optional
import torch
import numpy as np

class ConvolutionalAR:
    """Convolutional Autoregressive Model.

    Attributes
    ----------
    weight_matrix_ : 3d tensor
        Learned weights in matrix form (e.g. with duplicates).
    weight_vector_ : 2d tensor
        Learned weights as the unique of the weight matrix, sorted by distance.
    """

    def __init__(
        self,
        radius: int,
        loss_fn: Optional[Callable] = None,
        loss_thresh: Optional[float] = None,
        optim: Optional[type] = None,
        lr: Optional[float] = 1e-3,
        n_epochs: Optional[int] = 1000,
        adaptive_weights: Optional[int] = None,
        verbose: Optional[int] = 10,
        init_weight_matrix: Optional[torch.tensor]=None
    ):
        """Initialize.

        Parameters
        ----------
        radius : int
            Size of window from the center pixel.
        loss_fn : func, optional, default: None
            Loss function, e.g. torch.nn.MSELoss().
        loss_thresh : float, optional, default: None
            Loss threshold to exit optimization loop.
        optim : type : optional, default: None
            Unitialized optimizer, e.g. torch.optim.Adam.
        lr : float, optional, default: 1e-3
            Learning rate.
        n_epochs : int, optional, default: 1000
            Number of epochs.
        adaptive_weights : int, optional, default: None
            Starts next iteration with weigts from the last if not None.
            The int specifices the number of additional iterations to
            continue afer loss_thresh has been met to prevent immediately
            exiting optimization.
        verbose : int, optional, default: 10
            Number of epochs between loss reports.
        init_weight_matrix : torch.tensor, optional, default: None
            Initalize weight matrix. Useful for resuming optimization.
        """
        # Window
        self.radius = radius
        self.window_size = int(2 * self.radius + 1)
        if self.window_size % 2 != 1:
            raise ValueError("window_size should be odd.")
        self.ctr = (self.window_size - 1) // 2

        # Optimization
        self.lr = lr
        self.adaptive_weights = adaptive_weights
        self.n_epochs = n_epochs

        self.loss_fn = torch.nn.MSELoss() if loss_fn is None else loss_fn
        self.loss_thresh = -torch.inf if loss_thresh is None else loss_thresh
        self.optim = torch.optim.Adam if optim is None else optim

        self.verbose = verbose

        # Results
        self._init_weight_matrix = init_weight_matrix
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

        self.model = SolveConvolutionalAR(self.window_size, weight_matrix=self._init_weight_matrix)

        last_label = None

        for i_x in iterable:

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
                and self.adaptive_weights is None
                or (last_label is None or last_label != y[i_x])
            ):
                self.model.reset_weights()

            # Initialize optimizer
            optim = self.optim(self.model.parameters(), lr=self.lr)

            # Number of times after the adaptive threshold has been met to continue
            n_after = 0

            # Descent
            for i_epoch in range(self.n_epochs):

                y_pred = self.model(x_windowed)

                # _y_pred = y_pred.detach().clone().numpy()
                # np.save(f'/Users/ryanhammonds/projects/convolutional_ar/predicted_images/img_{str(i_epoch).zfill(4)}.npz',
                #         _y_pred
                # )

                # wm = self.model.weight_matrix.detach().clone().numpy()
                # np.save(f'/Users/ryanhammonds/projects/convolutional_ar/predicted_images/w_{str(i_epoch).zfill(4)}.npz',
                #         wm
                # )

                loss = self.loss_fn(y_pred, y_ctr)
                loss.backward()

                optim.step()
                optim.zero_grad()

                # Reporting
                if self.verbose is None and self.adaptive_weights is None:
                    continue

                if self.verbose is not None and (
                    i_epoch % self.verbose == 0 or i_epoch == self.n_epochs - 1
                ):
                    # Print progress
                    sys.stdout.write(
                        f"\rmodel {i_x}, epoch {i_epoch}, loss {float(loss)}"
                    )
                    sys.stdout.flush()

                if float(loss) < self.loss_thresh:
                    n_after += 1

                if n_after == self.adaptive_weights or (self.adaptive_weights is None and n_after == 1):
                    # Exit loop when loss is below threshold after n iterations
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
            for i_m in range(self.model.n_unique):
                self.weight_vector_[i_x][i_m] = self.weight_matrix_[i_x][
                    self.model.masks[i_m]
                ][0]

        self.y_pred = y_pred

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
                indexing="ij"
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
            #self.weight_vector = torch.exp(-self.unique_distances.clone() / 0.5)
            self.weight_vector = torch.rand(self.n_unique) #/ 100
            #self.weight_vector = torch.exp(torch.randn(self.n_unique))
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

    def forward(self, x: torch.tensor):
        """Define forward pass.

        Parameters
        ----------
        x : 3d torch.tensor
            Windowed image with shape [n_windows, window_size, window_size].
        """
        return x.view(len(x), -1) @ (
            self.weight_matrix.view(-1, 1) * self.weight_scale.view(-1, 1)
        )


def vector_to_matrix(weight_vector, masks):

    weight_matrix = torch.zeros(*masks[0].shape)

    for i_m in range(len(weight_vector)):
        weight_matrix[masks[i_m]] = weight_vector[i_m]

    return weight_matrix
"""Solve the convolutional problem as a linear system."""

from typing import Optional
import torch

class ConvAR:
    """Convolutional AR, solved as linear system using least-squares.

    Attributes
    ----------
    weights : 3d tensor
        Convolutional kernel as (n_obs, n, d).
    """
    def __init__(
        self,
        radius: int,
        stride: Optional[int]=1,
        dilation: Optional[int]=1,
        equidistant=True
    ):
        """Initialize.

        Parameters
        ----------
        radius : int
            Size of window from the center pixel, excluding the center pixel.
            Windows length is (2 * radius) + 1
        stride : int, optional, default: 1
            Steps to move from one window to the next.
        dilation : int, optional, default: 1
            Expansion of window.
        equidistant : bool, optional, default: True
        """
        self.radius = radius
        self.window_size = int(2*radius) + 1
        self.stride = stride
        self.dilation = dilation
        self.equidistant = equidistant
        self.weights = None

        # A faster algorithm for this exists
        self.distances = torch.hypot(
            *torch.meshgrid(
                torch.arange(self.window_size) - ((self.window_size - 1) / 2),
                torch.arange(self.window_size) - ((self.window_size - 1) / 2),
                indexing="ij"
            )
        )

        if self.equidistant:

            # Sort distances from center
            self.unique_distances = torch.unique(self.distances)[1:]
            self.n_unique = len(self.unique_distances)

            # Counts (number of pixels) per distance
            self.counts = torch.zeros(len(self.unique_distances))

            for id, d in enumerate(self.unique_distances):
                self.counts[id] = (self.distances == d).sum()

            # Masks that map to distance group
            #   (e.g. all pixels with distance from center == i)
            self.mask = torch.zeros(
                (len(self.unique_distances), self.window_size, self.window_size),
                dtype=bool
            )
            for i, d in enumerate(self.unique_distances):
                self.mask[i] = self.distances == d
        else:
            self.mask = torch.ones((self.window_size, self.window_size), dtype=bool)
            self.mask[self.radius, self.radius] = False

    def fit(self, X):
        """Learn the AR coefficients.

        Parameters
        ----------
        X : 2d or 3d tensor
            Greyscale image data. Should have shape of either:
            (n_observations, pixel_rows, pixel_columns) or (pixel_rows, pixel_columns)
        """
        if X.ndim == 2:
            X = X.unsqueeze(0)

        for i in range(len(X)):

            X_windowed = extract_windows(X[i], self.window_size, self.stride, self.dilation)

            if not self.equidistant:
                self.weights = torch.zeros((len(X), int(self.window_size*self.window_size)-1))
                Xw = X_windowed[:, self.mask]
                yw = X_windowed[:, ~self.mask]
                w, _, _, _ = torch.linalg.lstsq(Xw, yw)
                self.weights[i] = w[:, 0]
            else:
                self.weights = torch.zeros((len(X), self.n_unique))
                Xw = torch.zeros((len(X_windowed), self.n_unique))
                yw = torch.zeros(len(X_windowed))
                for i_dist in range(self.n_unique):
                    Xw[:, i_dist] = X_windowed[:, self.mask[i_dist]].mean(dim=1)
                yw = X_windowed[:, self.radius, self.radius]
                w, _, _, _ = torch.linalg.lstsq(Xw, yw)
                self.weights[i] = w


def extract_windows(X: torch.Tensor, window_size: int, stride: Optional[int] = 1, dilation: Optional[int] = 1):
    """Extract sub-windows from X.

    Parameters
    ----------
    X : 2d torch.Tensor
        Input image.
    window_size : int
        Size of sub-window.
    stride : int
        Steps to move from one window to the next.
    dilation : int
        Expansion of window.

    Return
    ------
    windows : 3d torch.Tensor
        Sliding window of image, shape: (n_windows, window_size, window_size).
    """

    H, W = X.shape
    eff_size = (window_size - 1) * dilation + 1

    out_h = (H - eff_size) // stride + 1
    out_w = (W - eff_size) // stride + 1

    shape = (out_h, out_w, window_size, window_size)

    strides = (
        X.stride(0) * stride,
        X.stride(1) * stride,
        X.stride(0) * dilation,
        X.stride(1) * dilation,
    )

    windows = torch.as_strided(X, size=shape, stride=strides)

    return windows.reshape(-1, window_size, window_size)

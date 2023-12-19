"""Power spectral density."""

import numpy as np
from scipy.fft import fftfreq
from spectrum import arma2psd

def ar_to_psd(ar_coefs: np.ndarray, nfft: int) -> (np.ndarray, np.ndarray):
    """Convert ar coeffcients to PSD.

    Parameters
    ----------
    ar_coefs : 1d array
        Autoregressive coefficients.
    nfft : int
        Number of points to use in the FFT.

    Returns
    -------
    freqs : 1d array
        Frequencies.
    powers : 1d array
        Power spectral density.
    """

    fs = 100 # this should really be in units of distance

    powers = arma2psd(A=-ar_coefs[::-1], rho=1., T=fs, NFFT=nfft)
    freqs = fftfreq(nfft, 1/fs)
    powers = powers[:len(freqs)//2]
    freqs = freqs[:len(freqs)//2]

    freqs = freqs[1:]
    powers = powers[1:]

    return freqs, powers
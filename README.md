# Autoregressive Models for Texture Images


## Convolutional AR Model

Rotaion invariant simultaneous autoregressive and circlular autoregressive (AR)
models are described in Jianchang and Anil 1994. These models were extended
in to a convolutional like framework to make weights translation invariant.

![car](https://github.com/ryanhammonds/test/blob/6a6e89600cc655f4303ed25da8ae0bad0ebf527e/docs/car.png)

A set of AR weights (w) are learned for each circle such that wx = y, where x are
values of past pixels and the y is the current pixel along the circle. The same
weights are applied to every cicle in the image such that one set of
weights is learned. Here we used an order of 20, so 20 weights are learned per image.

Circles are used to ensure rotational invaraince (e.g. the same result regardless)
of rotations of the image. The convolution of the cicular kernel ensures that the
weights are translation invariant. Scaling invariance is still an open question
of how to solve. This could involve using various size radii for the circle per
image to account for scaling difference.

Since the circles are distinct, standard AR solutions (e.g. Burg's method) can not
be used since there is an assumption of continuity. The analogy in the time domain
is that Burg's method can not be used to learn a common set of weights for distinct
signals. Instead, a simple neural network is used to learn the AR weights
simulataneously.

## Power Spectral Density

Given AR coeffiecients, power spectral density (PSD) may be computed per image.
The advantage of this is that the PSD is not effected by rotations or
translations, like the 2d FFT is.

Spectral parameterization has not yet been used. If a classification model
performed poorly on log power, parameters extracted will not improve
performance. Since decent accuracy was found here, spatial scales (e.g. image
analog of timescales) may be extract as a secondary analysis, but performance is
expected to decrease.

Log PSD comparisons between classes:

![PSD](https://github.com/ryanhammonds/test/blob/a4fe60d982e12a3ddd9cd92a915d5a953526ae19/docs/psd.png)

## Datasets

Kylberg textures. Examples of each class:

![kylberg](https://github.com/ryanhammonds/test/blob/6a6e89600cc655f4303ed25da8ae0bad0ebf527e/docs/example_x.png)

These stochastic / semi-random patterns can be learned as autoregressive coefficients.

## Results

A SVM found 87% accuracy for and 80:20 train test split. This is promising since chance
is 0.035%, since there are 28 classes. State of the art convolution networks have used
this dataset and found ~97% accuracy.

Improved tuning and cross-validation is needed. To be updated.

### Citations

Mao, J., & Jain, A. K. (1992). Texture classification and segmentation using multiresolution simultaneous autoregressive models. Pattern recognition, 25(2), 173-188.

Kylberg, G. (2011). Kylberg texture dataset v. 1.0. Centre for Image Analysis, Swedish University of Agricultural Sciences and Uppsala University.

# Autoregressive Models for Texture Images

# Convolutional Autoregressive Model

<img src="https://raw.githubusercontent.com/ryanhammonds/explorations/master/docs/convolution.png" width="500" style="width: 50%; display: block; margin-left: auto; margin-right: auto;"/>

The weights of a convolution kernel are optimized to best predict the center pixel of each window, $\mathbf{X}_i$. The weights of the kernel (Fig. 2b) are linked based on distance from the center, e.g. the first three weights, $\{w_0, w_1, w_2\}$, correspond to indices in the kernel with distances $\{1, \sqrt{2}, 2\}$ from the center pixel. Convolution is the Frobenius inner product between the image and kernel, optimized to best predict the center pixel, $c_i \in \mathbf{X}$.

<img src="https://raw.githubusercontent.com/ryanhammonds/explorations/master/docs/decimation.png" width="600" style="width: 50%; display: block; margin-left: auto; margin-right: auto;"/>

Multiple convolution kernels are learned to account for various spatial scales in image. This is performed by decimating the image by various factors using the same kernel size, resulting in the kernel expanding by the decimation factor. The above image demonstrates this. Decimating the image by a factor of two results in the kernel expanding as shown in c.

## Datasets

Kylberg textures. Examples of each class:

![kylberg](https://github.com/voytekresearch/convolutional_ar/blob/3443b828577c830e4c27d640cc0981f6310c489f/docs/example_x.png)


## Results

A SVM found 99% test accuracy.

### Citations

Mao, J., & Jain, A. K. (1992). Texture classification and segmentation using multiresolution simultaneous autoregressive models. Pattern recognition, 25(2), 173-188.

Kylberg, G. (2011). Kylberg texture dataset v. 1.0. Centre for Image Analysis, Swedish University of Agricultural Sciences and Uppsala University.

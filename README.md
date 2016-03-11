# iPAUR
### (image Processing Algorithms at University of Regensburg)

The code is implemented for my master thesis found in the **src/LaTex** folder.

## Main Publications

> *Chambolle Antonin and Pock Thomas.* **A first-order primal-dual algorithm for convex problems with applications to imaging.** ”Journal of Mathematical Imaging and Vision”, 40(1):120–145, 2011.
> *A. Chambolle, V. Caselles, D. Cremers, M. Novaga, and T. Pock.* **An introduction to total variation for image analysis.** In ”Theoretical Foundations and Numerical Methods for Sparse Recovery”. De Gruyter, 2010.
> *T. Pock, D. Cremers, H. Bischof, and A. Chambolle.* **An algorithm for minimizing the piecewise smooth mumford-shah functional.** In ”IEEE International Conference on Computer Vision (ICCV)”, Kyoto, Japan, 2009.
> *E. Strekalovskiy and D. Cremers.* **Real-time minimization of the piecewise smooth mumford-shah functional.** In ”European Conference on Computer Vision (ECCV)”, pages 127–141, 2014. https://github.com/tum-vision/fastms.

## Image References

* **img/examples** is taken from https://github.com/tum-vision/fastms.git and belongs to the last publication in the list above
* **img/lena** is the well known **Lena** test image
* **img/pock** are images taken from the third publication of the above list
* **img/keating** is a scan of a brain tumor provided by Steven Keating, see for instance http://stevenkeating.info/main.html

## Implemented Algorithms and Basic Image Processing

* Serial algorithms:
  * ROF-Model
  * TV-L1-Model
  * Image Inpainting
  * Huber-ROF-Model
  * Real-Time Minimizer for the Mumford-Shah functional
* Other basic serial algorithms:
  * mean value blur
  * gaussian blur
  * canny edge detection
  * dilatation
  * erosion
  * duto filter
  * gradient filter (Sobel, Prewitt, Robert's Cross)
  * color space conversions
  * inverse image
  * laplace filter (operator)
  * median filter
* Parallel algorithms:
  * Primal-Dual Algorithm to solve the convex relaxed Mumford-Shah functional using Dykstra's projection algorithm
  * Primal-Dual Algorithm to solve the convex relaxed Mumford-Shah functional using an approach with Lagrange multipliers

## Requirements

### CUDA:

For the use of the GPU implementation, [CUDA](https://developer.nvidia.com/cuda-downloads) is needed.

### OpenCV

For reading and writing images [OpenCV](http://opencv.org/downloads.html) is needed.
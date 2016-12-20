# Massively Parallel Autoencoders on CUDA


### Getting Started
Included in this repository are a small test and train dataset, `simple_test.txt` and `simple_train.txt` respectively.

Alternatively you can download the MNIST data via: 

1. Unzip each of the MNIST data files (all the zipped files are included in this repository).

```bash
gunzip http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
gunzip http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
gunzip http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
```
2. Convert MNIST data from binary to a CSV file via `python convert_mnist_to_txt.py`

After extracting the data, all parameters (including which data files to train on) can be set in the header file `dA.h` or `dA_naive.h`. Sensible defaults given for these.

### Compiling
Compile the programs with the MAKEFILE included in this repository. Presently, the MAKEFILE will not compile the naive implementation (although this code can be uncommented to do so).



### Literature
[1] Pascal Vincent, Hugo Larochelle, Isabelle Lajoie, YoshuaBengio, and Pierre-Antoine Manzagol. Stacked denoisingautoencoders: Learning useful representations in a deepnetwork with a local denoising criterion. 11:3371–3408,December 2010.

[2] Geoffrey E Hinton and Ruslan R Salakhutdinov. Reducing thedimensionality of data with neural networks.Science,313(5786):504–507, 2006.

[3] Pascal Vincent, Hugo Larochelle, Yoshua Bengio, andPierre-Antoine Manzagol. Extracting and composing robustfeatures with denoising autoencoders. InProceedings of the25th international conference on Machine learning, pages1096–1103. ACM, 2008.

[4] David B. Kirk and Wen-mei W. Hwu.Programming MassivelyParallel Processors: A Hands-on Approach. Morgan KaufmannPublishers Inc., San Francisco, CA, USA, 1st edition, 2010.

[5]Olivier Bousquet and Leon Bottou. The tradeoffs of large scalelearning, 2008.



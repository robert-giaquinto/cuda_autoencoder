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
https://cs224d.stanford.edu/reports/OshriBarak.pdf

http://timdettmers.com/2014/10/09/deep-learning-data-parallelism/

http://link.springer.com/chapter/10.1007/978-3-642-31656-2_140

2008 Paper on Denoising Autoencoders:
https://www.iro.umontreal.ca/~vincentp/Publications/denoising_autoencoders_tr1316.pdf



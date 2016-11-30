#ifndef _DA_KERNEL_H_
#define _DA_KERNEL_H_

#include <stdio.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include "dA.h" // dA struct

#define RANDOM_MAX 100

__global__ void dA_train_kernel(dA da, int *X_d, double learning_rate, double corruption_level);
__device__ int binomial_kernel(int n, double p);
__device__ double sigmoid_kernel(double x);
__global__ void dA_get_corrupted_input_kernel(int length, int *tilde_x, double p);
__global__ void dA_get_hidden_values_kernel(int n_hidden, int n_visible,double *dW, double *dhbias,
				 int *x, double *y, int ib, int batchsize);
//__device__ void dA_get_reconstructed_input_kernel(dA *model, double *y, double *z);
__global__ void dA_get_reconstructed_input_kernel(int n_hidden, int n_visible,double *dW, double *dvbias,
				 double *z, double *y, int ib, int batchsize);


/////////////////////////////////////////////////////////////////
// train functions called from host:

// 1. using global memory
__global__ void dA_train_kernel(dA da, int *X_d, double learning_rate, double corruption_level) {
  da.hbias[0] = 999.0;
  da.W_flat[0] = 999.0;
}



////////////////////////////////////////////////////////////////
// helper functions needed by the training function

//NOTE: cuda kernal may not have rand() and RAND_MAX!!!!
__device__ int binomial_kernel(int n, double p) {
  if (p < 0 || p > 1) return 0;

  // init cuda random state
  curandState_t state;
  curand_init(0,0,0, &state);

  int i;
  int c = 0;
  double r;

  for (i = 0; i < n; ++i) {
    r = curand_uniform_double(&state);  // should be between 0 and 1
    if (r < p) c++;
  }
  return c;
}


__device__ double sigmoid_kernel(double x) {
  return 1.0 / (1.0 + exp(-x));
}


__device__ void dA_get_corrupted_input_kernel(int length, int *tilde_x, double p) {

  int i = blockDim.x * blockIdx.x + threadIdx.x ;
  //for(i=0; i<model->n_visible; i++) {
  if (i < length){
  if(tilde_x[i] == 0) {
      tilde_x[i] = 0;
    } else {
      tilde_x[i] = binomial_kernel(1, p);
    }
  }
 // tilde_x[i] = 99;
 // }

}


__global__ void dA_get_hidden_values_kernel(int n_hidden, int n_visible,double *dW, double *dhbias,
				 int *x, double *y, int ib, int batchsize) {

  // We have weight matrix which is fixed n_hidden x n_feats
  // [batchsize][y[n_hidden] = w[hidden][n_feats] * x[n_feats] + hbias[n_hidden]] 
  int idx = blockDim.x * blockIdx.x + threadIdx.x; // row in batch
  int shiftYIdx = (ib * batchsize + threadIdx.x)*n_hidden ;
  int shiftXIdx = (ib * batchsize + threadIdx.x)*n_visible;
  //dW[0] = 11.0;
  if (threadIdx.x < batchsize) {
    int i,j;
    for (i=0; i< n_hidden; i++) {
      y[shiftYIdx+i] = 0;
      for (j=0; j< n_visible; j++) {
        y[shiftYIdx+i] += dW[i*n_hidden+j] * x[shiftXIdx+j];
      }
      y[shiftYIdx+i] += dhbias[i];
      y[shiftYIdx+i] = sigmoid_kernel(y[shiftYIdx+i]);
      //dW[0] = shiftYIdx+i;
    }
  }
  __syncthreads();

}


__global__ void dA_get_reconstructed_input_kernel(int n_hidden, int n_visible,double *dW, double *dvbias,
				 double *z, double *y, int ib, int batchsize) {

  int idx = blockDim.x * blockIdx.x + threadIdx.x; // row in batch
  int shiftYIdx = (ib * batchsize + threadIdx.x)*n_hidden ;
  int shiftZIdx = (ib * batchsize + threadIdx.x)*n_visible;

  if (threadIdx.x < batchsize)  {
  int i, j;
  for(i=0; i<n_visible; i++) {
    z[shiftZIdx+i] = 0;
    for(j=0; j<n_hidden; j++) {
      z[shiftZIdx+i] += dW[j*n_hidden+i] * y[shiftYIdx+j];
    }
    z[shiftZIdx+i] += dvbias[i];
    z[shiftZIdx+i] = sigmoid_kernel(z[shiftZIdx+i]);
  }
  }

  z[0] = 1.2;
  __syncthreads();

}


#endif // #ifndef _DA_KERNEL_H_

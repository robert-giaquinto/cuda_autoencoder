#ifndef _DA_KERNEL_H_
#define _DA_KERNEL_H_

#include <stdio.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include "dA.h" // dA struct


__global__ void dA_train_kernel(dA model, int *X_d, double learning_rate, double corruption_level, int iter);
__device__ int binomial_kernel(int n, double p);
__device__ double sigmoid_kernel(double x);
__device__ void dA_get_corrupted_input_kernel(dA *model, int *x, int *tilde_x, double p);
__device__ void dA_get_hidden_values_kernel(dA *model, int *x, double *y);
__device__ void dA_get_reconstructed_input_kernel(dA *model, double *y, double *z);



/////////////////////////////////////////////////////////////////
// train functions called from host:

// 1. using global memory, intermediate results may as well be in shared
__global__ void dA_train_kernel(dA model, int *X_d, double learning_rate, double corruption_level, int iter) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  // skip rows corresponding to previous mini-batches
  int start = gridDim.x * N_FEATS * iter;

  // intialize intermediate pieces in shared memory
  // biases:
  __shared__ double L_vbias[N_FEATS];
  __shared__ double L_hbias[N_HIDDEN];
  if (tid < N_HIDDEN) {
    L_hbias[tid] = 0.0;
  }
  __syncthreads();
  // tilde_x: da_get_corrupted_input()
  __shared__ int tilde_x[N_FEATS];
  if (X_d[start + bid*N_FEATS + tid] == 0) {
    tilde_x[tid] = 0;
  } else {
    tilde_x[tid] = binomial_kernel(1, 1.0 - corruption_level);
  }
  __syncthreads();

  // get_hidden_values() : y
  __shared__ double y[N_HIDDEN];
  if (tid < N_HIDDEN) {
    y[tid] = 0.0;
  }
  __syncthreads(); // is this needed?
  for (int h=0; h < N_HIDDEN; ++h) {
    y[h] += model.W_flat[h*N_FEATS + tid] * X_d[start + bid*N_FEATS + tid];
  }
  __syncthreads();  // is this needed?
  if (tid < N_HIDDEN) {
    y[tid] += model.hbias[tid];
    y[tid] = sigmoid_kernel(y[tid]);
  }
  __syncthreads();

  // get_reconstructed_input() : z
  __shared__ double z[N_FEATS];
  z[tid] = 0.0;
  for (int h=0; h < N_HIDDEN; ++h) {
    z[tid] += model.W_flat[h*N_FEATS + tid] * y[h];
  }
  z[tid] += model.vbias[tid];
  z[tid] = sigmoid_kernel(z[tid]);
  __syncthreads();
  

  // update vbias, each thread grabs a column of X_d, each block works on N_OBS / BATCH_SIZE rows
  L_vbias[tid] = X_d[start + bid * N_FEATS + tid] - z[tid];
  model.vbias[tid] += learning_rate * L_vbias[tid] / model.N;
  __syncthreads();  

  // update hbias
  for (int h=0; h < N_HIDDEN; ++h) {
    L_hbias[h] += model.W_flat[h*N_FEATS + tid] * L_vbias[tid];
  }
  if (tid < N_HIDDEN) {
    L_hbias[tid] *= y[tid] * (1 - y[tid]);
    model.hbias[tid] += learning_rate * L_hbias[tid] / model.N;
  }
  __syncthreads();  

  // Update weights
  for (int h=0; h < N_HIDDEN; ++h) {
    model.W_flat[h * N_FEATS + tid] += learning_rate * (L_hbias[h] * tilde_x[tid] + L_vbias[tid] * y[h]) / model.N;
  }

}



////////////////////////////////////////////////////////////////
// helper functions needed by the training function

//NOTE: cuda kernal may not have rand() and RAND_MAX!!!!
// NOTE MAKE THIS USE THE OPTIMIZED FUNCTIONS!!!
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


__device__ void dA_get_corrupted_input_kernel(dA* model, int *x, int *tilde_x, double p) {

}


__device__ void dA_get_hidden_values_kernel(dA* model, int *x, double *y) {

}


__device__ void dA_get_reconstructed_input_kernel(dA* model, double *y, double *z) {

}


#endif // #ifndef _DA_KERNEL_H_

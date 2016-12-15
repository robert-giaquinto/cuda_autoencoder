#ifndef _DA_KERNEL_H_
#define _DA_KERNEL_H_

#include <stdio.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include "dA.h" // dA struct


__global__ void dA_train_kernel(dA model, float *X_d, double learning_rate, double corruption_level, int iter, curandState *state);
__device__ float binomial_kernel(int n, double p, curandState *state, int id);
__device__ double sigmoid_kernel(double x);
__device__ void dA_get_corrupted_input_kernel(dA *model, float *x, float *tilde_x, double p);
__device__ void dA_get_hidden_values_kernel(dA *model, float *x, double *y);
__device__ void dA_get_reconstructed_input_kernel(dA *model, double *y, double *z);

__device__ double atomicAdd(double* address, double val);
__device__ double atomicMultiply(double* address, double val);

/////////////////////////////////////////////////////////////////
// train functions called from host:

// 1. using global memory, intermediate results may as well be in shared
__global__ void dA_train_kernel(dA model, float *X_d, double learning_rate, double corruption_level, int iter, curandState *state) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  // skip rows corresponding to previous mini-batches
  int start = gridDim.x * N_FEATS * iter;

  // random seed
  int id = bid * blockDim.x + tid;
  curand_init(iter + 1000*bid, id, 0, &state[id]);

  // intialize intermediate pieces in shared memory
  // biases:
  __shared__ double L_vbias[N_FEATS];
  __shared__ double L_hbias[N_HIDDEN];
  if (tid < N_HIDDEN) {
    L_hbias[tid] = 0.0;
  }
  __syncthreads();
  // tilde_x: da_get_corrupted_input()
  __shared__ float tilde_x[N_FEATS];
  if (X_d[start + bid*N_FEATS + tid] == 0) {
    tilde_x[tid] = 0;
  } else {
    tilde_x[tid] = binomial_kernel(1, 1.0 - corruption_level, state, id);
  }
  __syncthreads();

  // get_hidden_values() : y
  __shared__ double y[N_HIDDEN];
  if (tid < N_HIDDEN) {
    y[tid] = 0.0;
  }
  __syncthreads();
  for (int h=0; h < N_HIDDEN; ++h) {
    atomicAdd(&y[h], model.W_flat[h*N_FEATS + tid] * X_d[start + bid*N_FEATS + tid]);
    //y[h] += model.W_flat[h*N_FEATS + tid] * X_d[start + bid*N_FEATS + tid];
  }
  __syncthreads();
  if (tid < N_HIDDEN) {
    //atomicAdd(&y[tid], model.hbias[tid]);
    y[tid] += model.hbias[tid];
    y[tid] = sigmoid_kernel(y[tid]);
  }
  __syncthreads();

  // get_reconstructed_input() : z
  __shared__ double z[N_FEATS];
  z[tid] = 0.0;
  for (int h=0; h < N_HIDDEN; ++h) {
    //atomicAdd(&z[tid], model.W_flat[h*N_FEATS + tid] * y[h]);
    z[tid] += model.W_flat[h*N_FEATS + tid] * y[h];
  }
  __syncthreads();
  //atomicAdd(&z[tid], model.vbias[tid]);
  z[tid] += model.vbias[tid];
  z[tid] = sigmoid_kernel(z[tid]);
  __syncthreads();
  

  // update vbias, each thread grabs a column of X_d, each block works on N_OBS / BATCH_SIZE rows
  L_vbias[tid] = X_d[start + bid * N_FEATS + tid] - z[tid];
  //atomicAdd(&model.vbias[tid], learning_rate * L_vbias[tid] / model.N);
  model.vbias[tid] += learning_rate * L_vbias[tid] / model.N;
  __syncthreads();

  // update hbias
  for (int h=0; h < N_HIDDEN; ++h) {
    atomicAdd(&L_hbias[h], model.W_flat[h*N_FEATS + tid] * L_vbias[tid]);
    //L_hbias[h] += model.W_flat[h*N_FEATS + tid] * L_vbias[tid];
  }
  if (tid < N_HIDDEN) {
    //atomicMultiply(&L_hbias[tid], y[tid] * (1.0 - y[tid]));
    //atomicAdd(&model.hbias[tid], learning_rate * L_hbias[tid] / model.N);
    L_hbias[tid] *= y[tid] * (1.0 - y[tid]);
    model.hbias[tid] += learning_rate * L_hbias[tid] / model.N;
  }
  __syncthreads();  

  // Update weights
  for (int h=0; h < N_HIDDEN; ++h) {
    atomicAdd(&model.W_flat[h*N_FEATS + tid], learning_rate * (L_hbias[h] * tilde_x[tid] + L_vbias[tid] * y[h]) / (double)model.N / (double)BATCH_SIZE);
    //model.W_flat[h * N_FEATS + tid] += learning_rate * (L_hbias[h] * tilde_x[tid] + L_vbias[tid] * y[h]) / model.N;
  }

}



////////////////////////////////////////////////////////////////
// helper functions needed by the training function

//NOTE: cuda kernal may not have rand() and RAND_MAX!!!!
// NOTE MAKE THIS USE THE OPTIMIZED FUNCTIONS!!!
__device__ float binomial_kernel(int n, double p, curandState *state, int id) {
  if (p < 0 || p > 1) return 0;

  int i;
  float c = 0.0f;
  double r;

  for (i = 0; i < n; ++i) {
    r = curand_uniform_double(&state[id]);  // should be between 0 and 1
    if (r < p) c += 1.0f;
  }
  return c;
}


__device__ double sigmoid_kernel(double x) {
  return 1.0 / (1.0 + exp(-x));
}


__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
      (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed,
                      __double_as_longlong(val +
                                           __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}


__device__ double atomicMultiply(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed,
                      __double_as_longlong(val *
                                           __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}



__device__ void dA_get_corrupted_input_kernel(dA* model, int *x, int *tilde_x, double p) {

}


__device__ void dA_get_hidden_values_kernel(dA* model, int *x, double *y) {

}


__device__ void dA_get_reconstructed_input_kernel(dA* model, double *y, double *z) {

}


#endif // #ifndef _DA_KERNEL_H_

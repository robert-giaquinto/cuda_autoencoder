#ifndef _DA_KERNEL_H_
#define _DA_KERNEL_H_

#include <stdio.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include "dA.h" // dA struct

#define RANDOM_MAX 100

__global__ void dA_train_kernel();
__device__ int binomial_kernel(int n, double p);
__device__ double sigmoidl_kernel(double x);
__device__ void dA_get_corrupted_inputl_kernel(dA *model, int *x, int *tilde_x, double p);
__device__ void dA_get_hidden_valuesl_kernel(dA *model, int *x, double *y);
__device__ void dA_get_reconstructed_inputl_kernel(dA *model, double *y, double *z);



/////////////////////////////////////////////////////////////////
// train functions called from host:

// 1. using global memory
__global__ void dA_train_kernel() {

}



////////////////////////////////////////////////////////////////
// helper functions needed by the training function

//NOTE: cuda kernal may not have rand() and RAND_MAX!!!!
__device__ int binomiall_kernel(int n, double p) {
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


__device__ double sigmoidl_kernel(double x) {
  return 1.0 / (1.0 + exp(-x));
}


__device__ void dA_get_corrupted_inputl_kernel(dA* model, int *x, int *tilde_x, double p) {

}


__device__ void dA_get_hidden_valuesl_kernel(dA* model, int *x, double *y) {

}


__device__ void dA_get_reconstructed_inputl_kernel(dA* model, double *y, double *z) {

}


#endif // #ifndef _DA_KERNEL_H_

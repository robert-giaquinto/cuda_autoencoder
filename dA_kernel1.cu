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
__global__ void dA_get_hidden_values_kernel(int n_hidden, int n_visible, double *dW, double *dhbias, int *x, double *y, int ib, int batchsize);
__global__ void dA_get_reconstructed_input_kernel(int n_hidden, int n_visible,double *dW, double *dvbias,
				 double *z, double *y, int ib, int batchsize);
__global__ void dA_L_vbias_kernel(int N, double *dL_vbias, double *dvbias, int n_visible, int *x, double *z, int ib, int batchsize,double lr);
__global__ void dA_L_hbias_kernel(int N, double *dL_vbias, double *dL_hbias, double *dhbias, int n_hidden, int n_visible, double *y, double *dW, 
		int ib, int batchsize,double lr);
__global__ void dA_W_kernel(int N,double *dL_vbias,double *dL_hbias, int n_hidden, int n_visible, double *y, double *dW, 
		int *tilde_x, int ib, int batchsize,double lr);
/////////////////////////////////////////////////////////////////
// train functions called from host:

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
  if (i < length){
  if(tilde_x[i] == 0) {
      tilde_x[i] = 0;
    } else {
      tilde_x[i] = binomial_kernel(1, p);
    }
  }
}


__global__ void dA_get_hidden_values_kernel(int n_hidden, int n_visible, double *dW, double *dhbias, int *x, double *y, int ib, int batchsize) {

  //*
  //int idx = blockDim.x * blockIdx.x + threadIdx.x; // row in batch
  if (threadIdx.x < batchsize) {
    int i,j;
    double tempYi = 0.0;
    for (i=0; i< n_hidden; i++) {
      tempYi = 0.0;
      for (j=0; j< n_visible; j++) {
        tempYi += dW[i*n_hidden+j] * x[ib*batchsize*n_visible+j];
      }
      tempYi += dhbias[i];
      //atomicAdd(&y[shiftYIdx+i], dhbias[i]);
      //y[i] = sigmoid_kernel(tempYi);
      y[i] = sigmoid_kernel(tempYi);
    }
  }
  __syncthreads();

}


__global__ void dA_get_reconstructed_input_kernel(int n_hidden, int n_visible,double *dW, double *dvbias,
				 double *z, double *y, int ib, int batchsize) {

  int shiftZIdx = ib * batchsize *n_visible;
  //
  if (threadIdx.x < batchsize)  {
  int i, j;
  for(i=0; i<n_visible; i++) {
    z[shiftZIdx+i] = 0;
    for(j=0; j<n_hidden; j++) {
      z[shiftZIdx+i] += dW[j*n_visible+i] * y[j];
    }
    z[shiftZIdx+i] += dvbias[i];
    z[shiftZIdx+i] = sigmoid_kernel(z[shiftZIdx+i]);
  }
  }

  __syncthreads();

}


__global__ void dA_L_vbias_kernel(int N, double *dL_vbias, double *dvbias, int n_visible, int *x, double *z, int ib, int batchsize,double lr) {

  int shiftZIdx = ib * batchsize *n_visible;
  int shiftXIdx = ib * batchsize*n_visible;
  
  double templvbias = 0.0;
  if (threadIdx.x < batchsize)  {
     for(int i=0; i<n_visible; i++) {
	templvbias = x[shiftXIdx + i] - z[shiftZIdx + i];
	dL_vbias[i] = templvbias;
	atomicAdd(&dvbias[i],(lr*templvbias / N));
     }
  }
  __syncthreads();
}

__global__ void dA_L_hbias_kernel(int N,double *dL_vbias,double *dL_hbias, double *dhbias, int n_hidden, int n_visible, double *y, double *dW, 
		int ib, int batchsize,double lr) {

  double templhbias;
  if (threadIdx.x < batchsize)  {
     for(int i=0; i<n_hidden; i++) {
	templhbias = 0.0;
	for (int j=0;j<n_visible;j++){
	  templhbias += dW[i*n_visible+j] * dL_vbias[j];
	}
	templhbias *= y[i]*(1-y[i]);
	dL_hbias[i] = templhbias;
	atomicAdd(&dhbias[i], (lr * templhbias/N) );
	//dhbias[i] += (lr * templhbias/N);
     }
  }
  __syncthreads();
}

__global__ void dA_W_kernel(int N,double *dL_vbias,double *dL_hbias, int n_hidden, int n_visible, double *y, double *dW, 
		int *tilde_x, int ib, int batchsize,double lr) {

  int shiftTildeXIdx = ib * batchsize *  n_visible;
  double tempVal;
  if (threadIdx.x < batchsize)  {
     for(int i=0; i<n_hidden; i++) {
	tempVal = 0.0;
	for (int j=0;j<n_visible;j++){
	  tempVal = lr * (dL_hbias[i]*tilde_x[shiftTildeXIdx+j] + dL_vbias[j]*y[i]) / N;
	  atomicAdd(&dW[i*n_visible+j], tempVal);
	}
     }
  }
  __syncthreads();
}

#endif // #ifndef _DA_KERNEL_H_

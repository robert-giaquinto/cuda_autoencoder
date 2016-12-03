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
__global__ void dA_get_hidValues_kernel(int n_hidden, int n_visible, double *dW, double *dhbias, int *x, double *y, int ib, int batchsize);
//__device__ void dA_get_reconstructed_input_kernel(dA *model, double *y, double *z);
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


__global__ void dA_get_hidValues_kernel(int n_hidden, int n_visible, double *dW, double *dhbias, int *x, double *y, int ib, int batchsize) {

  // We have weight matrix which is fixed n_hidden x n_feats
  // [batchsize][y[n_hidden] = w[hidden][n_feats] * x[n_feats] + hbias[n_hidden]] 
  //*
  int idx = blockDim.x * blockIdx.x + threadIdx.x; // row in batch
  int shiftYIdx = (ib * batchsize + threadIdx.x)*n_hidden ;
  int shiftXIdx = (ib * batchsize + threadIdx.x)*n_visible;
  //int shiftXIdx = ib *n_visible;
  shiftYIdx = 0;
  shiftXIdx = 0;
  //*/
  //dW[0] = 101.0;//y[0] = 11.0;
  //*
  //if (threadIdx.x < batchsize) {
    int i,j;
    double tempYi = 0.0;
    for (i=0; i< n_hidden; i++) {
      //y[shiftYIdx+i] = 0;
      tempYi = 0.0;
      for (j=0; j< n_visible; j++) {
        //y[shiftYIdx+i] += dW[i*n_hidden+j] * x[shiftXIdx+j];
        tempYi += dW[i*n_hidden+j] * x[shiftXIdx+j];
        //atomicAdd(&y[shiftYIdx+i] , dW[i*n_hidden+j] * x[shiftXIdx+j]);
      }
      //y[shiftYIdx+i] += dhbias[i];
      tempYi += dhbias[i];
      //atomicAdd(&y[shiftYIdx+i], dhbias[i]);
      //y[shiftYIdx+i] = sigmoid_kernel(y[shiftYIdx+i]);
      y[i] = sigmoid_kernel(tempYi);
      //y[shiftYIdx+i] = 0.1;
      //y[shiftYIdx+i] = tempYi;
    }
  //*/
  //}
  __syncthreads();

}


__global__ void dA_get_reconstructed_input_kernel(int n_hidden, int n_visible,double *dW, double *dvbias,
				 double *z, double *y, int ib, int batchsize) {

  int idx = blockDim.x * blockIdx.x + threadIdx.x; // row in batch
  int shiftYIdx = (ib * batchsize + threadIdx.x)*n_hidden ;
  int shiftZIdx = (ib * batchsize + threadIdx.x)*n_visible;
  //
  shiftYIdx = 0;
  shiftZIdx = 0;
  //
  //if (threadIdx.x < batchsize)  {
  int i, j;
  for(i=0; i<n_visible; i++) {
    z[shiftZIdx+i] = 0;
    for(j=0; j<n_hidden; j++) {
      z[shiftZIdx+i] += dW[j*n_hidden+i] * y[shiftYIdx+j];
    }
    z[shiftZIdx+i] += dvbias[i];
    z[shiftZIdx+i] = sigmoid_kernel(z[shiftZIdx+i]);
  }
  //}

  //z[0] = 1.2;
  __syncthreads();

}


__global__ void dA_L_vbias_kernel(int N, double *dL_vbias, double *dvbias, int n_visible, int *x, double *z, int ib, int batchsize,double lr) {

  int idx = blockDim.x * blockIdx.x + threadIdx.x; // row in batch
  int shiftZIdx = (ib * batchsize + threadIdx.x)*n_visible;
  int shiftXIdx = (ib * batchsize + threadIdx.x)*n_visible;
  int lvbiasIdx = (ib * batchsize + threadIdx.x)*n_visible;
  //shiftZIdx = 0;
  //shiftXIdx = 0;
  lvbiasIdx = 0;
  double templvbias = 0.0;
  //if (threadIdx.x < batchsize)  {
     for(int i=0; i<n_visible; i++) {
	templvbias = x[shiftXIdx + i] - z[shiftZIdx + i];
	dL_vbias[lvbiasIdx + i] = templvbias;
	dvbias[i] += (lr*templvbias / N);
     }
  //}
  __syncthreads();
}

__global__ void dA_L_hbias_kernel(int N,double *dL_vbias,double *dL_hbias, double *dhbias, int n_hidden, int n_visible, double *y, double *dW, 
		int ib, int batchsize,double lr) {

  int idx = blockDim.x * blockIdx.x + threadIdx.x; // row in batch
  int shiftYIdx = (ib * batchsize + threadIdx.x)*n_hidden;
  int lhbiasIdx = (ib * batchsize + threadIdx.x)*n_hidden;
  double templhbias;
  shiftYIdx = 0;
  lhbiasIdx = 0;
  //if (threadIdx.x < batchsize)  {
     for(int i=0; i<n_hidden; i++) {
	templhbias = 0.0;
	for (int j=0;j<n_visible;j++){
	  templhbias += dW[i*n_hidden+j] * dL_vbias[j];
	}
	templhbias *= y[shiftYIdx+i]*(1-y[shiftYIdx+i]);
	dL_hbias[lhbiasIdx + i] = templhbias;
	//atomicAdd(&dhbias[i], (lr * templhbias/N) );
	dhbias[i] += (lr * templhbias/N);
     }
  //}
  __syncthreads();
}

__global__ void dA_W_kernel(int N,double *dL_vbias,double *dL_hbias, int n_hidden, int n_visible, double *y, double *dW, 
		int *tilde_x, int ib, int batchsize,double lr) {

  int idx = blockDim.x * blockIdx.x + threadIdx.x; // row in batch
  int shiftYIdx = (ib * batchsize + threadIdx.x)*n_hidden;
  int shiftTildeXIdx = (ib * batchsize + threadIdx.x)*n_visible;
  int lhbiasIdx = (ib * batchsize + threadIdx.x)*n_hidden;
  int lvbiasIdx = (ib * batchsize + threadIdx.x)*n_visible;
  lhbiasIdx = 0;
  lvbiasIdx = 0;
  shiftYIdx = 00;
  double tempVal;
  //if (threadIdx.x < batchsize)  {
     for(int i=0; i<n_hidden; i++) {
	tempVal = 0.0;
	for (int j=0;j<n_visible;j++){
	  tempVal = lr*(dL_hbias[lhbiasIdx+i]*tilde_x[shiftTildeXIdx+j] + dL_vbias[lvbiasIdx+j]*y[shiftYIdx+i]) / N;
	  atomicAdd(&dW[i*n_hidden+j], tempVal );
	}
     }
  //}
  __syncthreads();
}

#endif // #ifndef _DA_KERNEL_H_

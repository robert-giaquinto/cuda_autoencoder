#ifndef _DA_KERNEL_H_
#define _DA_KERNEL_H_

#include <stdio.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include "dA.h" // dA struct and constants

__global__ void dA_train_kernel(dA da, float *X_d, double learning_rate, double corruption_level);
__device__ float binomial_kernel(int n, double p);
__device__ double sigmoid_kernel(double x);
__global__ void dA_get_corrupted_input_kernel(int lenTileX, float *x, float *tilde_x, double p, int offsetX);
__global__ void dA_get_hidden_values_kernel(int n_hidden, int n_visible, double *dW, double *dhbias, float *x, double *y, int ib);
__global__ void dA_get_reconstructed_input_kernel(int n_hidden, int n_visible,double *dW, double *dvbias,double *z, double *y, int ib, int batchsize);
__global__ void dA_L_vbias_kernel(int N, double *dL_vbias, double *dvbias, int n_visible, float *x, double *z, int offsetX, int batchsize,double lr);
__global__ void dA_L_hbias_kernel(int N, double *dL_vbias, double *dL_hbias, double *dhbias, int n_hidden, int n_visible, double *y, double *dW,int ib, int batchsize,double lr);
__global__ void dA_W_kernel(int N,double *dL_vbias,double *dL_hbias, int n_hidden, int n_visible, double *yb, double *dW, float *tilde_x, int ib, int batchsize,double lr);

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
//*
__device__ float binomial_kernel(int n, double p) {
  if (p < 0 || p > 1) return 0;

  // init cuda random state
  curandState_t state;
  curand_init(0,0,0, &state);

  int i;
  float c = 0.0f;
  float r;

  for (i = 0; i < n; ++i) {
    r = curand_uniform_double(&state);  // should be between 0 and 1
    if (r < p) c += 1.0f;
  }
  return c;
}
//
__device__ double sigmoid_kernel(double x) {
  return 1.0 / (1.0 + exp(-x));
}

//
__global__ void dA_get_corrupted_input_kernel(int lenTileX, float *x, float *tilde_x, double p, int offsetX) {
  // We allow each thread to load one feature of one record. So, each block will have n_visible threads
  // while each record will be processed by a separate block in grid. So, gridsize is batch size i.e. no
  // of records in a batch.
  int idx = blockIdx.x*blockDim.x + threadIdx.x; // idx is a global index of each thread..
  if (idx < lenTileX) {
  	if(x[offsetX+idx] == 0) {
      		tilde_x[idx] = 0;
    	} else {
     		tilde_x[idx] = binomial_kernel(1,p);
    	}  
  }
}
//
__global__ void dA_get_hidden_values_kernel(int n_hidden, int n_visible, double *dW, double *dhbias, float *x, double *yb, int ib) {
  //*
  // yi = WX'
  //
  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
  
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int Row = by * TILE_WIDTH  + ty;
  int Col = bx * TILE_WIDTH  + tx;

  int nTiles = n_visible / TILE_WIDTH; // how many to load and calculate from each matrices
  if (n_visible % TILE_WIDTH) nTiles++;
  float Pvalue = 0.0;
  for (int m = 0; m < nTiles; m++) {
    if ((Row < n_hidden) && ((m*TILE_WIDTH + tx ) < n_visible))
	Mds[ty][tx] = dW[Row * n_visible + m*TILE_WIDTH + tx];
    else
	Mds[ty][tx] = 0.0;

    if ((Col < BATCHSIZE) && ((m*TILE_WIDTH + ty) < n_visible))
    	Nds[ty][tx] = x[(Col*n_visible) + (m*TILE_WIDTH + tx)];
    else
	Nds[ty][tx] = 0.0;
    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; ++k) {
 	Pvalue += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();
  }
 
  if ((Row < n_hidden) && (Col < BATCHSIZE))  {
     yb[(by * blockDim.y + ty)*BATCHSIZE + (bx * blockDim.x + tx)] = sigmoid_kernel(Pvalue + dhbias[Row]);
  }
  //__syncthreads();
	
}
//
__global__ void dA_get_reconstructed_input_kernel(int n_hidden, int n_visible,double *dW, double *dvbias,
				 double *z, double *y, int ib, int batchsize) {
  //* W'y = visible x batch
  __shared__ double Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ double Nds[TILE_WIDTH][TILE_WIDTH];
  
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int Row = by * TILE_WIDTH  + ty;
  int Col = bx * TILE_WIDTH  + tx;

  int nTiles = n_hidden / TILE_WIDTH;
  if (n_hidden % TILE_WIDTH) nTiles++;
  double Pvalue = 0.0;
  for (int m = 0; m < nTiles; m++) {
    if ((Row < n_visible) && ((m*TILE_WIDTH + tx) < n_hidden))
    	Mds[ty][tx] = dW[(m*TILE_WIDTH+tx)*n_visible+Row];
    	//Mds[ty][tx] = dW[Col*n_visible+(m*TILE_WIDTH+ty)];
    else
	Mds[ty][tx] = 0.0;

    if ((Col < BATCHSIZE) && ((m*TILE_WIDTH + ty) < n_hidden))
    	Nds[ty][tx] = y[(m*TILE_WIDTH + ty)* BATCHSIZE +Col];
    else
	Nds[ty][tx] = 0.0;
    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; ++k) {
	Pvalue += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();
  }
 
  if ((Row < n_visible) && (Col < BATCHSIZE))  {
     z[(by * blockDim.y + ty)*BATCHSIZE + (bx * blockDim.x + tx)] = sigmoid_kernel(Pvalue + dvbias[Row]);
  }
  //__syncthreads();
}

//
__global__ void dA_L_vbias_kernel(int N, double *dL_vbias, double *dvbias, int n_visible, float *x, double *z, int offsetX, int batchsize,double lr) {
  // We allow each thread to load one feature of one record. So, each block will have n_visible threads
  // while each record will be processed by a separate block in grid. So, gridsize is batch size i.e. no
  // of records in a batch.
  int idx = blockIdx.x * n_visible + threadIdx.x; // idx is a global index of each thread..
  double templvbias = 0.0;
  double tempAddlvbias = 0.0;
  if (idx < batchsize*n_visible) {
	templvbias = x[offsetX+idx] - z[threadIdx.x*batchsize+blockIdx.x];
	dL_vbias[idx] = templvbias;
	tempAddlvbias = (templvbias * lr / N);
	atomicAdd(&dvbias[threadIdx.x],tempAddlvbias);
  }
}

// 
__global__ void dA_L_hbias_kernel(int N,double *dL_vbias,double *dL_hbias, double *dhbias, int n_hidden, int n_visible, double *y, double *dW, 
		int ib, int batchsize,double lr) {
  //*
  // L_hbias = W x L_vbias'
  //
  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
  
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int Row = by * TILE_WIDTH  + ty;
  int Col = bx * TILE_WIDTH  + tx;

  int nTiles = n_visible / TILE_WIDTH; // how many to load and calculate from each matrices
  if (n_visible % TILE_WIDTH) nTiles++;
  float Pvalue = 0.0;
  for (int m = 0; m < nTiles; m++) {
    if ((Row < n_hidden) && ((m*TILE_WIDTH + tx ) < n_visible))
	Mds[ty][tx] = dW[Row * n_visible + m*TILE_WIDTH + tx];
    else
	Mds[ty][tx] = 0.0;

    if ((Col < BATCHSIZE) && ((m*TILE_WIDTH + ty) < n_visible))
    	Nds[ty][tx] = dL_vbias[(Col*n_visible) + (m*TILE_WIDTH + tx)];
    else
	Nds[ty][tx] = 0.0;
    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; ++k) {
 	Pvalue += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();
  }
 
  if ((Row < n_hidden) && (Col < BATCHSIZE))  {
     //
     double yi = 0.0;
     double templhbias = 0.0;
     yi = y[Row];
     templhbias = Pvalue * yi * (1 - yi);
     dL_hbias[(by * blockDim.y + ty)*BATCHSIZE + (bx * blockDim.x + tx)] = templhbias;
     atomicAdd(&dhbias[Row],(lr*templhbias/ N));
  }
  //__syncthreads();
  
}

//
__global__ void dA_W_kernel(int N, double *dL_vbias, double *dL_hbias, int n_hidden, int n_visible, double *yb, double *dW, 
		float *tilde_x, int ib, int batchsize,double lr) {
  // We will try to use hidden elements in global memory and try to put visible dimensions in shared memory
  __shared__ double tilde_x_s[N_FEATS];
  __shared__ double dL_vbias_s[N_FEATS];
  //
  // One hidden row in one block to be processed sp we are done with one row at once
  // first load two elements
 for (int batchIdx = 0; batchIdx < batchsize; batchIdx++){
 	tilde_x_s[threadIdx.x] = tilde_x[batchIdx*N_FEATS+threadIdx.x];
 	dL_vbias_s[threadIdx.x] = dL_vbias[batchIdx*N_FEATS+threadIdx.x];
 	//__syncthreads();
 	double tempVal = 0.0;
 	if (blockIdx.x < N_HIDDEN){
 		tempVal = dL_hbias[blockIdx.x*batchsize+batchIdx] * tilde_x_s[threadIdx.x];
     		tempVal += yb[blockIdx.x*batchsize+batchIdx] * dL_vbias_s[threadIdx.x];
     		tempVal *= lr / N;     
     		atomicAdd(&dW[blockIdx.x*n_visible+threadIdx.x], tempVal);
 	}
	//__syncthreads();	
}
  
}
//
#endif // #ifndef _DA_KERNEL_H_

#ifndef _DA_KERNEL_H_
#define _DA_KERNEL_H_

#include <stdio.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include "dA.h" // dA struct

#define RANDOM_MAX 100
#define N_FEATS 20
#define N_OBS 10
#define BATCHSIZE 1
#define N_HIDDEN 5
#define N_HIDXFEATS (N_HIDDEN*N_FEATS)
#define BLOCKSIZE 1
#define TILE_WIDTH BLOCKSIZE

__global__ void dA_train_kernel(dA da, int *X_d, double learning_rate, double corruption_level);
__device__ int binomial_kernel(double p);
__device__ double sigmoid_kernel(double x);
__global__ void dA_get_corrupted_input_kernel(int lenTileX, int *x, int *tilde_x, double p, int offsetX);
__global__ void dA_get_hidden_values_kernel2(int n_hidden, int n_visible, double *dW, double *dhbias, int *x, double *y, int ib);
__global__ void dA_get_hidden_values_kernel(int n_hidden, int n_visible, double *dW, double *dhbias, int *x, double *y, int ib);
__global__ void dA_get_hidden_values_batch_kernel2(int n_hidden, double *yb, double *y, double *dhbias);
__global__ void dA_get_reconstructed_input_kernel(int n_hidden, int n_visible,double *dW, double *dvbias,double *z, double *y, int ib, int batchsize);
__global__ void dA_L_vbias_kernel(int N, double *dL_vbias, double *dvbias, int n_visible, int *x, double *z, int ib, int batchsize,double lr);
__global__ void dA_L_hbias_kernel(int N, double *dL_vbias, double *dL_hbias, double *dhbias, int n_hidden, int n_visible, double *y, double *dW,int ib, int batchsize,double lr);
__global__ void dA_W_kernel(int N,double *dL_vbias,double *dL_hbias, int n_hidden, int n_visible, double *yb, double *dW,int *tilde_x, int ib, int batchsize,double lr);

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
    //r = curand_uniform_double(&state);  // should be between 0 and 1
    r = curand_uniform_double(&state);  // should be between 0 and 1
    if (r < p) c = 1;
    else c = 0;
  }
  return c;
}


__device__ double sigmoid_kernel(double x) {
  return 1.0 / (1.0 + exp(-x));
}

//
__global__ void dA_get_corrupted_input_kernel(int lenTileX, int *x, int *tilde_x, double p, int offsetX) {

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


__global__ void dA_get_hidden_values_kernel(int n_hidden, int n_visible, double *dW, double *dhbias, int *x, double *yb, int ib) {

  //*
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

    if ((Row < BATCHSIZE) && ((m*TILE_WIDTH + tx) < n_visible))
    	Nds[ty][tx] = x[Row * n_visible + m*TILE_WIDTH + tx];
    else
	Nds[ty][tx] = 0.0;
    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; ++k) {
	Pvalue += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();
  }
 
  if ((Row < n_hidden) && (Col < BATCHSIZE))  {
     //y[(by * blockDim.y + ty)* 1 + (bx * blockDim.x + tx)] = sigmoid_kernel(Pvalue + dhbias[(by * blockDim.y + ty)* n_visible + (bx * blockDim.x + tx)]);
     //yb[(by * blockDim.y + ty)* BATCHSIZE + (bx * blockDim.x + tx)] = Pvalue;
     //yb[(by * blockDim.y + ty) + (bx * blockDim.x + tx)*BATCHSIZE] = sigmoid_kernel(Pvalue + dhbias[Row]);
     yb[(by * blockDim.y + ty)*BATCHSIZE + (bx * blockDim.x + tx)] = sigmoid_kernel(Pvalue + dhbias[Row]);
     //y[ty] = sigmoid_kernel(Pvalue + dhbias[ty]);
  }
  __syncthreads();
  
  //if ((Row < n_hidden) && (Col < BATCHSIZE)) {
  //   y[Row] = sigmoid_kernel(y[Row]/BATCHSIZE + dhbias[Row]);
  //}
	
}

// It seems this is the most time consuming part now - we will use parallel scan  to leverage
// memory coalescing for this..since it is expected that batch size will be more than n_hidden, it might
// work better
/*
__global__ void dA_get_hidden_values_batch_kernel2(int n_hidden, double *yb, double *y, double *dhbias) {

  __shared__ double partialSum[N_HIDDEN][BATCHSIZE];
  int tx = threadIdx.x + blockDim.x*blockIdx.x;
  if (tx < BATCHSIZE) {
    for (int i=0;i<N_HIDDEN;i++) {
    	partialSum[i][threadIdx.x] = yb[i*BATCHSIZE+tx];
    }
  }
  //
  if (tx + (BATCHSIZE/2) < BATCHSIZE) {
    for (int i=0;i<N_HIDDEN;i++) {
    	partialSum[i][threadIdx.x+(BATCHSIZE/2)] = yb[i*BATCHSIZE+tx+(BATCHSIZE/2)];
    }
  }
  //
  for (int stride = BATCHSIZE/2; stride >= 1; stride >>= 1)
  {
     __syncthreads();
    if (threadIdx.x < stride) {
       for (int j=0;j<N_HIDDEN;j++)
           partialSum[j][threadIdx.x] += partialSum[j][threadIdx.x + stride];
    }
  }
  
  if (threadIdx.x == 0) {
     double tempPartSum = 0.0;
     for (int k=0;k<N_HIDDEN;k++) {
        tempPartSum = partialSum[k][0]/ BATCHSIZE;
     	y[k] = sigmoid_kernel(tempPartSum + dhbias[k]); 
     }
     //y[k] = partialSum[k][0]; 
  }
	
}
*/

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
    if ((Col < n_visible) && ((m*TILE_WIDTH + ty) < n_hidden))
    	Mds[ty][tx] = dW[(m*TILE_WIDTH + ty)* n_visible +Col];
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
  __syncthreads();
  

}

//
__global__ void dA_L_vbias_kernel(int N, double *dL_vbias, double *dvbias, int n_visible, int *x, double *z, int ib, int batchsize,double lr) {
  // We allow each thread to load one feature of one record. So, each block will have n_visible threads
  // while each record will be processed by a separate block in grid. So, gridsize is batch size i.e. no
  // of records in a batch.
  int idx = blockIdx.x*blockDim.x + threadIdx.x; // idx is a global index of each thread..
   
  double templvbias = 0.0;
  if (idx < batchsize*n_visible) {
	templvbias = x[idx] - z[threadIdx.x * batchsize + blockIdx.x];
	dL_vbias[idx] = templvbias;
	atomicAdd(&dvbias[threadIdx.x],(lr*templvbias / N));
  }
}

//
/*
__global__ void dA_L_vbias_kernel(int N, double *dL_vbias, double *dvbias, int n_visible, int *x, double *z, int ib, int batchsize,double lr) {

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  double templvbias = 0.0;
  if ((tx < batchsize) && (ty < n_visible)) {
	templvbias = x[ty*BATCHSIZE+tx] - z[ty*BATCHSIZE+tx];
	dL_vbias[ty*BATCHSIZE+tx] = templvbias;
	atomicAdd(&dvbias[ty],(lr*templvbias / N));
  }
  __syncthreads();
}
*/

// W * dL_vbias 
// 
__global__ void dA_L_hbias_kernel(int N,double *dL_vbias,double *dL_hbias, double *dhbias, int n_hidden, int n_visible, double *y, double *dW, 
		int ib, int batchsize,double lr) {
  //* W y' W is n_hidden x n_visible, y ix batchsize x n_visible
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

    if ((Row < BATCHSIZE) && ((m*TILE_WIDTH + tx) < n_visible))
    	Nds[ty][tx] = dL_vbias[Row * n_visible + m*TILE_WIDTH + tx];
    else
	Nds[ty][tx] = 0.0;
    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; ++k) {
	Pvalue += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();
  }
  //
  double yi = 0.0;
  double templhbias = 0.0;
  if ((Row < n_hidden) && (Col < BATCHSIZE))  {
     yi = y[(by * blockDim.y + ty)*BATCHSIZE + (bx * blockDim.x + tx)];
     templhbias = Pvalue * yi * (1 - yi);
     //dL_hbias[(by * blockDim.y + ty)*BATCHSIZE + (bx * blockDim.x + tx)] = templhbias;
     dL_hbias[Row*BATCHSIZE + Col] = Pvalue;
     atomicAdd(&dhbias[by * blockDim.y + ty],(lr*templhbias/N));
  }
  __syncthreads();
}

/*
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
*/
//
/*
__global__ void dA_W_kernel(int N, double *dL_vbias, double *dL_hbias, int n_hidden, int n_visible, double *y, double *dW, 
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
*/
//

__global__ void dA_W_kernel(int N, double *dL_vbias, double *dL_hbias, int n_hidden, int n_visible, double *yb, double *dW, 
		int *tilde_x, int ib, int batchsize,double lr) {

  int tx = threadIdx.x;
  double tempVal;
  if (tx < batchsize)  {
     for(int i=0; i<n_hidden; i++) {
	tempVal = 0.0;
	for (int j=0;j<n_visible;j++){
	  tempVal = lr * (dL_hbias[i]*tilde_x[j*batchsize+tx] + dL_vbias[j*batchsize+tx]*yb[i*batchsize+tx]) / N;
	  atomicAdd(&dW[i*n_visible+j], tempVal);
	}
     }
  }
  __syncthreads();
}
//
#endif // #ifndef _DA_KERNEL_H_

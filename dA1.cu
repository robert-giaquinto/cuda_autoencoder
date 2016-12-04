// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// includes, project
#include <cutil.h>

// includes, kernel
#include <dA_kernel1.cu>


#define N_FEATS 20
#define N_OBS 10
#define BATCHSIZE 1
#define N_HIDDEN 5

// declarations for CPU train functions
extern "C"
void dA_train_gold(dA*, int*, double, double);
void dA_get_hidden_values(dA*, int*, double*);
void dA_get_reconstructed_input(dA*, double*, double*);


// functions defined in this file are for intializing the autoencoder
double uniform(double min, double max);
void dA__construct(dA *model, int N, int n_visible, int n_hidden, double **W, double *hbias, double *vbais);
void dA__destruct(dA *model);
void dA_reconstruct(dA *model, int *x, double *z);
void test_dbn();
void dA_train_on_device(dA*, int[][N_FEATS], double, double);


// Begin definign functions
double uniform(double min, double max) {
  return rand() / (RAND_MAX + 1.0) * (max - min) + min;
}


void dA__construct(dA* model, int N, int n_visible, int n_hidden, double **W, double *hbias, double *vbias) {
  int i, j;
  double a = 1.0 / n_visible;

  model->N = N;
  model->n_visible = n_visible;
  model->n_hidden = n_hidden;

  if(W == NULL) {
    model->W = (double **)malloc(sizeof(double*) * n_hidden);
    model->W_flat = (double*)malloc(sizeof(double)*n_hidden*n_visible);
    model->W[0] = (double *)malloc(sizeof(double) * n_visible * n_hidden);
    for(i=0; i<n_hidden; i++) model->W[i] = model->W[0] + i * n_visible;

    for(i=0; i<n_hidden; i++) {
      for(j=0; j<n_visible; j++) {
        double u = uniform(-a, a);
        model->W_flat[i*n_visible + j] = u;
        model->W[i][j] = u;
      }
    }
  } else {
    model->W = W;
  }

  if(hbias == NULL) {
    model->hbias = (double *)malloc(sizeof(double) * n_hidden);
    for(i=0; i<n_hidden; i++) model->hbias[i] = 0;
  } else {
    model->hbias = hbias;
  }

  if(vbias == NULL) {
    model->vbias = (double *)malloc(sizeof(double) * n_visible);
    for(i=0; i<n_visible; i++) model->vbias[i] = 0;
  } else {
    model->vbias = vbias;
  }
}


void dA__destruct(dA* model) {
  free(model->W[0]);
  free(model->W);
  free(model->W_flat);
  free(model->hbias);
  free(model->vbias);
}


void dA_reconstruct(dA* model, int *x, double *z) {
  double *y = (double *)malloc(sizeof(double) * model->n_hidden);

  dA_get_hidden_values(model, x, y);
  dA_get_reconstructed_input(model, y, z);

  free(y);
}


int* flatten_array(int arr[N_OBS][N_FEATS]) {
  int *flat = (int *)malloc(sizeof(int) * N_OBS * N_FEATS);
  for (int i=0; i < N_OBS; ++i) {
    for (int j=0; j < N_FEATS; ++j) {
      flat[i*N_FEATS + j] = arr[i][j];
    }
  }
  return flat;
}

double* flatten_w(double **W, int n_visible, int n_hidden) {
  double *flat = (double *)malloc(sizeof(double) * n_visible * n_hidden);
  for (int i=0; i < n_hidden; ++i) {
    for (int j=0; j < n_visible; ++j) {
      flat[i*n_visible + j] = W[i][j];
    }
  }
  return flat;
}

int * allocate_device_x() {
  int *x_d = NULL;
  int size = N_OBS * N_FEATS * sizeof(int);
  cudaMalloc((void**)&x_d, size);
  return x_d;
}


double * allocate_device_z(int m, int n) {
  double *z_d = NULL;
  int size = m * n * sizeof(double);
  cudaMalloc((void**)&z_d, size);
  return z_d;
}

double*  allocate_device_dW() {
  double *dW_flat;
  int dW_size = sizeof(double) * N_HIDDEN * N_FEATS;
  cudaMalloc((void**)&dW_flat, dW_size);
  return dW_flat;
  
}

double* allocate_device_dhbias() {
  double *dhbias;
  int dhbias_size = sizeof(double) * N_HIDDEN;
  cudaMalloc((void**)&dhbias, dhbias_size);
  return dhbias;
}

double* allocate_device_dvbias() {
  double *dvbias;
  int dvbias_size = sizeof(double) * N_FEATS;
  cudaMalloc((void**)&dvbias, dvbias_size);
  return dvbias;  
}

double* allocate_device_dL_vbias(int m, int n) {
  double *dL_vbias;
  int dL_vbias_size = sizeof(double) * m * n;
  cudaMalloc((void**)&dL_vbias, dL_vbias_size);
  return dL_vbias;  
}

double* allocate_device_dL_hbias(int m,int n) {
  double *dL_hbias;
  int dL_hbias_size = sizeof(double) * m * n;
  cudaMalloc((void**)&dL_hbias, dL_hbias_size);
  return dL_hbias;  
}

void copy_x_to_device(int *x_d, int *x_h) {
  int size = N_OBS * N_FEATS * sizeof(int);
  cudaMemcpy(x_d, x_h, size, cudaMemcpyHostToDevice);
}

void copy_x_to_host(int *x_d, int *x_h) {
  int size = N_OBS * N_FEATS * sizeof(int);
  cudaMemcpy(x_h, x_d, size, cudaMemcpyDeviceToHost);
}

void dA_train_on_device1(dA *model_h, int train_X[N_OBS][N_FEATS], double learning_rate, double corruption_level,int training_epochs) {
  //
  cudaError_t cuda_ret;
  int epoch;

  printf("\nin device processing..."); 
  double p = 1 - corruption_level;  

  // flatten input array
  int *X_h = flatten_array(train_X);

  // allocate space on device
  int *X_d = allocate_device_x();
  int *tilde_x_d = allocate_device_x();
  int *tilde_x_h = (int*)malloc(sizeof(int)* N_OBS * N_FEATS);
  double *y_h = (double*)malloc(sizeof(double)* 1* N_HIDDEN);
  double *y_d = allocate_device_z(1,N_HIDDEN); 
  //double *hW_flat = (double*)malloc(sizeof(double)* N_HIDDEN * N_FEATS);
  double *dW_flat = allocate_device_dW();
  double *dhbias = allocate_device_dhbias();
  double *dvbias = allocate_device_dvbias(); 
  double *z_d = allocate_device_z(1, N_FEATS);
  double *z_h = (double*)malloc(sizeof(double)*1* N_FEATS);
  //
  double *L_vbias = (double *)malloc(sizeof(double) * 1 * N_FEATS);
  double *dL_vbias = allocate_device_dL_vbias(1,N_FEATS);
  double *L_hbias = (double *)malloc(sizeof(double) * 1 * N_HIDDEN);
  double *dL_hbias = allocate_device_dL_hbias(1, N_HIDDEN);
  //

  // copy data over to device
  copy_x_to_device(X_d, X_h);
  copy_x_to_device(tilde_x_d, X_h);
  //dA model_d = init_device_ae(*model_h);
  // copy over data
  cudaMemcpy(dW_flat, model_h->W_flat, sizeof(double)*N_HIDDEN*N_FEATS, cudaMemcpyHostToDevice);
  cudaMemcpy(dhbias, model_h->hbias, sizeof(double)*N_HIDDEN, cudaMemcpyHostToDevice);
  cudaMemcpy(dvbias, model_h->vbias, sizeof(double)*N_FEATS, cudaMemcpyHostToDevice);
  
  printf("X_h %d %d",X_h[1],X_h[2]);
  
  for(epoch=0; epoch<training_epochs; epoch++) {
	//  copy_ae_to_device(model_d, model_h);

  	//1. set up corrupted input for all together
  	dim3 dimGrid1(1);
     	dim3 dimBlock1(N_OBS*N_FEATS);
     	dA_get_corrupted_input_kernel<<<dimGrid1, dimBlock1>>>(N_OBS*N_FEATS, tilde_x_d, p);
  	//cudaMemcpy(X_h, tilde_x, sizeof(int) * N_OBS * N_FEATS, cudaMemcpyDeviceToHost);
  	//copy_x_to_host(tilde_x, X_h);

  	cudaDeviceSynchronize();
	//
  	int n_batches = ceil(N_OBS / BATCHSIZE); 
  	//printf("Batches %d %d",BATCHSIZE,n_batches);
  	//2. encode to get hidden values y
  	dim3 dimGrid2(1);
  	dim3 dimBlock2(BATCHSIZE);
  	//dim3 dimBlock2(1);
  	//printf("\n N : %d",model_h->N);
  	int ib=0;
  	//n_batches = 1;
  	for (ib=0; ib<n_batches;ib++) {
    	    //2. encode to get hidden values y
   	    dA_get_hidden_values_kernel<<<dimGrid2,dimBlock2>>>(N_HIDDEN,N_FEATS,dW_flat,dhbias,tilde_x_d,y_d,ib,BATCHSIZE);
	    //3.decode by reconstrution to get z
	    dA_get_reconstructed_input_kernel<<<dimGrid2,dimBlock2>>>(N_HIDDEN,N_FEATS,dW_flat,dvbias,z_d,y_d,ib,BATCHSIZE);
	    //4. Update error in reconstruction - visible error for every minibatch by atomic add kernel
	    dA_L_vbias_kernel<<<dimGrid2,dimBlock2>>>(model_h->N,dL_vbias,dvbias,N_FEATS,X_d,z_d,ib,BATCHSIZE,learning_rate);
  	    //5.Update error in hidden units outputs, we would use it to update weights
	    dA_L_hbias_kernel<<<dimGrid2,dimBlock2>>>(model_h->N,dL_vbias,dL_hbias,dhbias,N_HIDDEN,N_FEATS,y_d,dW_flat,ib,BATCHSIZE,learning_rate);
	    //6. Weights updates for minibatch
	    dA_W_kernel<<<dimGrid2,dimBlock2>>>(model_h->N,dL_vbias,dL_hbias,model_h->n_hidden,model_h->n_visible,
						y_d,dW_flat,tilde_x_d,ib,BATCHSIZE,learning_rate);
 	}
  	//
	cuda_ret = cudaDeviceSynchronize();
	if (cuda_ret != cudaSuccess)
	    printf("Error in kernel");
	//
  	cudaMemcpy(tilde_x_h, tilde_x_d,sizeof(double) * N_OBS * N_FEATS, cudaMemcpyDeviceToHost);
	cudaMemcpy(y_h, y_d,sizeof(double) * 1*N_HIDDEN, cudaMemcpyDeviceToHost);
	cudaMemcpy(z_h, z_d,sizeof(double) * 1*N_FEATS, cudaMemcpyDeviceToHost);
	cudaMemcpy(L_vbias, dL_vbias,sizeof(double) * 1*N_FEATS, cudaMemcpyDeviceToHost);
	cudaMemcpy(L_hbias, dL_hbias,sizeof(double) * 1*N_HIDDEN, cudaMemcpyDeviceToHost);
	cudaMemcpy(model_h->W_flat, dW_flat,sizeof(double) * N_HIDDEN * N_FEATS, cudaMemcpyDeviceToHost);
	cudaMemcpy(model_h->vbias, dvbias,sizeof(double) * N_FEATS, cudaMemcpyDeviceToHost);
	cudaMemcpy(model_h->hbias, dhbias,sizeof(double) * N_HIDDEN, cudaMemcpyDeviceToHost);
	///cudaDeviceSynchronize();
	//
	/*
	printf("ibb is: %d\n",ib);
	for(int i=0;i<N_OBS;i++) {
	    printf("\ntile_x_h : "); for(int j=0;j<5;j++){ printf(" %f ",tilde_x_h[i*N_OBS+j]); }
	}
	printf("\ny_h : "); for(int j=0;j<5;j++){ printf(" %f ",y_h[j]); }
	//for(int i=0;i<N_OBS;i++) {
	printf("\nz_h: "); for(int j=0;j<5;j++){ printf(" %f ",z_h[j]); }
	//}
	printf("\nh vbias: "); for(int j=0;j<5;j++){ printf(" %f ",model_h->vbias[j]); }
	printf("\nh hbias: "); for(int j=0;j<5;j++){ printf(" %f ",model_h->hbias[j]); }
	printf("\nh Weights: ");for(int j=0;j<5;j++){ printf(" %f ",model_h->W_flat[j]); }
	*/
	//
	//model_h->W_flat = hW_flat;
	// read results from device
  }
  // free up memory
  cudaFree(X_d); cudaFree(tilde_x_d); cudaFree(dW_flat);
  cudaFree(dhbias); cudaFree(dvbias); cudaFree(dL_vbias);
  cudaFree(dL_hbias); cudaFree(y_d); cudaFree(z_d); free(y_h);
  X_d = NULL;tilde_x_d = NULL; dW_flat = NULL; dhbias = NULL;
  dvbias = NULL; dL_vbias = NULL;dL_hbias = NULL;
  y_d = NULL; z_d = NULL; y_h = NULL;
  //

  
}


void test_dbn(void) {
  srand(0);
  int i, j, epoch;

  double learning_rate = 0.1;
  double corruption_level = 0.3;
  int training_epochs = 100;

  int train_N = 10;
  int test_N = 2;
  int n_visible = 20;
  int n_hidden = 5;

  // training data
  int train_X[10][20] = {
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0}
  };

  // construct dA
  dA da_gold, da_h;
  dA__construct(&da_gold, train_N, n_visible, n_hidden, NULL, NULL, NULL);
  dA__construct(&da_h, train_N, n_visible, n_hidden, NULL, NULL, NULL);

  // train using gold standard
  for(epoch=0; epoch<training_epochs; epoch++) {
    for(i=0; i<train_N; i++) {
      dA_train_gold(&da_gold, train_X[i], learning_rate, corruption_level);
    }
  }
  
  //* Start of GPU Code
  // train using kernel
  //dA_train_on_device(&da_h, train_X, learning_rate, corruption_level);
  dA_train_on_device1(&da_h, train_X, learning_rate, corruption_level,training_epochs);
  printf("\nCPU :"); for(int j=0;j<5;j++) {printf("%f ", da_gold.W_flat[j]);};
  printf("\nGPU :"); for(int j=0;j<5;j++) {printf("%f ", da_h.W_flat[j]);};
  //* End of GPU Coode
  // test data
  int test_X[2][20] = {
    {1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0}
  };
  double reconstructed_X[2][20];

  printf("\n : test now: \n");
  // test
  for(i=0; i<test_N; i++) {
    dA_reconstruct(&da_gold, test_X[i], reconstructed_X[i]);
    for(j=0; j<n_visible; j++) {
      printf("%.5f ", reconstructed_X[i][j]);
    }
    printf("\n");
  }

  // test GPU
  for(i=0; i<test_N; i++) {
    dA_reconstruct(&da_h, test_X[i], reconstructed_X[i]);
    for(j=0; j<n_visible; j++) {
      printf("%.5f ", reconstructed_X[i][j]);
    }
    printf("\n");
  }


  // destruct dA
  dA__destruct(&da_gold);
  dA__destruct(&da_h);
}


int main(int argc, char** argv) {
  test_dbn();
  return 0;
}

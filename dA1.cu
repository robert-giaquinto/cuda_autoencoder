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
#define BATCHSIZE 5
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


double * allocate_device_y(int obs, int n_hidden) {
  double *y_d = NULL;
  int size = obs * n_hidden * sizeof(double);
  cudaMalloc((void**)&y_d, size);
  return y_d;
}


double * allocate_device_z() {
  double *z_d = NULL;
  int size = N_OBS * N_FEATS* sizeof(double);
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

void copy_x_to_device(int *x_d, int *x_h) {
  int size = N_OBS * N_FEATS * sizeof(int);
  cudaMemcpy(x_d, x_h, size, cudaMemcpyHostToDevice);
}


void copy_x_to_host(int *x_d, int *x_h) {
  int size = N_OBS * N_FEATS * sizeof(int);
  cudaMemcpy(x_h, x_d, size, cudaMemcpyDeviceToHost);
}

  

void dA_train_on_device1(dA *model_h, int train_X[N_OBS][N_FEATS], double learning_rate, double corruption_level) {
  // call kernel function from here
  // assign one observation to each block, each thread parallelizes a feature
  
  // flatten input array
  int *X_h = flatten_array(train_X);

  // allocate space on device
  int *X_d = allocate_device_x();
  int *tilde_x_d = allocate_device_x();
  double *y_h = (double*)malloc(sizeof(double)* N_HIDDEN * N_OBS);
  double *y_d = allocate_device_y(N_OBS, N_HIDDEN); 
  double *hW_flat = (double*)malloc(sizeof(double)* N_HIDDEN * N_FEATS);
  double *dW_flat = allocate_device_dW();
  double *dhbias = allocate_device_dhbias();
  double *dvbias = allocate_device_dvbias(); 
  double *z_d = allocate_device_z();
  double *z_h = (double*)malloc(sizeof(double)* N_FEATS * N_OBS);
  
  // copy data over to device
  copy_x_to_device(X_d, X_h);
  copy_x_to_device(tilde_x_d, X_h);
  //  copy_ae_to_device(model_d, model_h);

  printf("in device processing..."); 
  double p = 1 - corruption_level;  

  // set up corrupted input for all together
  printf("X_h %d %d",X_h[1],X_h[2]);
  dim3 dimGrid1(1);
  dim3 dimBlock1(N_OBS*N_FEATS);
  dA_get_corrupted_input_kernel<<<dimGrid1, dimBlock1>>>(N_OBS*N_FEATS, tilde_x_d, p);
  //cudaMemcpy(X_h, tilde_x, sizeof(int) * N_OBS * N_FEATS, cudaMemcpyDeviceToHost);
  //copy_x_to_host(tilde_x, X_h);

  cudaDeviceSynchronize();
  //

  int n_batches = ceil(N_OBS / BATCHSIZE); 
  //printf("tilde_x %d %d",X_h[1],X_h[2]);
  //2. encode to get hidden values y
  dim3 dimGrid2(1);
  dim3 dimBlock2(BATCHSIZE);
  int ib=0;
  for (ib=0; ib<n_batches;ib++) 
     //2. encode to get hidden values y
     dA_get_hidden_values_kernel<<<dimGrid2,dimBlock2>>>(N_HIDDEN,N_FEATS,dW_flat,dhbias, tilde_x_d,y_d,ib,BATCHSIZE);
     //3.decode by reconstrution to get z
     dA_get_reconstructed_input_kernel<<<dimGrid2,dimBlock2>>>(N_HIDDEN,N_FEATS,dW_flat,dvbias,z_d,y_d,ib,BATCHSIZE);

  cudaMemcpy(y_h, y_d,sizeof(double) * N_HIDDEN * N_OBS, cudaMemcpyDeviceToHost);
  cudaMemcpy(z_h, z_d,sizeof(double) * N_FEATS * N_OBS, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
 
  printf("ibb is %d",ib);
  printf("\ny : %f %f %f\n",y_h[0],y_h[3],y_h[7]);
  printf("\nz : %f %f %f\n",z_h[0],z_h[3],z_h[7]);


  /*
  // define kernel dimensions
  int batch_size = 1;
  dim3 dim_grid(batch_size, 1, 1);
  dim3 dim_block(N_FEATS, 1, 1);
  dA_train_kernel<<<dim_grid, dim_block>>>(model_d, X_d, learning_rate, corruption_level);
  cudaDeviceSynchronize();
  */
  // read results from device
  // free up memory
  free(X_h);
  cudaFree(X_d);
  cudaFree(tilde_x_d);
  free(hW_flat);
  cudaFree(dW_flat);
  cudaFree(dhbias);
  cudaFree(dvbias);
  
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
  
  // train using kernel
  printf("\nBefore: %f, %f", da_h.W_flat[0], da_h.hbias[0]);
  //dA_train_on_device(&da_h, train_X, learning_rate, corruption_level);
  dA_train_on_device1(&da_h, train_X, learning_rate, corruption_level);
  printf("\nAfter: %f, %f\n", da_h.W_flat[0], da_h.hbias[0]);

  // test data
  int test_X[2][20] = {
    {1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0}
  };
  double reconstructed_X[2][20];


  // test
  for(i=0; i<test_N; i++) {
    dA_reconstruct(&da_gold, test_X[i], reconstructed_X[i]);
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

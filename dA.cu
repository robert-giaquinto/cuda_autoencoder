// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// includes, project
#include <cutil.h>

// includes, kernel
#include <dA_kernel.cu>


#define N_FEATS 20
#define N_OBS 10


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

dA init_device_ae(const dA model_h) {
  // allocate space
  dA model_d;
  model_d.N = model_h.N;
  model_d.n_visible = model_h.n_visible;
  model_d.n_hidden = model_h.n_hidden;
  model_d.hbias = NULL;
  model_d.vbias = NULL;
  model_d.W = NULL;
  model_d.W_flat = NULL;

  int W_size = sizeof(double) * model_h.n_hidden * model_h.n_visible;
  int hbias_size = sizeof(double) * model_h.n_hidden;
  int vbias_size = sizeof(double) * model_h.n_visible;

  cudaMalloc((void**)&model_d.W_flat, W_size);
  cudaMalloc((void**)&model_d.hbias, hbias_size);
  cudaMalloc((void**)&model_d.vbias, vbias_size);

  // flatten w
  //double *flat_w = flatten_w(model_h.W, model_h.n_visible, model_h.n_hidden);

  // copy over data
  cudaMemcpy(model_d.W_flat, model_h.W_flat, W_size, cudaMemcpyHostToDevice);
  cudaMemcpy(model_d.hbias, model_h.hbias, hbias_size, cudaMemcpyHostToDevice);
  cudaMemcpy(model_d.vbias, model_h.vbias, vbias_size, cudaMemcpyHostToDevice);
  
  //free(flat_w);
  return model_d;
}

void copy_x_to_device(int *x_d, int *x_h) {
  int size = N_OBS * N_FEATS * sizeof(int);
  cudaMemcpy(x_d, x_h, size, cudaMemcpyHostToDevice);
}


void copy_ae_from_device(dA *model_h, const dA model_d) {
  int W_size = sizeof(double) * model_h->n_hidden * model_h->n_visible;
  int hbias_size = sizeof(double) * model_h->n_hidden;
  int vbias_size = sizeof(double) * model_h->n_visible;

  cudaMemcpy(model_h->W_flat, model_d.W_flat, W_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(model_h->hbias, model_d.hbias, hbias_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(model_h->vbias, model_d.vbias, vbias_size, cudaMemcpyDeviceToHost);
}

void free_device(dA *model) {
  cudaFree(model->W_flat);
  cudaFree(model->hbias);
  cudaFree(model->vbias);
  model->W_flat = NULL;
  model->hbias = NULL;
  model->vbias = NULL;
}
  

void dA_train_on_device(dA *model_h, int train_X[][N_FEATS], double learning_rate, double corruption_level) {
  // call kernel function from here
  // assign one observation to each block, each thread parallelizes a feature
  
  // flatten input array
  int *X_h = flatten_array(train_X);

  // allocate space on device
  int *X_d = allocate_device_x();
  dA model_d = init_device_ae(*model_h);

  // copy data over to device
  copy_x_to_device(X_d, X_h);
  //  copy_ae_to_device(model_d, model_h);

  // define kernel dimensions
  int batch_size = 1;
  dim3 dim_grid(batch_size, 1, 1);
  dim3 dim_block(N_FEATS, 1, 1);
  dA_train_kernel<<<dim_grid, dim_block>>>(model_d, X_d, learning_rate, corruption_level);
  cudaDeviceSynchronize();
  
  // read results from device
  copy_ae_from_device(model_h, model_d);

  // free up memory
  free(X_h);
  free_device(&model_d);
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
  dA_train_on_device(&da_h, train_X, learning_rate, corruption_level);
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

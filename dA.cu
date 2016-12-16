// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <fstream>

// includes, project
#include <cutil.h>

// includes, kernel
#include <dA_kernel.cu>


// declarations for CPU train functions
extern "C"
void dA_train_gold(dA*, float*, double, double);
void dA_get_hidden_values(dA*, float*, double*, bool);
void dA_get_reconstructed_input(dA*, double*, double*, bool);


// functions defined in this file are for intializing the autoencoder
double uniform(double min, double max);
void dA__construct(dA *model, int N, int n_visible, int n_hidden, double **W, double *hbias, double *vbais);
void dA__destruct(dA *model);
void dA_reconstruct(dA *model, float *x, double *z, bool flat);
void test_dbn();

//void dA_train_on_device(dA*, float[][N_FEATS], double, double, int);
void dA_train_on_device(dA*, float*, double, double, int);
float* flatten_array(float arr[][N_FEATS], int n_rows);
float* allocate_device_x();
dA init_device_ae(const dA model_h);
void copy_x_to_device(float *x_d, float *x_h);
void copy_ae_from_device(dA *model_h, const dA model_d);
void free_device(dA *model);
void print_reconstruction(dA, float[N_TEST][N_FEATS], bool);
int read_file(float *X, char *filename);


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

  model->W_flat = (double*)malloc(sizeof(double)*n_hidden*n_visible);
  if(W == NULL) {
    model->W = (double **)malloc(sizeof(double*) * n_hidden);
    model->W[0] = (double *)malloc(sizeof(double) * n_visible * n_hidden);
    for(i=0; i<n_hidden; i++) {
      model->W[i] = model->W[0] + i * n_visible;
    }
    for(i=0; i<n_hidden; i++) {
      for(j=0; j<n_visible; j++) {
        double u = uniform(-a, a);
        model->W_flat[i*n_visible + j] = u;
        model->W[i][j] = u;
      }
    }
  } else {
    model->W = W;
    for (int i=0; i < n_hidden; ++i) {
      for (int j=0; j < n_visible; ++j) {
        model->W_flat[i*N_FEATS + j] = W[i][j];
      }
    }
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


void dA_reconstruct(dA* model, float *x, double *z, bool flat) {
  double *y = (double *)malloc(sizeof(double) * model->n_hidden);

  dA_get_hidden_values(model, x, y, flat);
  dA_get_reconstructed_input(model, y, z, flat);

  free(y);
}



int read_file(float *arr, const char* filename, int n_rows) {
  // initialize the data, is this still needed?
  for (int i=0; i < n_rows * N_FEATS; ++i) {
    arr[i] = 0.0f;
  }

  unsigned int data_size = n_rows * N_FEATS;

  // intermediate storage for the data read
  std::vector<float>  data_read;
  
  // open file for reading
  std::fstream fh( filename, std::fstream::in);
  // read data elements 
  float token;
  for (int i=0; i < data_size; ++i) {
    fh >> token;
    data_read.push_back(token);
    if (fh.good() == 0) {
      std::cerr << "FILE HANDLE NOT GOOD" << std::endl;
      return 1;
    }
  }
    
  // the last element is read twice
  fh.close();
  
  // copy data
  memcpy(arr, &data_read.front(), sizeof(float) * data_read.size());  
  return 0;
}



float * allocate_device_x() {
  float *x_d = NULL;
  int size = N_OBS * N_FEATS * sizeof(float);
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

  // copy over data
  cudaMemcpy(model_d.W_flat, model_h.W_flat, W_size, cudaMemcpyHostToDevice);
  cudaMemcpy(model_d.hbias, model_h.hbias, hbias_size, cudaMemcpyHostToDevice);
  cudaMemcpy(model_d.vbias, model_h.vbias, vbias_size, cudaMemcpyHostToDevice);
  
  //free(flat_w);
  return model_d;
}

void copy_x_to_device(float *x_d, float *x_h) {
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
  

void dA_train_on_device(dA *model_h, float *train_X, double learning_rate, double corruption_level, int epochs) {
  // call kernel function from here
  // assign one observation to each block, each thread parallelizes a featires
  float total_time=0.0f;
  cudaError_t cuda_ret;

  // allocate space on device
  float *X_d = allocate_device_x();
  dA model_d = init_device_ae(*model_h);

  // copy data over to device
  copy_x_to_device(X_d, train_X);

  // define kernel dimensions
  dim3 dim_grid(BATCH_SIZE, 1, 1);
  dim3 dim_block(N_FEATS, 1, 1);
  // how many mini-batch updates to run?
  int n_updates = ceil((float)N_OBS / (float)BATCH_SIZE);
  // initialize a random state for each thread;
  curandState *d_state;
  cudaMalloc(&d_state, N_FEATS * BATCH_SIZE);

  unsigned int timer1; cutCreateTimer(&timer1); cutStartTimer(timer1); 
  // run kernel!
  for (int epoch=0; epoch < epochs; ++epoch){
    for (int batch=0; batch < n_updates; ++batch) {
      dA_train_kernel<<<dim_grid, dim_block>>>(model_d, X_d, learning_rate, corruption_level, batch, d_state);
      cuda_ret = cudaDeviceSynchronize();
      if (cuda_ret != cudaSuccess) printf("error in kernel");
    }
  }
  cutStopTimer(timer1); total_time += cutGetTimerValue(timer1); cutDeleteTimer(timer1);
  printf("\nKernel call: %f\n", total_time);  

  // read results from device
  copy_ae_from_device(model_h, model_d);

  // free up memory
  //free(X_h);
  free_device(&model_d);
  cudaFree(d_state);
}


void print_reconstruction(dA *model, float *X, bool flat) {
  double reconstructed_X[N_TEST][N_FEATS];
  for(int i=0; i < N_TEST; i++) {
    dA_reconstruct(model, &X[i*N_FEATS], reconstructed_X[i], flat);
    for(int j=0; j < N_FEATS; j++) {
      printf("%.5f ", reconstructed_X[i][j]);
    }
    printf("\n");
  }
}

void print_W(dA *model, bool flat) {
  printf("\nflat = %d:\n", flat);
  for (int i=0; i < N_HIDDEN; ++i) {
    for (int j=0; j < N_FEATS; ++j) {
      if (flat) {
        printf("%f ", model->W_flat[i*N_FEATS + j]);
      } else {
        printf("%f ", model->W[i][j]);
      }
    }
    printf("\n");
  }
}


void test_dbn(void) {
  srand(0);
  float device_time;
  float host_time;

  double learning_rate = 0.1;
  double corruption_level = 0.3;
  int training_epochs = 100;

  // training data
  float *train_X = (float*)malloc(sizeof(float) * N_OBS * N_FEATS);
  int error_train = 0;
  error_train = read_file(train_X, "/home/class/smit7982/app/C/src/ee5351/simple_train.txt", N_OBS);
  if (error_train) {
    printf("Error reading training input file");
  }

  // construct dA
  dA da_gold, da_h;
  dA__construct(&da_gold, N_OBS, N_FEATS, N_HIDDEN, NULL, NULL, NULL);
  dA__construct(&da_h, N_OBS, N_FEATS, N_HIDDEN, NULL, NULL, NULL);


  printf("\nStarting gold training...");
  unsigned int cputimer;
  cutCreateTimer(&cputimer);
  cutStartTimer(cputimer);
  // train using gold standard
  for(int epoch=0; epoch<training_epochs; epoch++) {
    for(int i=0; i < N_OBS; i++) {
      dA_train_gold(&da_gold, &train_X[i*N_FEATS], learning_rate, corruption_level);
    }
  }
  cutStopTimer(cputimer);
  host_time = cutGetTimerValue(cputimer);
  cutDeleteTimer(cputimer);

  // train using kernel
  printf("Starting device training...");
  unsigned int gputimer;
  cutCreateTimer(&gputimer);
  cutStartTimer(gputimer);
  dA_train_on_device(&da_h, train_X, learning_rate, corruption_level, training_epochs);
  cutStopTimer(gputimer);
  device_time = cutGetTimerValue(gputimer);
  cutDeleteTimer(gputimer);


  // test data
  float *test_X = (float*)malloc(sizeof(float) * N_TEST * N_FEATS);
  int error_test = 0;
  error_test = read_file(test_X, "/home/class/smit7982/app/C/src/ee5351/simple_test.txt", N_TEST);
  if (error_test) {
    printf("Error reading test input file");
  }

  // test
  printf("\nGold test output:\n");
  print_reconstruction(&da_gold, test_X, 0);
  printf("\nKernel test output:\n");
  print_reconstruction(&da_h, test_X, 1);

  printf("Host time          : %f\n", host_time);
  printf("Device time        : %f\n", device_time);
  printf("Speedup host/device: %fX\n", host_time/device_time);

  // destruct dA
  dA__destruct(&da_gold);
  dA__destruct(&da_h);
}


int main(int argc, char** argv) {
  test_dbn();
  return 0;
}

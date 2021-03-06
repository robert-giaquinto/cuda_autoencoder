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
//#include <dA.h>


// declarations for CPU train functions
extern "C"
void dA_train_gold(dA*, float*, double, double);
void dA_get_hidden_values(dA*, float*, double*);
void dA_get_reconstructed_input(dA*, double*, double*);
// functions defined in this file are for intializing the autoencoder
double uniform(double min, double max);
void dA__construct(dA *model, int N, int n_visible, int n_hidden, double **W, double *hbias, double *vbais);
void dA__destruct(dA *model);
void dA_reconstruct(dA *model, float *x, double *z);
void test_dbn();
void dA_train_on_device1(dA*, float*, double, double);

// MNIST and simple file load
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

//* end of file load function
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
  free(model->W);
  free(model->W_flat);
  free(model->hbias);
  free(model->vbias);
}


void dA_reconstruct(dA* model, float *x, double *z) {
  double *y = (double *)malloc(sizeof(double) * model->n_hidden);

  dA_get_hidden_values(model, x, y);
  dA_get_reconstructed_input(model, y, z);

  free(y);
}


float* flatten_array(float arr[N_OBS][N_FEATS]) {
  float *flat = (float *)malloc(sizeof(float) * N_OBS * N_FEATS);
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

float * allocate_device_x(int n) {
  float *x_d = NULL;
  int size = n * sizeof(float);
  cudaMalloc((void**)&x_d, size);
  return x_d;
}


float * allocate_device_tile_x(int m) {
  float *tile_x_d = NULL;
  int size = m * sizeof(float);
  cudaMalloc((void**)&tile_x_d, size);
  return tile_x_d;
}


double * allocate_device_z(int m) {
  double *z_d = NULL;
  int size = m * sizeof(double);
  cudaMalloc((void**)&z_d, size);
  return z_d;
}

double * allocate_device_y(int n) {
  double *y_d = NULL;
  int size = n * sizeof(double);
  cudaMalloc((void**)&y_d, size);
  return y_d;
}

double*  allocate_device_dW(int n) {
  double *dW_flat;
  int dW_size = sizeof(double) * n;
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

void copy_x_to_device(float *x_d, float *x_h) {
  int size = N_OBS * N_FEATS * sizeof(float);
  cudaMemcpy(x_d, x_h, size, cudaMemcpyHostToDevice);
}

void copy_x_to_host(float *x_d, float *x_h) {
  int size = N_OBS * N_FEATS * sizeof(float);
  cudaMemcpy(x_h, x_d, size, cudaMemcpyDeviceToHost);
}

void dA_train_on_device1(dA *model_h, float *train_X, double lr, double corruption_level,int training_epochs) {
  //
  cudaError_t cuda_ret;
  int epoch;
  int offsetXval;
  //
  //
  double p = 1 - corruption_level;  
  float *X_d = allocate_device_x(N_OBS*N_FEATS);
  // allocate space on for tilde_x on device - we do not need it on host. We will allocate
  // per batchsize space for it here
  float *tilde_x_d = allocate_device_tile_x(BATCHSIZE*N_FEATS);
  double *yb_d = allocate_device_y(N_HIDDEN*BATCHSIZE); 
  // Weight matrices are allocated
  double *hW_flat = (double*)malloc(sizeof(double)* N_HIDDEN * N_FEATS);
  double *dW_flat = allocate_device_dW(N_HIDDEN * N_FEATS);
  //hidden bias and visible bias are allocated on device
  double *dhbias = allocate_device_dhbias();
  double *dvbias = allocate_device_dvbias(); 
  //dA_get_reconstructed_input
  double *z_d = allocate_device_z(N_FEATS*BATCHSIZE);
  //Allocate intermediate L_vbias and L_hbias
  double *dL_vbias = allocate_device_dL_vbias(BATCHSIZE,N_FEATS);
  double *dL_hbias = allocate_device_dL_hbias(BATCHSIZE, N_HIDDEN);
  //
  //1. kernel configurations for dA_get_corrupted_input_kernel
  dim3 dimGrid31(BATCHSIZE);
  dim3 dimBlock31(N_FEATS);
  //
  //2. kernel config for hidden values y
  int n32Threads = BLOCKSIZE;
  int n32Blocks = N_HIDDEN / n32Threads;
  int m32Blocks = BATCHSIZE / n32Threads;
  if (N_HIDDEN % n32Threads) n32Blocks++;
  if (BATCHSIZE % n32Threads) m32Blocks++;
  dim3 dimGrid32(m32Blocks, n32Blocks);
  dim3 dimBlock32(n32Threads,n32Threads);
  //
  //3. kernel config for dA_get_reconstructed_input_kernel
  int n33Threads = BLOCKSIZE;
  int n33Blocks = N_FEATS / n33Threads;
  int m33Blocks = BATCHSIZE / n33Threads;
  if (N_FEATS % n33Threads) n33Blocks++;
  if (BATCHSIZE % n33Threads) m33Blocks++;
  dim3 dimGrid33(m33Blocks, n33Blocks);
  dim3 dimBlock33(n33Threads,n33Threads);
  //
  //4. kernel config for dA_L_vbias_kernel
  dim3 dimGrid34(BATCHSIZE);
  dim3 dimBlock34(N_FEATS);
  //
  //5. kernel config for dA_L_hbias_kernel
  int n35Threads = BLOCKSIZE;
  int n35Blocks = N_HIDDEN / n35Threads;
  int m35Blocks = BATCHSIZE / n35Threads;
  if (N_HIDDEN % n35Threads) n35Blocks++;
  if (BATCHSIZE % n35Threads) m35Blocks++;
  dim3 dimGrid35(m35Blocks, n35Blocks);
  dim3 dimBlock35(n35Threads,n35Threads);
  //
  //6. kernel config for dA_W_kernel
  dim3 dimGrid36(N_HIDDEN);
  dim3 dimBlock36(N_FEATS);
  //
  copy_x_to_device(X_d, train_X);
  // copy over data so that initial weights and biases are same
  cudaMemcpy(dW_flat, model_h->W_flat, sizeof(double)*N_HIDDEN*N_FEATS, cudaMemcpyHostToDevice);
  cudaMemcpy(dhbias, model_h->hbias, sizeof(double)*N_HIDDEN, cudaMemcpyHostToDevice);
  cudaMemcpy(dvbias, model_h->vbias, sizeof(double)*N_FEATS, cudaMemcpyHostToDevice);
  //
  //Start of epochs ---------------------------------------------------------------------------
  for(epoch=0; epoch<training_epochs; epoch++) {
  	int n_batches = ceil(N_OBS / BATCHSIZE); 
  	//2. encode to get hidden values y
  	for (int ib=0; ib<n_batches;ib++) {
  		//start1. dA_get_corrupted_input_kernel to get tilde_x
		offsetXval = ib*BATCHSIZE*N_FEATS; // this is offset for each batch in input x
     		dA_get_corrupted_input_kernel<<<dimGrid31, dimBlock31>>>(BATCHSIZE*N_FEATS, X_d, tilde_x_d, p, offsetXval);
  		cuda_ret = cudaDeviceSynchronize();
 		if (cuda_ret != cudaSuccess) 	printf("Error in kernel dA_get_corrupted_input_kernel");
		//end1
        	//start2. 
		//encode to get hidden values y
       		dA_get_hidden_values_kernel<<<dimGrid32,dimBlock32>>>(N_HIDDEN,N_FEATS,dW_flat,dhbias,tilde_x_d,yb_d,ib);
  		cuda_ret = cudaDeviceSynchronize();
 		if (cuda_ret != cudaSuccess) printf("Error in kernel dA_get_hidden_values_kernel");
 		//end2 
		//start3
    		//3.decode by reconstrution to get z: dA_get_reconstructed_input_kernel
    		dA_get_reconstructed_input_kernel<<<dimGrid33,dimBlock33>>>(N_HIDDEN,N_FEATS,dW_flat,dvbias,z_d,yb_d,ib,BATCHSIZE);
  		cuda_ret = cudaDeviceSynchronize();
 		if (cuda_ret != cudaSuccess) printf("Error in kernel dA_get_reconstructed_input_kernel");
  		//cudaMemcpy(z_h, z_d,sizeof(double) * N_FEATS * BATCHSIZE, cudaMemcpyDeviceToHost);
  		//printf("\nz_h: "); for(int j=0;j<N_FEATS*BATCHSIZE;j++){ printf(" %f ",z_h[j]); }
  		//cutStopTimer(timer33); time33 += cutGetTimerValue(timer33); cutDeleteTimer(timer33);
		//end3
		//start4
    		//4. Update error in reconstruction - visible error for every minibatch
  		//unsigned int timer34; cutCreateTimer(&timer34);	cutStartTimer(timer34);
    		dA_L_vbias_kernel<<<dimGrid34,dimBlock34>>>(model_h->N,dL_vbias,dvbias,N_FEATS,X_d,z_d,offsetXval,BATCHSIZE,lr);
  		cuda_ret = cudaDeviceSynchronize();
 		if (cuda_ret != cudaSuccess) printf("Error in kernel dA_L_vbias_kernel ");
		//end4
		//start5
      		//5.Update error in hidden units outputs, we would use it to update weights
    		dA_L_hbias_kernel<<<dimGrid35,dimBlock35>>>(model_h->N,dL_vbias,dL_hbias,dhbias,N_HIDDEN,N_FEATS,yb_d,dW_flat,ib,BATCHSIZE,lr);
  		cuda_ret = cudaDeviceSynchronize();
 		if (cuda_ret != cudaSuccess) printf("Error in kernel dA_L_hbias_kernel ");
		//end5
		//start6
    		//6. Weights updates for minibatch
    		dA_W_kernel<<<dimGrid36,dimBlock36>>>(model_h->N,dL_vbias,dL_hbias,model_h->n_hidden,model_h->n_visible,
							yb_d,dW_flat,tilde_x_d,ib,BATCHSIZE,lr);
  		cuda_ret = cudaDeviceSynchronize();
 		if (cuda_ret != cudaSuccess) printf("Error in kernel dA_W_kernel");
		//end6
 	}
  }
  //End of epochs ---------------------------------------------------------------------------
  //
  //Copy weights, vbias, hbias learned from the model back to cpu model_h for testing
  cudaMemcpy(model_h->vbias, dvbias,sizeof(double) * N_FEATS, cudaMemcpyDeviceToHost);
  cudaMemcpy(model_h->hbias, dhbias,sizeof(double) * N_HIDDEN, cudaMemcpyDeviceToHost);
  cudaMemcpy(model_h->W_flat, dW_flat,sizeof(double) * N_HIDDEN * N_FEATS, cudaMemcpyDeviceToHost);
  //
  //Testing needs W[i][j] format, so copying flattened W to W[i][j]
  //We can not directly copy to W and W is used to test, so we populate it using a loop
  for(int i=0; i<model_h->n_hidden; i++) {
      for(int j=0; j<model_h->n_visible; j++) {
       	model_h->W[i][j] = model_h->W_flat[i*model_h->n_visible + j];
      }	
  }
  // free up memory
  cudaFree(X_d); cudaFree(tilde_x_d); cudaFree(dW_flat);
  cudaFree(dhbias); cudaFree(dvbias); cudaFree(dL_vbias);
  cudaFree(dL_hbias); cudaFree(z_d); 
  X_d = NULL;tilde_x_d = NULL; dW_flat = NULL; dhbias = NULL;
  dvbias = NULL; dL_vbias = NULL;dL_hbias = NULL; z_d = NULL; 
  //
  free(hW_flat); hW_flat = NULL;
  //
}

void test_dbn(void) {
  srand(0);
  int i, epoch;
  float device_time;
  float host_time;

  double learning_rate = 0.1;
  double corruption_level = 0.3;
  int training_epochs = MAX_EPOCHS;

  int train_N = N_OBS;
  int n_visible = N_FEATS;
  int n_hidden = N_HIDDEN;

  // training data
  float *train_X = (float*)malloc(sizeof(float) * N_OBS * N_FEATS);
  int error_train = 0;
  //error_train = read_file(train_X, "/home/class/kumar250/ee5351proj/simple_train.txt", N_OBS);
  error_train = read_file(train_X, "/home/class/kumar250/ee5351proj/mnist_train.txt", N_OBS);
  if (error_train) {
    printf("Error reading training input file");
  }
  // construct dA
  dA da_gold, da_h;
  dA__construct(&da_gold, train_N, n_visible, n_hidden, NULL, NULL, NULL);
  dA__construct(&da_h, train_N, n_visible, n_hidden, NULL, NULL, NULL);
  for (int i=0;i<n_hidden;i++) {
    for (int j=0;j<n_visible;j++) {
	da_h.W[i][j] = da_gold.W[i][j];
	da_h.W_flat[i*n_visible+j] = da_gold.W_flat[i*n_visible+j];
    }
  }
  //
  printf("\nStarting gold training..");
  unsigned int cputimer;
  cutCreateTimer(&cputimer);
  cutStartTimer(cputimer);
  // train using gold standard
  for(epoch=0; epoch<training_epochs; epoch++) {
    for(i=0; i<N_OBS; i++) {
      dA_train_gold(&da_gold, &train_X[i*N_FEATS], learning_rate, corruption_level);
    }
  }
  //
  cutStopTimer(cputimer);
  host_time = cutGetTimerValue(cputimer);
  cutDeleteTimer(cputimer);
  printf("...Finished gold training..");
  //

  //* Start of GPU Kernel Call Code
  printf("\nStarting device training..");
  unsigned int gputimer;
  cutCreateTimer(&gputimer);
  cutStartTimer(gputimer);
  // train using kernel
  dA_train_on_device1(&da_h, train_X, learning_rate, corruption_level,training_epochs);
  //
  cutStopTimer(gputimer);
  device_time = cutGetTimerValue(gputimer);
  cutDeleteTimer(gputimer);
  printf("\nEnding device training..");
  //
  //* End of GPU Coode
  
  // test data
  float *test_X = (float*)malloc(sizeof(float) * N_TEST * N_FEATS);
  int error_test = 0;
  //error_test = read_file(test_X, "/home/class/kumar250/ee5351proj/simple_test.txt", N_TEST);
  error_test = read_file(test_X, "/home/class/kumar250/ee5351proj/mnist_test.txt", N_TEST);
  if (error_test) {
    printf("Error reading test input file");
  }
  //
  double reconstructed_X[N_TEST][N_FEATS];
  //
  //printf("\n : CPU test now: \n");
  //write cpu output file
  FILE *fp1;
  fp1 = fopen(CPU_TEST_FNAME, "w+");
  // test CPU using &da_gold object
  for(i=0; i<N_TEST; i++) {
    dA_reconstruct(&da_gold, &test_X[i*N_FEATS], reconstructed_X[i]);
    for (int j=0;j<N_FEATS;j++) {
	//printf("%.5f ", reconstructed_X[i][j]);
    	fprintf(fp1,"%lf\t",reconstructed_X[i][j]);
    }
    fprintf(fp1,"\n");
    //printf("\n");
  }
  fclose(fp1);
  //
  //printf("\n : GPU test now: \n");
  //write gpu output file
  FILE *fp2;
  fp2 = fopen(GPU_TEST_FNAME, "w+");
  // test GPU using &da_h object
  for(i=0; i<N_TEST; i++) {
    dA_reconstruct(&da_h, &test_X[i*N_FEATS], reconstructed_X[i]);
    for(int j=0; j<N_FEATS; j++) {
	//printf("%.5f ", reconstructed_X[i][j]);
    	fprintf(fp2,"%lf\t",reconstructed_X[i][j]);
    }
    fprintf(fp2,"\n");
    //printf("\n");
  }
  fclose(fp2);
  //
  // destruct dA
  dA__destruct(&da_gold);
  dA__destruct(&da_h);
  //
   printf("\n");
  printf("BATCHSIZE	     : %d\n",BATCHSIZE);		
  printf("BLOCKSIZE	     : %d\n",BLOCKSIZE);		
  printf("N_OBS		     : %d\n",N_OBS);		
  printf("\n");
  printf("Host time          : %f\n", host_time);
  printf("Device time        : %f\n", device_time);
  printf("Speedup host/device: %fX\n", host_time/device_time);
  printf("\n");
  printf("CPU test output file is : %s\n",CPU_TEST_FNAME);
  printf("GPU test output file is : %s\n",GPU_TEST_FNAME);
  printf("***testing over***\n");

}


int main(int argc, char** argv) {
  test_dbn();
  return 0;
}

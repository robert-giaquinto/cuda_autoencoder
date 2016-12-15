// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// includes, project
#include <cutil.h>

// includes, kernel
#include <dA_kernel1.cu>
//#include <dA.h>


// declarations for CPU train functions
extern "C"
void dA_train_gold(dA*, int*, double, double);
void dA_get_hidden_values(dA*, int*, double*);
void dA_get_reconstructed_input(dA*, double*, double*);
//**
void dA_get_corrupted_input(dA*, int*, int*, double);
void dA_get_hidden_values(dA*, int*, double*);
void dA_get_reconstructed_input(dA*, double*, double*);
int binomial(int n, double p);
double sigmoid(double x);
//**

// functions defined in this file are for intializing the autoencoder
double uniform(double min, double max);
void dA__construct(dA *model, int N, int n_visible, int n_hidden, double **W, double *hbias, double *vbais);
void dA__destruct(dA *model);
void dA_reconstruct(dA *model, int *x, double *z);
void test_dbn();
void dA_train_on_device(dA*, int[][N_FEATS], double, double);

 // Temporary testing

//* end of temporary testing
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
  //free(model->W[0]);
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


int * allocate_device_tile_x(int m) {
  int *tile_x_d = NULL;
  int size = m * sizeof(int);
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

void dA_get_corrupted_input_miniBatch(int n_visible, int *x, int *tilde_x, double p,int ib,int batchsize) {
  for (int j=0;j<batchsize;j++){
  	for(int i=0; i<n_visible; i++) {
   	 if(x[ib*batchsize+j*n_visible+i] == 0) {
   	   tilde_x[i*batchsize+j] = 0;
   	 } else {
  	    tilde_x[i*batchsize+j] = binomial(1, p);
   	 }
  	}
  }
}

void dA_train_on_device1(dA *model_h, int train_X[N_OBS][N_FEATS], double lr, double corruption_level,int training_epochs) {
  //
  cudaError_t cuda_ret;
  int epoch;
  int offsetXval;
  //int i,j;
  printf("\n ** in device processing ** \n"); 
  //
  float time1, time2, time3,time31,time32,time33,time34,time35,time36,time37,time38,time39,time322;
  time1=time2=time3=time31=time32=time33=time34=time35=time36=time37=time38=time39=time322=0.0;
  unsigned int timer1; cutCreateTimer(&timer1); cutStartTimer(timer1);  
  //
  double p = 1 - corruption_level;  
  // flatten input array
  int *X_h = flatten_array(train_X);
  // allocate space on device
  int *X_d = allocate_device_x();
  int *tilde_x_d = allocate_device_tile_x(BATCHSIZE*N_FEATS);
  int *tilde_x_h = (int*)malloc(sizeof(int)* BATCHSIZE * N_FEATS);
  //double *y_h = (double*)malloc(sizeof(double)* N_HIDDEN);
  double *yb_h = (double*)malloc(sizeof(double)* N_HIDDEN*BATCHSIZE);
  double *yb_d = allocate_device_y(N_HIDDEN*BATCHSIZE); 
  //double *y_d = allocate_device_y(N_HIDDEN); 
  double *hW_flat = (double*)malloc(sizeof(double)* N_HIDDEN * N_FEATS);
  double *dW_flat = allocate_device_dW();
  double *dhbias = allocate_device_dhbias();
  double *dvbias = allocate_device_dvbias(); 
  double *z_d = allocate_device_z(N_FEATS*BATCHSIZE);
  double *z_h = (double*)malloc(sizeof(double)*BATCHSIZE* N_FEATS);
   //
  double *L_vbias = (double *)malloc(sizeof(double) * BATCHSIZE * N_FEATS);
  double *dL_vbias = allocate_device_dL_vbias(BATCHSIZE,N_FEATS);
  double *L_hbias = (double *)malloc(sizeof(double) * BATCHSIZE * N_HIDDEN);
  double *dL_hbias = allocate_device_dL_hbias(BATCHSIZE, N_HIDDEN);
  //
  // initialize a random state for each thread;
  //curandState *d_state;
  //cudaMalloc(&d_state, N_FEATS * BATCHSIZE);
  //
  cutStopTimer(timer1); time1 = cutGetTimerValue(timer1); cutDeleteTimer(timer1);
  //
  unsigned int timer2;cutCreateTimer(&timer2); cutStartTimer(timer2);
  // copy data over to device
  copy_x_to_device(X_d, X_h);
  //copy_x_to_device(tilde_x_d, X_h);
  //dA model_d = init_device_ae(*model_h);
  // copy over data
  cudaMemcpy(dW_flat, model_h->W_flat, sizeof(double)*N_HIDDEN*N_FEATS, cudaMemcpyHostToDevice);
  cudaMemcpy(dhbias, model_h->hbias, sizeof(double)*N_HIDDEN, cudaMemcpyHostToDevice);
  cudaMemcpy(dvbias, model_h->vbias, sizeof(double)*N_FEATS, cudaMemcpyHostToDevice);
  //
  cutStopTimer(timer2); time2 = cutGetTimerValue(timer2); cutDeleteTimer(timer2);
  //
  //printf("X_h %d %d",X_h[1],X_h[2]);
  //
  //unsigned int timer3; cutCreateTimer(&timer3); cutStartTimer(timer3);
  //
  for(epoch=0; epoch<training_epochs; epoch++) {

  	int n_batches = ceil(N_OBS / BATCHSIZE); 

  	//2. encode to get hidden values y
  	dim3 dimGrid2(1);
  	dim3 dimBlock2(BATCHSIZE);
  	//dim3 dimBlock2(1);
  	//printf("\n N : %d",model_h->N);
  	int ib=0;
  	//n_batches = 1;
  	for (ib=0; ib<n_batches;ib++) {
		//  copy_ae_to_device(model_d, model_h);
  		unsigned int timer31; cutCreateTimer(&timer31);	cutStartTimer(timer31);
  		//1. set up corrupted input for all together
  		//printf("Batches %d %d",BATCHSIZE,n_batches);
		//int n31Threads  = 400; // 20 * n_feats say
		//int n31Blocks = ceil (model_h->n_visible*BATCHSIZE / n31Threads);
		dim3 dimGrid31(BATCHSIZE);
     		dim3 dimBlock31(N_FEATS);
		offsetXval = ib*BATCHSIZE*N_FEATS;
     		//dA_get_corrupted_input_kernel<<<dimGrid31, dimBlock31>>>(BATCHSIZE*N_FEATS, X_d, tilde_x_d, p, offsetXval,d_state);
     		dA_get_corrupted_input_kernel<<<dimGrid31, dimBlock31>>>(BATCHSIZE*N_FEATS, X_d, tilde_x_d, p, offsetXval);
  		cuda_ret = cudaDeviceSynchronize();
 		if (cuda_ret != cudaSuccess) 	printf("Error in kernel");
		
		//dA_get_corrupted_input(model_h, X_h, tilde_x_h, p);
  		//cudaMemcpy(tilde_x_h, tilde_x_d, sizeof(int) * BATCHSIZE * N_FEATS, cudaMemcpyDeviceToHost);
		//dA_get_corrupted_input_miniBatch(model_h->n_visible, X_h, tilde_x_h, p,ib,BATCHSIZE);
  		//cudaMemcpy(tilde_x_d, tilde_x_h, sizeof(int) * BATCHSIZE * model_h->n_visible, cudaMemcpyHostToDevice);
  		// printf("\ntilde_x_h : "); for(int j=0;j<BATCHSIZE*model_h->n_visible;j++){ printf(" %d ",tilde_x_h[j]); }
		
  		//copy_x_to_host(tilde_x, X_h);
  		cutStopTimer(timer31); time31 += cutGetTimerValue(timer31); cutDeleteTimer(timer31);
  		//cudaDeviceSynchronize();
		//
        	//2. encode to get hidden values y
  		unsigned int timer32; cutCreateTimer(&timer32);	cutStartTimer(timer32);
		//
		int n32Threads = TILE_WIDTH;
		int n32Blocks = N_HIDDEN / n32Threads;
		int m32Blocks = BATCHSIZE / n32Threads;
		if (N_HIDDEN % n32Threads) n32Blocks++;
		if (BATCHSIZE % n32Threads) m32Blocks++;
		dim3 dimGrid32(m32Blocks, n32Blocks);
 		//dim3 dimGrid32(n32Blocks, m32Blocks);
		dim3 dimBlock32(n32Threads,n32Threads);
       		//dA_get_hidden_values_kernel<<<dimGrid32,dimBlock32>>>(N_HIDDEN,N_FEATS,dW_flat,dhbias,tilde_x_d,y_d,ib);
       		//dA_get_hidden_values_kernel1<<<dimGrid32,dimBlock32>>>(N_HIDDEN,N_FEATS,dW_flat,dhbias,tilde_x_d,y_d,ib);
       		dA_get_hidden_values_kernel<<<dimGrid32,dimBlock32>>>(N_HIDDEN,N_FEATS,dW_flat,dhbias,tilde_x_d,yb_d,ib);
  		cuda_ret = cudaDeviceSynchronize();
 		if (cuda_ret != cudaSuccess)
      			printf("Error in kernel");

  		cutStopTimer(timer32); time32 += cutGetTimerValue(timer32); cutDeleteTimer(timer32);
  		//cudaMemcpy(yb_h, yb_d,sizeof(double) * N_HIDDEN*BATCHSIZE, cudaMemcpyDeviceToHost);
		//printf("\nyb_h : "); for(int j=0;j<N_HIDDEN*BATCHSIZE;j++){	printf(" %f ",yb_h[j]); } 
    		//3.decode by reconstrution to get z
  		unsigned int timer33; cutCreateTimer(&timer33);	cutStartTimer(timer33);
		//
		int n33Threads = TILE_WIDTH;
		int n33Blocks = N_FEATS / n33Threads;
		int m33Blocks = BATCHSIZE / n33Threads;
		if (N_FEATS % n32Threads) n33Blocks++;
		if (BATCHSIZE % n33Threads) m33Blocks++;
		dim3 dimGrid33(m33Blocks, n33Blocks);
		dim3 dimBlock33(n33Threads,n33Threads);
    		 //dA_get_reconstructed_input_kernel<<<dimGrid2,dimBlock2>>>(N_HIDDEN,N_FEATS,dW_flat,dvbias,z_d,y_d,ib,BATCHSIZE);
    		dA_get_reconstructed_input_kernel<<<dimGrid33,dimBlock33>>>(N_HIDDEN,N_FEATS,dW_flat,dvbias,z_d,yb_d,ib,BATCHSIZE);
  		cuda_ret = cudaDeviceSynchronize();
 		if (cuda_ret != cudaSuccess)
      			printf("Error in kernel");
  		//cudaMemcpy(z_h, z_d,sizeof(double) * N_FEATS * BATCHSIZE, cudaMemcpyDeviceToHost);
  		//printf("\nz_h: "); for(int j=0;j<N_FEATS*BATCHSIZE;j++){ printf(" %f ",z_h[j]); }
  		cutStopTimer(timer33); time33 += cutGetTimerValue(timer33); cutDeleteTimer(timer33);
    		//4. Update error in reconstruction - visible error for every minibatch by atomic add kernel
  		unsigned int timer34; cutCreateTimer(&timer34);	cutStartTimer(timer34);
		dim3 dimGrid34(BATCHSIZE);
     		dim3 dimBlock34(N_FEATS);
    		dA_L_vbias_kernel<<<dimGrid34,dimBlock34>>>(model_h->N,dL_vbias,dvbias,N_FEATS,X_d,z_d,offsetXval,BATCHSIZE,lr);
  		cuda_ret = cudaDeviceSynchronize();
 		if (cuda_ret != cudaSuccess)
      			printf("Error in kernel");
		//cudaMemcpy(L_vbias, dL_vbias,sizeof(double) *N_FEATS*BATCHSIZE, cudaMemcpyDeviceToHost);
		//printf("\nL_vbias: "); for(int j=0;j<N_FEATS*BATCHSIZE;j++){ printf(" %f ",L_vbias[j]); }				
  		cutStopTimer(timer34); time34 += cutGetTimerValue(timer34); cutDeleteTimer(timer34);
      		//5.Update error in hidden units outputs, we would use it to update weights
  		unsigned int timer35; cutCreateTimer(&timer35);	cutStartTimer(timer35);
		int n35Threads = TILE_WIDTH;
		int n35Blocks = N_HIDDEN / n35Threads;
		int m35Blocks = BATCHSIZE / n35Threads;
		if (N_HIDDEN % n35Threads) n35Blocks++;
		if (BATCHSIZE % n35Threads) m35Blocks++;
		dim3 dimGrid35(m32Blocks, n32Blocks);
		dim3 dimBlock35(n35Threads,n35Threads);
    		dA_L_hbias_kernel<<<dimGrid35,dimBlock35>>>(model_h->N,dL_vbias,dL_hbias,dhbias,N_HIDDEN,N_FEATS,yb_d,dW_flat,ib,BATCHSIZE,lr);
  		cuda_ret = cudaDeviceSynchronize();
 		if (cuda_ret != cudaSuccess)
      			printf("Error in kernel");
		//cudaMemcpy(L_hbias, dL_hbias,sizeof(double) *N_HIDDEN*BATCHSIZE, cudaMemcpyDeviceToHost);
		//printf("\nL_hbias: "); for(int j=0;j<N_HIDDEN*BATCHSIZE;j++){ printf(" %f ",L_hbias[j]); }				
  		cutStopTimer(timer35); time35 += cutGetTimerValue(timer35); cutDeleteTimer(timer35);
    		//6. Weights updates for minibatch
  		unsigned int timer36; cutCreateTimer(&timer36);	cutStartTimer(timer36);
  		     //2. encode to get hidden values y
  		dim3 dimGrid36(N_HIDDEN);
  		dim3 dimBlock36(N_FEATS);
    		dA_W_kernel<<<dimGrid36,dimBlock36>>>(model_h->N,dL_vbias,dL_hbias,model_h->n_hidden,model_h->n_visible,
							yb_d,dW_flat,tilde_x_d,ib,BATCHSIZE,lr);
  		cuda_ret = cudaDeviceSynchronize();
 		if (cuda_ret != cudaSuccess)
      			printf("Error in kernel");
  		//cudaMemcpy(hW_flat, dW_flat,sizeof(double) * N_HIDDEN * N_FEATS, cudaMemcpyDeviceToHost);
		//printf("\nhW_flat: "); for(int j=0;j<N_HIDDEN*N_FEATS;j++){ printf(" %f ",hW_flat[j]); }				
  		cutStopTimer(timer36); time36 += cutGetTimerValue(timer36); cutDeleteTimer(timer36);
 	}
	//******************************************************************************************************

  }
  //
  //
  unsigned int timer37; cutCreateTimer(&timer37);cutStartTimer(timer37);
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
      printf("Error in kernel");
  cutStopTimer(timer37); time37 += cutGetTimerValue(timer37); cutDeleteTimer(timer37);
  //
  unsigned int timer38; cutCreateTimer(&timer38);cutStartTimer(timer38);
  //cudaMemcpy(tilde_x_h, tilde_x_d,sizeof(double) * N_OBS * N_FEATS, cudaMemcpyDeviceToHost);
  //cudaMemcpy(yb_h, yb_d,sizeof(double) * N_HIDDEN*BATCHSIZE, cudaMemcpyDeviceToHost);
  //printf("\nyb_h : "); for(int j=0;j<N_HIDDEN*BATCHSIZE;j++){ printf(" %f ",yb_h[j]); }

  //cudaMemcpy(z_h, z_d,sizeof(double) * 1*N_FEATS, cudaMemcpyDeviceToHost);
  //cudaMemcpy(L_vbias, dL_vbias,sizeof(double) * 1*N_FEATS, cudaMemcpyDeviceToHost);
  cudaMemcpy(model_h->vbias, dvbias,sizeof(double) * N_FEATS, cudaMemcpyDeviceToHost);
  //cudaMemcpy(L_hbias, dL_hbias,sizeof(double) * 1*N_HIDDEN, cudaMemcpyDeviceToHost);
  cudaMemcpy(model_h->hbias, dhbias,sizeof(double) * N_HIDDEN, cudaMemcpyDeviceToHost);
  cudaMemcpy(model_h->W_flat, dW_flat,sizeof(double) * N_HIDDEN * N_FEATS, cudaMemcpyDeviceToHost);
  //
  //*
  //printf("ibb is: %d\n",ib);
  //for(int i=0;i<N_OBS;i++) {
  //    printf("\ntile_x_h : "); for(int j=0;j<5;j++){ printf(" %f ",tilde_x_h[i*N_OBS+j]); }
  //}

  //for(int i=0;i<N_OBS;i++) {
  //printf("\nz_h: "); for(int j=0;j<5;j++){ printf(" %f ",z_h[j]); }
  //}
  //printf("\nh vbias: "); for(int j=0;j<N_FEATS;j++){ printf(" %f ",model_h->vbias[j]); }
  //printf("\nh hbias: "); for(int j=0;j<N_HIDDEN;j++){ printf(" %f ",model_h->hbias[j]); }
  //printf("\nh Weights: ");for(int j=0;j<N_HIDDEN*N_FEATS;j++){ printf(" %f ",model_h->W_flat[j]); }
  //*/
  //
  cutStopTimer(timer38); time38 += cutGetTimerValue(timer38); cutDeleteTimer(timer38);
  //cudaMemcpy(model_h->W, dW_flat,sizeof(double) * N_HIDDEN * N_FEATS, cudaMemcpyDeviceToHost);
  unsigned int timer39; cutCreateTimer(&timer39);	cutStartTimer(timer39);
  //We can not directly copy to W and W is used to test, so we populate it using a loop
  for(int i=0; i<model_h->n_hidden; i++) {
      for(int j=0; j<model_h->n_visible; j++) {
       	model_h->W[i][j] = model_h->W_flat[i*model_h->n_visible + j];
      }	
  }
  //
  cutStopTimer(timer39); time39 += cutGetTimerValue(timer39); cutDeleteTimer(timer39);

  //cutStopTimer(timer3);time3 = cutGetTimerValue(timer3); cutDeleteTimer(timer3);
  //
  // free up memory
  cudaFree(X_d); cudaFree(tilde_x_d); cudaFree(dW_flat);
  cudaFree(dhbias); cudaFree(dvbias); cudaFree(dL_vbias);
  cudaFree(dL_hbias); //cudaFree(y_d); 
  cudaFree(z_d); 
  //free(y_h);
  X_d = NULL;tilde_x_d = NULL; dW_flat = NULL; dhbias = NULL;
  dvbias = NULL; dL_vbias = NULL;dL_hbias = NULL;
  //y_d = NULL; 
  z_d = NULL; 

  //
  free(L_hbias);free(L_vbias);free(z_h);
  free(yb_h); yb_h = NULL;
  //free(y_h); y_h = NULL;
  free(tilde_x_h); tilde_x_h = NULL;
  free(hW_flat); hW_flat = NULL;
   //
  printf("\ntime1  : %f\n", time1);
  printf("time2  : %f\n", time2);
  printf("time3  : %f\n", time3);
  printf("time31 : %f\n", time31);
  printf("time32 : %f\n", time32);
  printf("time322 : %f\n", time322);
  printf("time33 : %f\n", time33);
  printf("time34 : %f\n", time34);
  printf("time35 : %f\n", time35);
  printf("time36 : %f\n", time36);
  printf("time37 : %f\n", time37);
  printf("time38 : %f\n", time38);
  printf("time39 : %f\n", time39);
  //
  
}


void test_dbn(void) {
  srand(0);
  int i, j, epoch;
  float device_time;
  float host_time;

  double learning_rate = 0.1;
  double corruption_level = 0.3;
  //int training_epochs = 100;
  int training_epochs = 100;

  //int train_N = 10;
  int train_N = N_OBS;
  int test_N = 2;
  int n_visible = N_FEATS;
  int n_hidden = N_HIDDEN;

  // training data
  /*
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
  */
  //*
  //  To increase size of array for testing
  int train_X[N_OBS][N_FEATS];
  for (int i1=0;i1<N_OBS;i1++) {
   for (int i2=0;i2<N_FEATS;i2++) {
        train_X[i1][i2] = rand() % 2;
     }
   }
  printf("training input : %f %f\n",train_X[0][0],train_X[N_OBS-1][N_FEATS-1]);
  //*/
  //

  // construct dA
  dA da_gold, da_h;
  dA__construct(&da_gold, train_N, n_visible, n_hidden, NULL, NULL, NULL);
  dA__construct(&da_h, train_N, n_visible, n_hidden, NULL, NULL, NULL);
  //memcpy(da_h.W, da_gold.W, sizeof(double)*n_hidden*n_visible);
  //memcpy(da_h.W_flat, da_gold.W_flat,sizeof(double)*n_hidden*n_visible);
  for (int i=0;i<n_hidden;i++) {
    for (int j=0;j<n_visible;j++) {
	da_h.W[i][j] = da_gold.W[i][j];
	da_h.W_flat[i*n_visible+j] = da_gold.W_flat[i*n_visible+j];
    }
  }
  //
  //*** to compare, initial values should be same for both the objects
  printf("da_gold W      : %f %f %f \n", da_gold.W[0][0],da_gold.W[0][1],da_gold.W[0][2]);
  printf("da_h W         : %f %f %f \n", da_h.W[0][0],da_h.W[0][1],da_h.W[0][2]);
  printf("da_gold W_flat : %f %f %f \n", da_gold.W_flat[0],da_gold.W_flat[1],da_gold.W_flat[2]);
  printf("da_h W_flat    : %f %f %f \n", da_h.W_flat[0],da_h.W_flat[1],da_h.W_flat[2]);
  printf("da_gold n_visi : %d \n", da_gold.n_visible);
  printf("da_h n_visible : %d \n", da_h.n_visible);
  printf("da_gold hbias  : %f %f %f \n",da_gold.hbias[0],da_gold.hbias[1],da_gold.hbias[2]);
  printf("da_h hbias     : %f %f %f \n",da_h.hbias[0],da_h.hbias[1],da_h.hbias[2]);
  printf("da_gold vbias  : %f %f %f \n",da_gold.vbias[0],da_gold.vbias[1],da_gold.vbias[2]);
  printf("da_h vbias     : %f %f %f \n",da_h.vbias[0],da_h.vbias[1],da_h.vbias[2]);
  //***
  printf("  Starting gold training..");
  unsigned int cputimer;
  cutCreateTimer(&cputimer);
  cutStartTimer(cputimer);
  // train using gold standard
  for(epoch=0; epoch<training_epochs; epoch++) {
    for(i=0; i<train_N; i++) {
      dA_train_gold(&da_gold, train_X[i], learning_rate, corruption_level);
    }
  }
  //
  cutStopTimer(cputimer);
  host_time = cutGetTimerValue(cputimer);
  cutDeleteTimer(cputimer);
  printf("Ending gold training..");
  //

  //* Start of GPU Kernel Call Code
  printf("Starting device training..");
  unsigned int gputimer;
  cutCreateTimer(&gputimer);
  cutStartTimer(gputimer);
  // train using kernel
  //dA_train_on_device(&da_h, train_X, learning_rate, corruption_level);
  dA_train_on_device1(&da_h, train_X, learning_rate, corruption_level,training_epochs);
  //
  cutStopTimer(gputimer);
  device_time = cutGetTimerValue(gputimer);
  cutDeleteTimer(gputimer);
  printf("Ending device training..");
  /*
  //
  printf("\nCPU Weights:"); for(int j=0;j<5;j++) {printf("%f ", da_gold.W_flat[j]);};
  printf("\nGPU Weights:"); for(int j=0;j<5;j++) {printf("%f ", da_h.W_flat[j]);};
  //
  //
  printf("\nCPU hbias:"); for(int j=0;j<5;j++) {printf("%f ", da_gold.hbias[j]);};
  printf("\nGPU hbias:"); for(int j=0;j<5;j++) {printf("%f ", da_h.hbias[j]);};
  //
  //
  printf("\nCPU vbias:"); for(int j=0;j<5;j++) {printf("%f ", da_gold.vbias[j]);};
  printf("\nGPU vbias:"); for(int j=0;j<5;j++) {printf("%f ", da_h.vbias[j]);};
  //
  */
  //* End of GPU Coode
  
  // test data
  /*
  int test_X[2][20] = {
    {1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0}
  };
  */
  //*  To increase size of array for testing
  int test_X[2][N_FEATS];
  for (int i1=0;i1<test_N;i1++) {
   for (int i2=0;i2<N_FEATS;i2++) {
        train_X[i1][i2] = rand() % 2;
     }
   }
  //*/
  double reconstructed_X[2][N_FEATS];

  printf("\n : CPU test now: \n");
  // test CPU using &da_gold object
  for(i=0; i<test_N; i++) {
    dA_reconstruct(&da_gold, test_X[i], reconstructed_X[i]);
    for(j=0; j<n_visible; j++) { printf("%.5f ", reconstructed_X[i][j]);}
    printf("\n");
  }

   printf("\n : GPU test now: \n");
  // test GPU using &da_h object
  for(i=0; i<test_N; i++) {
    dA_reconstruct(&da_h, test_X[i], reconstructed_X[i]);
    for(j=0; j<n_visible; j++) {printf("%.5f ", reconstructed_X[i][j]);}
    printf("\n");
  }

  // destruct dA
  dA__destruct(&da_gold);
  dA__destruct(&da_h);
    //
  printf("Host time          : %f\n", host_time);
  printf("Device time        : %f\n", device_time);
  printf("Speedup host/device: %fX\n", host_time/device_time);
  printf("***testing over***\n");

}


int main(int argc, char** argv) {
  test_dbn();
  return 0;
}

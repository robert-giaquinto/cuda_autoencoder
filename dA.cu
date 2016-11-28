// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// includes, project
#include <cutil.h>

// includes, kernel
#include <dA_kernel.cu>

// declarations
extern "C"
void dA_train_gold(dA*, int*, double, double);




double uniform(double min, double max) {
  return rand() / (RAND_MAX + 1.0) * (max - min) + min;
}


int binomial(int n, double p) {
  if(p < 0 || p > 1) return 0;

  int i;
  int c = 0;
  double r;

  for(i=0; i<n; i++) {
    r = rand() / (RAND_MAX + 1.0);
    if (r < p) c++;
  }

  return c;
}


double sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
}


void dA__construct(dA* model, int N, int n_visible, int n_hidden, double **W, double *hbias, double *vbias) {
  int i, j;
  double a = 1.0 / n_visible;

  model->N = N;
  model->n_visible = n_visible;
  model->n_hidden = n_hidden;

  if(W == NULL) {
    model->W = (double **)malloc(sizeof(double*) * n_hidden);
    model->W[0] = (double *)malloc(sizeof(double) * n_visible * n_hidden);
    for(i=0; i<n_hidden; i++) model->W[i] = model->W[0] + i * n_visible;

    for(i=0; i<n_hidden; i++) {
      for(j=0; j<n_visible; j++) {
        model->W[i][j] = uniform(-a, a);
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
  free(model->hbias);
  free(model->vbias);
}

void dA_get_corrupted_input(dA* model, int *x, int *tilde_x, double p) {
  int i;
  for(i=0; i<model->n_visible; i++) {
    if(x[i] == 0) {
      tilde_x[i] = 0;
    } else {
      tilde_x[i] = binomial(1, p);
    }
  }
}

// Encode
void dA_get_hidden_values(dA* model, int *x, double *y) {
  int i,j;
  for(i=0; i<model->n_hidden; i++) {
    y[i] = 0;
    for(j=0; j<model->n_visible; j++) {
      y[i] += model->W[i][j] * x[j];
    }
    y[i] += model->hbias[i];
    y[i] = sigmoid(y[i]);
  }
}

// Decode
void dA_get_reconstructed_input(dA* model, double *y, double *z) {
  int i, j;
  for(i=0; i<model->n_visible; i++) {
    z[i] = 0;
    for(j=0; j<model->n_hidden; j++) {
      z[i] += model->W[j][i] * y[j];
    }
    z[i] += model->vbias[i];
    z[i] = sigmoid(z[i]);
  }
}


void dA_reconstruct(dA* model, int *x, double *z) {
  double *y = (double *)malloc(sizeof(double) * model->n_hidden);

  dA_get_hidden_values(model, x, y);
  dA_get_reconstructed_input(model, y, z);

  free(y);
}



void dA_train_on_device(dA*, int*, double, double) {
  // call kernel function from here
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
  dA da;
  dA__construct(&da, train_N, n_visible, n_hidden, NULL, NULL, NULL);

  // train using gold standard
  for(epoch=0; epoch<training_epochs; epoch++) {
    for(i=0; i<train_N; i++) {
      dA_train_gold(&da, train_X[i], learning_rate, corruption_level);
    }
  }

  // test data
  int test_X[2][20] = {
    {1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0}
  };
  double reconstructed_X[2][20];


  // test
  for(i=0; i<test_N; i++) {
    dA_reconstruct(&da, test_X[i], reconstructed_X[i]);
    for(j=0; j<n_visible; j++) {
      printf("%.5f ", reconstructed_X[i][j]);
    }
    printf("\n");
  }


  // destruct dA
  dA__destruct(&da);
}


int main(int argc, char** argv) {
  test_dbn();
  return 0;
}

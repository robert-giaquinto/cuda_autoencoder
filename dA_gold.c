#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dA.h"

// export C interface
extern "C"
void dA_train_gold(dA*, float*, double, double);
void dA_get_corrupted_input(dA*, float*, float*, double);
void dA_get_hidden_values(dA*, float*, double*);
void dA_get_reconstructed_input(dA*, double*, double*);
float binomial(int n, double p);
double sigmoid(double x);


float binomial(int n, double p) {
  if(p < 0 || p > 1) return 0;

  int i;
  float c = 0.0f;
  double r;

  for(i=0; i<n; i++) {
    r = rand() / (RAND_MAX + 1.0);
    if (r < p) c += 1.0f;
  }

  return c;
}


double sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
}

void dA_get_corrupted_input(dA* model, float *x, float *tilde_x, double p) {
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
void dA_get_hidden_values(dA* model, float *x, double *y) {
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



// Train for one observation
void dA_train_gold(dA* model, float *x, double lr, double corruption_level) {
  int i, j;

  float *tilde_x = (float *)malloc(sizeof(float) * model->n_visible);
  double *y = (double *)malloc(sizeof(double) * model->n_hidden);
  double *z = (double *)malloc(sizeof(double) * model->n_visible);

  double *L_vbias = (double *)malloc(sizeof(double) * model->n_visible);
  double *L_hbias = (double *)malloc(sizeof(double) * model->n_hidden);

  double p = 1 - corruption_level;

  dA_get_corrupted_input(model, x, tilde_x, p);
  
  //printf("\n tile_x: "); for(int i=0;i<model->n_visible;i++){ printf(" %d ",tilde_x[i]); }
  
  dA_get_hidden_values(model, tilde_x, y);
  
  //printf("\n y: "); for(int i=0;i<model->n_hidden;i++){ printf(" %f ",y[i]); }

  dA_get_reconstructed_input(model, y, z);

  //printf("\n z: "); for(int i=0;i<model->n_visible;i++){ printf(" %f ",z[i]); }

  // vbias
  for(i=0; i<model->n_visible; i++) {
    L_vbias[i] = x[i] - z[i];
    model->vbias[i] += lr * L_vbias[i] / model->N;
  }

  //printf("\n vbias: "); for(int i=0;i<model->n_visible;i++){ printf(" %f ",model->vbias[i]);}

  // hbias
  for(i=0; i<model->n_hidden; i++) {
    L_hbias[i] = 0;
    for(j=0; j<model->n_visible; j++) {
      L_hbias[i] += model->W[i][j] * L_vbias[j];
    }
    L_hbias[i] *= y[i] * (1 - y[i]);

    model->hbias[i] += lr * L_hbias[i] / model->N;
  }

  //printf("\n hbias: "); for(int i=0;i<model->n_hidden;i++){ printf(" %f ",model->hbias[i]);}

  // W
  for(i=0; i<model->n_hidden; i++) {
    for(j=0; j<model->n_visible; j++) {
      model->W[i][j] += lr * (L_hbias[i] * tilde_x[j] + L_vbias[j] * y[i]) / model->N;
    }
  }

  //printf("\n W: "); for(int i=0;i<model->n_hidden*model->n_visible;i++){ printf(" %f ",model->W[i]);}

  free(L_hbias);
  free(L_vbias);
  free(z);
  free(y);
  free(tilde_x);
}

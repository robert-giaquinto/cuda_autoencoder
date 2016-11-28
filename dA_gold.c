#include <stdlib.h>
#include "dA.h"

// export C interface
extern "C"
void dA_train_gold(dA*, int*, double, double);

// gold standard
void dA_train_gold(dA* model, int *x, double lr, double corruption_level) {
  int i, j;

  int *tilde_x = (int *)malloc(sizeof(int) * model->n_visible);
  double *y = (double *)malloc(sizeof(double) * model->n_hidden);
  double *z = (double *)malloc(sizeof(double) * model->n_visible);

  double *L_vbias = (double *)malloc(sizeof(double) * model->n_visible);
  double *L_hbias = (double *)malloc(sizeof(double) * model->n_hidden);

  double p = 1 - corruption_level;

  dA_get_corrupted_input(model, x, tilde_x, p);
  dA_get_hidden_values(model, tilde_x, y);
  dA_get_reconstructed_input(model, y, z);

  // vbias
  for(i=0; i<model->n_visible; i++) {
    L_vbias[i] = x[i] - z[i];
    model->vbias[i] += lr * L_vbias[i] / model->N;
  }

  // hbias
  for(i=0; i<model->n_hidden; i++) {
    L_hbias[i] = 0;
    for(j=0; j<model->n_visible; j++) {
      L_hbias[i] += model->W[i][j] * L_vbias[j];
    }
    L_hbias[i] *= y[i] * (1 - y[i]);

    model->hbias[i] += lr * L_hbias[i] / model->N;
  }

  // W
  for(i=0; i<model->n_hidden; i++) {
    for(j=0; j<model->n_visible; j++) {
      model->W[i][j] += lr * (L_hbias[i] * tilde_x[j] + L_vbias[j] * y[i]) / model->N;
    }
  }

  free(L_hbias);
  free(L_vbias);
  free(z);
  free(y);
  free(tilde_x);
}

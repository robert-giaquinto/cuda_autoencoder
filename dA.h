//#ifndef _DA_H_
//#define _DA_H_

#define N_FEATS 784
#define N_OBS 100
#define N_TEST 2
#define BATCH_SIZE 1
#define N_HIDDEN 500
#define EPOCHS 100
#define RANDOM_MAX 100


typedef struct {
  int N;
  int n_visible;
  int n_hidden;
  double **W;
  double *W_flat;
  double *hbias;
  double *vbias;
} dA;


//#endif // #ifndef _DA_H_

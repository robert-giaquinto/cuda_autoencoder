//#ifndef _DA_H_
//#define _DA_H_

typedef struct {
  int N;
  int n_visible;
  int n_hidden;
  double **W;
  double *W_flat;
  double *hbias;
  double *vbias;
} dA;


#define RANDOM_MAX 100
//#define N_FEATS 20
#define N_FEATS 784
//#define N_OBS 10
#define N_OBS 1000
//#define N_TEST 2
#define N_TEST 100
#define BATCHSIZE 8
//#define N_HIDDEN 5
#define N_HIDDEN 500
#define BLOCKSIZE 8
#define TILE_WIDTH BLOCKSIZE
#define MAX_EPOCHS 10


#define CPU_TEST_FNAME "ee5351cpu-otest-btest-ttest.txt"
#define GPU_TEST_FNAME "ee5351gpu-otest-btest-ttest.txt"


//#endif // #ifndef _DA_H_

//#ifndef _DA_NAIVE_H_
//#define _DA_NAIVE_H_

#define ROOT_DIR "/home/class/smit7982/app/C/src/ee5351/"

// Settings for simple file
#define TRAIN_FILE ROOT_DIR"simple_train.txt"
#define TEST_FILE ROOT_DIR"simple_test.txt"
#define N_FEATS 20
#define N_OBS 10
#define N_TEST 2
#define BATCH_SIZE 1
#define N_HIDDEN 5
#define EPOCHS 100

// Settings for MNIST file
/*
#define TRAIN_FILE "/home/class/smit7982/app/C/src/ee5351/mnist_train.txt"
#define TEST_FILE "/home/class/smit7982/app/C/src/ee5351/mnist_test.txt"
#define N_FEATS 784
#define N_OBS 250
#define N_TEST 2
#define BATCH_SIZE 1
#define N_HIDDEN 500
#define EPOCHS 100
*/


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


//#endif // #ifndef _DA_NAIVE_H_

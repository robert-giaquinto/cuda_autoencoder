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


//#endif // #ifndef _DA_H_

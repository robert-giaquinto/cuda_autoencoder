//#ifndef _DA_H_
//#define _DA_H_

typedef struct {
  int N;
  int n_visible;
  int n_hidden;
  double **W;
  double *hbias;
  double *vbias;
} dA;


double uniform(double min, double max);
int binomial(int n, double p);
double sigmoid(double x);
void test_dbn(void);
void dA__construct(dA*, int, int, int, double**, double*, double*);
void dA__destruct(dA*);
void dA_get_corrupted_input(dA*, int*, int*, double);
void dA_get_hidden_values(dA*, int*, double*);
void dA_get_reconstructed_input(dA*, double*, double*);
void dA_reconstruct(dA*, int*, double*);

void dA_train_on_device(dA*, int*, double, double);



//#endif // #ifndef _DA_H_

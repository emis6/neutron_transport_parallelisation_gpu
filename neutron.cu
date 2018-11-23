#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <ctime>
#define OUTPUT_FILE "/tmp/absorbed.dat"

char info[] = "\
Usage:\n\
    neutron-seq H Nb C_c C_s\n\
\n\
    H  : épaisseur de la plaque\n\
    Nb : nombre d'échantillons\n\
    C_c: composante absorbante\n\
    C_s: componente diffusante\n\
\n\
Exemple d'execution : \n\
    neutron-seq 1.0 500000000 0.5 0.5\n\
";

/* setup_kernel <<<NbBlocks,NbThreadsParBloc>>> (devStates,unsigned(time(NULL))); //initialisation de l'état curandState pour chaque thread
 * notre gettimeofday()
 */
double my_gettimeofday(){
  struct timeval tmp_time;
  gettimeofday(&tmp_time, NULL);
  return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}


__global__ void setup_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * 64;
    /* Each thread gets same seed, a different sequence 
       number, no offset */
    curand_init(1234, id, 0, &state[id]);
}


__global__ void generate_kernel(curandState *state,
                                int step, unsigned int nbThread, dim3 TailleGrille, unsigned int N, float c, float c_c,float h, int* R, int* B,int* T,float* absorbed)
{
	int id = threadIdx.x + blockIdx.x *TailleGrille.x;

	/* Copy state to local memory for efficiency */
    curandState localState = state[id];
    /* Generate pseudo-random unsigned ints */
    float u;

    int r=0;
    int b=0;
    int t=0;
    float d, x, L; 

  	int debut_for = id*step ;
  	int fin_for = (id+1)*step;

  	if(id == nbThread) fin_for = N;
  
  for (int i = debut_for; i < fin_for; i++) {
    d = 0.0;
    x = 0.0;

    while (1) {

      u = curand_uniform(&localState);
      L = -(1 / c) * log(u);
      x = x + L * cos(d);  //initialisation de l'état curandState pour chaque thread
      if (x < 0) {
	r++;
	break;
      } else if (x >= h) {
	t++;
	break;
      } else if ((u = curand_uniform(&localState)) < c_c / c) {
	b++;
	absorbed[i] = x;
	break;
      } else {
	u = curand_uniform(&localState);
	d = u * M_PI;
      }
    }
  }
  /* Copy state back to global memory */
  state[id] = localState;
  r=atomicAdd(R,r);
  b=atomicAdd(B,b);
  t=atomicAdd(T,t);
}



int main(int argc, char *argv[]) {
  // La distance moyenne entre les interactions neutron/atome est 1/c. 
  // c_c et c_s sont les composantes absorbantes et diffusantes de c. 
  float c, c_c, c_s;
  // épaisseur de la plaque
  float h;
  // nombre d'échantillons
  int n;
  // nombre de neutrons refléchis, absorbés et transmis
  int r, b, t;
  // chronometrage
  double start, finish;
  int j = 0; // compteurs 

  if( argc == 1)
    fprintf( stderr, "%s\n", info);

  // valeurs par defaut
  h = 1.0;
  n = 500000000;
  c_c = 0.5;
  c_s = 0.5;

  // recuperation des parametres
  if (argc > 1)
    h = atof(argv[1]);
  if (argc > 2)
    n = atoi(argv[2]);
  if (argc > 3)
    c_c = atof(argv[3]);
  if (argc > 4)
    c_s = atof(argv[4]);
  r = b = t = 0;
  c = c_c + c_s;

    // affichage des parametres pour verificatrion
    printf("Épaisseur de la plaque : %4.g\n", h);
    printf("Nombre d'échantillons  : %d\n", n);
    printf("C_c : %g\n", c_c);
    printf("C_s : %g\n", c_s);

    float *absorbed;
    absorbed = (float *) calloc(n, sizeof(float));
    dim3 TailleGrille, ThreadparBlock;

	ThreadparBlock.x = 1024; //32*32
	ThreadparBlock.y = 1;
	ThreadparBlock.z = 1;

	TailleGrille.x = 1024;
	TailleGrille.y = 1;
	TailleGrille.z = 1;


int nbThread = TailleGrille.x*ThreadparBlock.x;

int step = n/nbThread; //nb de neutrons gérés par chaque thread
  // debut du chronometrage
  start = my_gettimeofday();

  //appel kernel 

  // fin du chronometrage
  finish = my_gettimeofday();

  printf("\nPourcentage des neutrons refléchis : %4.2g\n", (float) r / (float) n);
  printf("Pourcentage des neutrons absorbés : %4.2g\n", (float) b / (float) n);
  printf("Pourcentage des neutrons transmis : %4.2g\n", (float) t / (float) n);

  printf("\nTemps total de calcul: %.8g sec\n", finish - start);
  printf("Millions de neutrons /s: %.2g\n", (double) n / ((finish - start)*1e6));

  // ouverture du fichier pour ecrire les positions des neutrons absorbés
  FILE *f_handle = fopen(OUTPUT_FILE, "w");
  if (!f_handle) {
    fprintf(stderr, "Cannot open " OUTPUT_FILE "\n");
    exit(EXIT_FAILURE);
  }

  for (j = 0; j < b; j++)
    fprintf(f_handle, "%f\n", absorbed[j]);

  // fermeture du fichier
  fclose(f_handle);
  printf("Result written in " OUTPUT_FILE "\n"); 

  free(absorbed);
}

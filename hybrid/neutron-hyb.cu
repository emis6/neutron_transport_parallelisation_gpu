#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <ctime>
#include <omp.h>


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


struct drand48_data alea_buffer;

void init_uniform_random_number() {
  srand48_r(0, &alea_buffer);
}

float uniform_random_number() {
  double res = 0.0; 
  drand48_r(&alea_buffer,&res);
  return res;
}

/* 
 * notre gettimeofday()
 */
double my_gettimeofday(){
  struct timeval tmp_time;
  gettimeofday(&tmp_time, NULL);
  return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}


__global__ void setup_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence 
       number, no offset */
    curand_init(0, id, 0, &state[id]);
}





__global__ void generate_kernel(curandState *state,
    int step, unsigned int nbThread, dim3 TailleGrille, unsigned int N, float c, float c_c,float h, int* R, int* B,int* T,float* absorbed, int* j)
{
	int id = threadIdx.x + blockIdx.x *TailleGrille.x;

	/* Copy state to local memory for efficiency */
    curandState localState = state[id];
    /* Generate pseudo-random unsigned ints */
    float u;

    int r=0;
    int b=0;
    int t=0;

    __shared__ int Rtab[1024];
    __shared__ int Btab[1024];
    __shared__ int Ttab[1024];

    
    float d, x, L; 

    int debut_for = id*step ;
    int fin_for = (id+1)*step;

    if(id == nbThread) fin_for = N;

  //#pragma omp parallel for private(x, u, d, L) reduction(+:t,r,b)
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
                atomicAdd(j,1);
                absorbed[(*j)] = x;
                break;
            } else {
                u = curand_uniform(&localState);
                d = u * M_PI;
            }
        }
    }

    Rtab[threadIdx.x]=r;
    Btab[threadIdx.x]=b;
    Ttab[threadIdx.x]=t;

    //reduction :
__syncthreads();
    int k=blockDim.x/2;
    while (k>0){
        if(threadIdx.x<k){
            Rtab[threadIdx.x]+=Rtab[threadIdx.x +k];
            Btab[threadIdx.x]+=Btab[threadIdx.x +k];
            Ttab[threadIdx.x]+=Ttab[threadIdx.x +k];
        }
        k/=2;
        __syncthreads();
    } 
    if(threadIdx.x==0){     //Copy state back to global memory 
        atomicAdd(R,Rtab[0]);
        atomicAdd(B,Btab[0]);
        atomicAdd(T,Ttab[0]);
    }

    state[id] = localState;

}



int main(int argc, char *argv[]) {
    // La distance moyenne entre les interactions neutron/atome est 1/c. 
    // c_c et c_s sont les composantes absorbantes et diffusantes de c. 



    float *absorbed_gpu, *absorbed_cpu;
    float c, c_c, c_s;
    // épaisseur de la plaque
    float h;
    // nombre d'échantillons
    int n, n_loc;
    // nombre de neutrons refléchis, absorbés et transmis
    int r, b, t;
    int r_gpu, b_gpu, t_gpu;
    int r_cpu, b_cpu, t_cpu;

    // chronometrage
    double start, finish;
    int j = 0; // compteurs 

    float pourcentage_gpu = 0.8;

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
    c = c_c + c_s;

    //compteurs
    r = b = t = 0;
    r_gpu= b_gpu= t_gpu = 0;
    r_cpu = b_cpu= t_cpu=0;


    // affichage des parametres pour verificatrion
    printf("Épaisseur de la plaque : %4.g\n", h);
    printf("Nombre d'échantillons  : %d\n", n);
    // printf("C_c : %g\n", c_c);
    // printf("C_s : %g\n", c_s);

n_loc = (int)(pourcentage_gpu * n);
            //n_loc = n - 10000000;
           
            


    
            float *d_absorbed;
            int size = n_loc*sizeof(float);
            cudaMalloc((void**)&d_absorbed, size);

            int *d_r, *d_b, *d_t, *d_j;
           

         cudaMalloc((void**)&d_r, sizeof(int));
            cudaMalloc((void**)&d_b, sizeof(int));
            cudaMalloc((void**)&d_t, sizeof(int));
            cudaMalloc((void**)&d_j, sizeof(int));

    // Transfert CPU -> GPU
            cudaMemcpy(d_absorbed, absorbed_gpu, size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_r, &r_gpu, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, &b_gpu, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_t, &t_gpu, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_j, &j, sizeof(int), cudaMemcpyHostToDevice);

    // Definition nombre de threads
            dim3 TailleGrille, ThreadparBlock;

            ThreadparBlock.x = 1024; //32*32
            ThreadparBlock.y = 1;
            ThreadparBlock.z = 1;
    
            TailleGrille.x = 1024;
            TailleGrille.y = 1;
            TailleGrille.z = 1;
    
    
            int nbThread = TailleGrille.x*ThreadparBlock.x;
    
            int step = n_loc/nbThread; //nb de neutrons gérés par chaque thread
    
            curandState *d_States;
        /* Allocation un vecteur d'etat par thread */
            cudaMalloc((void **)&d_States, nbThread*sizeof(curandState));  
    
     



      // debut du chronometrage
  start = my_gettimeofday();

    #pragma omp parallel num_threads(2)
    {
       
        if( omp_get_thread_num()== 0) // fait GPU
        {

absorbed_gpu = (float *) calloc(n_loc, sizeof(float)); 
    
     
        //appel kernel1 : initialisation states
            setup_kernel<<<TailleGrille,ThreadparBlock>>>(d_States);
    
        //appel kernel2 : calcul
            generate_kernel<<<TailleGrille,ThreadparBlock>>>(d_States,step,nbThread,TailleGrille,n_loc,c,c_c,h,d_r,d_b,d_t,d_absorbed, d_j);

    
        // Transfert GPU-> CPU
            cudaMemcpy(absorbed_gpu, d_absorbed, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(&r_gpu, d_r, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&b_gpu, d_b, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&t_gpu, d_t, sizeof(int), cudaMemcpyDeviceToHost);

            printf("--------> b = %d b = %d t= %d", r_gpu, b_gpu, t_gpu);

            cudaFree(d_absorbed);
            cudaFree(d_States);
            cudaFree(d_r);
            cudaFree(d_b);
            cudaFree(d_t);

        
        }

        else if(omp_get_thread_num()== 1) // fait CPU
        {
            n_loc = (int)( (1.0 - pourcentage_gpu) * n);
            
                    
            float L;
            // direction du neutron (0 <= d <= PI)
            float d;
            // variable aléatoire uniforme
            float u;
            // position de la particule (0 <= x <= h)
            float x;
            j =0;

            
            //float *absorbed_cpu;
            absorbed_cpu = (float *) calloc(n_loc, sizeof(float));
                
            init_uniform_random_number();
            int i;
            // #pragma omp parallel for private(x, u, d, L) reduction(+:t_cpu,r_cpu,b_cpu)
              for (i = 0; i < n_loc; i++) {
                d = 0.0;
                x = 0.0;

                while (1) {
            
                  u = uniform_random_number();
                  L = -(1 / c) * log(u);
                  x = x + L * cos(d);
                  if (x < 0) {
                r_cpu++;
                break;
                  } else if (x >= h) {
                t_cpu++;
                break;
                  } else if ((u = uniform_random_number()) < c_c / c) {
                b_cpu++;
                absorbed_cpu[j++] = x;
                break;
                  } else {
                u = uniform_random_number();
                d = u * M_PI;
                  }
                }
              }
            

        
        }
    } // fin de parallel
            
            // Calcul de vrais r, b, t:
            r = r_cpu + r_gpu;
            b = b_cpu + b_gpu;
            t = t_cpu + t_gpu;

            finish = my_gettimeofday();


            printf("\nPourcentage des neutrons refléchis GPU : %4.2g\n", (r_gpu +0.0 )/ (n + 0.0)*pourcentage_gpu);
            printf("Pourcentage des neutrons absorbés GPU: %4.2g\n", (float) b_gpu / (float) n*pourcentage_gpu);
            printf("Pourcentage des neutrons transmis GPU: %4.2g\n", (float) t_gpu / (float) n*pourcentage_gpu);


            printf("\nPourcentage des neutrons refléchis CPU : %4.2g\n", (float) r_cpu / (float) n*(1.0 - pourcentage_gpu));
            printf("Pourcentage des neutrons absorbés CPU: %4.2g\n", (float) b_cpu / (float) n*(1.0 - pourcentage_gpu));
            printf("Pourcentage des neutrons transmis CPU: %4.2g\n", (float) t_cpu / (float) n*(1.0 - pourcentage_gpu));
    

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
    
            for (j = 0; j < b_cpu; j++){
                    fprintf(f_handle, "%f\n", absorbed_cpu[j]);
            }

            for (j = 0; j < b_gpu; j++){
                    fprintf(f_handle, "%f\n", absorbed_gpu[j]);
            }
    
    // f    ermeture du fichier
            fclose(f_handle);
            printf("Result written in " OUTPUT_FILE "\n"); 
    
            free(absorbed_cpu);
            free(absorbed_gpu);
    
}

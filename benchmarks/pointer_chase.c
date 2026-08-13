#include "bench_common.h"
int main(int argc,char **argv){size_t n=wam_size(argc,argv,1<<18); size_t *next=malloc(n*sizeof(*next)); if(!next)return 1; wam_seed(19); for(size_t i=0;i<n;i++)next[i]=wam_rng()%n; size_t p=0; uint64_t s=0; for(size_t i=0;i<n*4;i++){p=next[p];s^=p;} wam_finish(s); free(next);}

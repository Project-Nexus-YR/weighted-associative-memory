#include "bench_common.h"
int main(int argc,char **argv){size_t n=wam_size(argc,argv,1<<17),m=n*2+1;uint64_t *t=calloc(m,sizeof(*t));if(!t)return 1;wam_seed(31);for(size_t i=0;i<n;i++){uint64_t k=wam_rng();size_t p=k%m;while(t[p]&&t[p]!=k)p=(p+1)%m;t[p]=k;}uint64_t s=0;for(size_t i=0;i<n*3;i++){uint64_t k=wam_rng();size_t p=k%m;while(t[p]&&t[p]!=k)p=(p+1)%m;s^=t[p];}wam_finish(s);free(t);}

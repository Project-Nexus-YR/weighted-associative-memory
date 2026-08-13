#include "bench_common.h"
int main(int argc,char **argv){size_t n=wam_size(argc,argv,1<<20);uint64_t *a=malloc(n*sizeof(*a));if(!a)return 1;for(size_t i=0;i<n;i++)a[i]=i*3;uint64_t s=0;for(size_t i=0;i<n;i+=4)s^=a[i];wam_finish(s);free(a);}

#include "bench_common.h"
int main(int argc,char **argv){size_t n=wam_size(argc,argv,1024),total=n*n;double *a=malloc(total*sizeof(*a));if(!a)return 1;for(size_t i=0;i<total;i++)a[i]=(double)(i%97);double s=0;for(size_t i=0;i<n;i++)for(size_t j=0;j<n;j++)s+=a[i*n+j];wam_finish((uint64_t)s);free(a);}

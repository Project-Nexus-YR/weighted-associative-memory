#include "bench_common.h"
static void qs(uint64_t*a,long l,long r){if(l>=r)return;uint64_t p=a[(l+r)/2];long i=l,j=r;while(i<=j){while(a[i]<p)i++;while(a[j]>p)j--;if(i<=j){uint64_t t=a[i];a[i++]=a[j];a[j--]=t;}}qs(a,l,j);qs(a,i,r);}
int main(int argc,char **argv){size_t n=wam_size(argc,argv,1<<18);uint64_t*a=malloc(n*sizeof(*a));if(!a)return 1;wam_seed(41);for(size_t i=0;i<n;i++)a[i]=wam_rng();qs(a,0,(long)n-1);uint64_t s=0;for(size_t i=0;i<n;i+=17)s^=a[i];wam_finish(s);free(a);}

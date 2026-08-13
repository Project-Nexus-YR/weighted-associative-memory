#include "bench_common.h"
int main(int argc,char **argv){size_t n=wam_size(argc,argv,1<<17); uint64_t *a=malloc(n*sizeof(*a)); if(!a)return 1; wam_seed(23); for(size_t i=0;i<n;i++)a[i]=wam_rng(); uint64_t s=0; for(size_t q=0;q<n*2;q++){size_t p=1; uint64_t key=wam_rng(); while(p<n){s^=a[p];p=2*p+(key>a[p]);}} wam_finish(s); free(a);}

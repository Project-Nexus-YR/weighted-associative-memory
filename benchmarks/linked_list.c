#include "bench_common.h"
typedef struct Node { struct Node *next; uint64_t value; } Node;
int main(int argc, char **argv) { size_t n=wam_size(argc,argv,1<<18); Node *a=calloc(n,sizeof(*a)); if(!a)return 1; wam_seed(7); for(size_t i=0;i<n;i++)WAM_STORE(&a[i].value,i^0x55); for(size_t i=0;i<n;i++){size_t j=wam_rng()%n; Node t=WAM_LOAD(&a[i]); WAM_STORE(&a[i],a[j]); WAM_STORE(&a[j],t);} for(size_t i=0;i+1<n;i++)WAM_STORE(&a[i].next,&a[i+1]); WAM_STORE(&a[n-1].next,NULL); uint64_t s=0; for(Node *p=&a[0];p;){s^=WAM_LOAD(&p->value);p=WAM_LOAD(&p->next);} wam_finish(s); free(a); }

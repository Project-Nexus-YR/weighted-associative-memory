#include "bench_common.h"
int main(int argc,char **argv){size_t n=wam_size(argc,argv,1<<16); size_t *st=malloc(n*sizeof(*st));uint8_t *seen=calloc(n,1);if(!st||!seen)return 1;size_t top=0;st[top++]=0;uint64_t s=0;while(top){size_t v=st[--top];if(seen[v])continue;seen[v]=1;s^=v;for(size_t e=0;e<8;e++)st[top++]=(v*1664525ULL+e*101+13)%n;}wam_finish(s);free(st);free(seen);}

#ifndef WAM_BENCH_COMMON_H
#define WAM_BENCH_COMMON_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uint64_t wam_rng_state = 0x9e3779b97f4a7c15ULL;
static uint64_t wam_rng(void) {
    wam_rng_state ^= wam_rng_state >> 12;
    wam_rng_state ^= wam_rng_state << 25;
    wam_rng_state ^= wam_rng_state >> 27;
    return wam_rng_state * 0x2545F4914F6CDD1DULL;
}
static void wam_seed(uint64_t seed) { wam_rng_state = seed ? seed : 1; }
static void wam_finish(uint64_t checksum) { printf("checksum=%llu\n", (unsigned long long)checksum); }
static size_t wam_size(int argc, char **argv, size_t fallback) {
    return argc > 1 ? (size_t)strtoull(argv[1], NULL, 10) : fallback;
}

#endif

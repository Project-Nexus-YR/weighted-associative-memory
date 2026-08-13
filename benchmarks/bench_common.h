#ifndef WAM_BENCH_COMMON_H
#define WAM_BENCH_COMMON_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stddef.h>

#ifdef WAM_SOURCE_TRACE
static FILE *wam_trace_file;
static void wam_trace_record(char kind, const void *address) {
    if (wam_trace_file) fprintf(wam_trace_file, "%c 0x%" PRIxPTR "\n", kind, (uintptr_t)address);
}
__attribute__((constructor)) static void wam_trace_open(void) {
    const char *path = getenv("WAM_TRACE_OUT");
    if (path) wam_trace_file = fopen(path, "w");
}
__attribute__((destructor)) static void wam_trace_close(void) {
    if (wam_trace_file) fclose(wam_trace_file);
}
#define WAM_LOAD(ptr) ({ __typeof__(ptr) wam_p = (ptr); wam_trace_record('L', wam_p); *wam_p; })
#define WAM_STORE(ptr, value) do { __typeof__(ptr) wam_p = (ptr); wam_trace_record('S', wam_p); *wam_p = (value); } while (0)
#else
#define WAM_LOAD(ptr) (*(ptr))
#define WAM_STORE(ptr, value) do { *(ptr) = (value); } while (0)
#endif

static uint64_t wam_rng_state = 0x9e3779b97f4a7c15ULL;
static uint64_t wam_rng(void) {
    wam_rng_state ^= wam_rng_state >> 12;
    wam_rng_state ^= wam_rng_state << 25;
    wam_rng_state ^= wam_rng_state >> 27;
    return wam_rng_state * 0x2545F4914F6CDD1DULL;
}
static void wam_seed(uint64_t seed) { const char *override = getenv("WAM_SEED"); if (override) seed = strtoull(override, NULL, 10); wam_rng_state = seed ? seed : 1; }
static void wam_finish(uint64_t checksum) { printf("checksum=%llu\n", (unsigned long long)checksum); }
static size_t wam_size(int argc, char **argv, size_t fallback) {
    return argc > 1 ? (size_t)strtoull(argv[1], NULL, 10) : fallback;
}

#endif

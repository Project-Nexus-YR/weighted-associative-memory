#include "hybrid_spp_wam.h"

#include <iostream>

void hybrid_spp_wam::prefetcher_initialize()
{
  spp_dev::prefetcher_initialize();
  sidecar_.prefetcher_initialize();
  primary_accesses_ = primary_useful_ = sidecar_accesses_ = sidecar_enabled_ = selector_abstains_ = 0;
}

uint32_t hybrid_spp_wam::prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit,
                                                  bool useful_prefetch, access_type type, uint32_t metadata_in)
{
  // The native primary is run unchanged. The sidecar selector uses only
  // already-observed callback utility from a bounded recent window; it does
  // not inspect future demand addresses or train an ML policy.
  const uint32_t primary_metadata = spp_dev::prefetcher_cache_operate(addr, ip, cache_hit, useful_prefetch, type, metadata_in);
  if (type == access_type::PREFETCH)
    return primary_metadata;

  ++primary_accesses_;
  if (useful_prefetch)
    ++primary_useful_;
  if (primary_accesses_ >= WINDOW) {
    primary_accesses_ = 0;
    primary_useful_ = 0;
  }

  const bool primary_confident = primary_accesses_ >= 64 &&
                                 primary_useful_ * MIN_PRIMARY_USEFUL_RATE_DEN >=
                                     primary_accesses_ * MIN_PRIMARY_USEFUL_RATE_NUM;
  if (primary_confident) {
    ++selector_abstains_;
    return primary_metadata;
  }

  ++sidecar_accesses_;
  ++sidecar_enabled_;
  (void)sidecar_.prefetcher_cache_operate(addr, ip, cache_hit, useful_prefetch, type, primary_metadata);
  return primary_metadata;
}

uint32_t hybrid_spp_wam::prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch,
                                               champsim::address evicted_addr, uint32_t metadata_in)
{
  const uint32_t primary_metadata = spp_dev::prefetcher_cache_fill(addr, set, way, prefetch, evicted_addr, metadata_in);
  return sidecar_.prefetcher_cache_fill(addr, set, way, prefetch, evicted_addr, primary_metadata);
}

void hybrid_spp_wam::prefetcher_cycle_operate()
{
  spp_dev::prefetcher_cycle_operate();
}

void hybrid_spp_wam::prefetcher_final_stats()
{
  spp_dev::prefetcher_final_stats();
  sidecar_.prefetcher_final_stats();
  std::cout << "Hybrid-SPP-WAM primary_accesses: " << primary_accesses_
            << " sidecar_accesses: " << sidecar_accesses_
            << " sidecar_enabled: " << sidecar_enabled_
            << " selector_abstains: " << selector_abstains_ << "\n";
}

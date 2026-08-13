#ifndef HYBRID_SPP_WAM_H
#define HYBRID_SPP_WAM_H

#include <cstdint>

#include "modules.h"
#include "../../../external/champsim/prefetcher/spp_dev/spp_dev.h"
#include "../wam_h16/wam_h16.h"

// A deliberately small online sidecar: native SPP always supplies the
// primary stream, while DirectWAM-H16 is enabled only when recent primary
// usefulness is weak. This is an implementable selector, not an oracle.
class hybrid_spp_wam : public spp_dev {
public:
  explicit hybrid_spp_wam(CACHE* cache) : spp_dev(cache), sidecar_(cache) {}

  void prefetcher_initialize();
  uint32_t prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch,
                                    access_type type, uint32_t metadata_in);
  uint32_t prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch,
                                 champsim::address evicted_addr, uint32_t metadata_in);
  void prefetcher_cycle_operate();
  void prefetcher_final_stats();

private:
  static constexpr uint64_t WINDOW = 1024;
  static constexpr uint64_t MIN_PRIMARY_USEFUL_RATE_NUM = 1;
  static constexpr uint64_t MIN_PRIMARY_USEFUL_RATE_DEN = 20;
  uint64_t primary_accesses_ = 0;
  uint64_t primary_useful_ = 0;
  uint64_t sidecar_accesses_ = 0;
  uint64_t sidecar_enabled_ = 0;
  uint64_t selector_abstains_ = 0;
  wam_h16 sidecar_;
};

#endif

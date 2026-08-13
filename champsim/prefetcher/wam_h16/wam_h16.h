#ifndef WAM_H16_H
#define WAM_H16_H

#include <array>
#include <cstdint>
#include <deque>

#include "modules.h"

class wam_h16 : public champsim::modules::prefetcher {
public:
  using prefetcher::prefetcher;

  static constexpr std::size_t HORIZON = 16;
  static constexpr std::size_t CONTEXT_DEPTH = 4;
  static constexpr std::size_t TABLE_ENTRIES = 256;
  static constexpr uint8_t CONFIDENCE_MAX = 15;
  static constexpr uint8_t CONFIDENCE_THRESHOLD = 8;

  struct entry {
    uint64_t key = 0;
    int64_t delta = 0;
    uint16_t age = 0;
    uint8_t confidence = 0;
    bool valid = false;
    uint64_t reserved = 0;
  };
  static_assert(sizeof(entry) == 32, "WAM storage accounting assumes 32-byte entries");

  struct pending_context {
    uint64_t key = 0;
    uint64_t line = 0;
  };

  void prefetcher_initialize();
  uint32_t prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch,
                                    access_type type, uint32_t metadata_in);
  uint32_t prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch,
                                 champsim::address evicted_addr, uint32_t metadata_in);
  void prefetcher_final_stats();

private:
  std::array<entry, TABLE_ENTRIES> table_{};
  std::deque<pending_context> pending_;
  std::deque<uint64_t> history_;
  uint64_t accesses_ = 0;
  uint64_t predictions_ = 0;
  uint64_t issued_ = 0;
  uint64_t useful_ = 0;
  uint64_t training_updates_ = 0;
  uint64_t leaked_future_updates_ = 0;

  static uint64_t mix(uint64_t value);
  uint64_t line(champsim::address addr) const;
  uint64_t context_key() const;
  entry& slot(uint64_t key);
  void train(uint64_t key, uint64_t target_line);
};

#endif

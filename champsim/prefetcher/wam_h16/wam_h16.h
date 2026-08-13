#ifndef WAM_H16_H
#define WAM_H16_H

#include <array>
#include <cstdint>
#include <deque>
#include <fstream>
#include <unordered_map>

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

  struct diagnostic_event {
    uint64_t index;
    uint64_t raw_address;
    uint64_t normalized_line;
    uint64_t cycle;
    uint8_t warmup;
    uint8_t cache_hit;
    uint8_t access_type;
    uint8_t reserved;
    uint32_t padding;
  };
  static_assert(sizeof(diagnostic_event) == 40, "diagnostic event format must remain stable");

  struct diagnostic_prediction {
    uint64_t event_index;
    uint64_t context_key;
    uint64_t predicted_line;
    uint32_t support;
    uint8_t confidence;
    uint8_t above_threshold;
    uint8_t generated;
    uint8_t reserved;
  };
  static_assert(sizeof(diagnostic_prediction) == 32, "diagnostic prediction format must remain stable");

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

  // Opt-in diagnostic state. These fields are inert unless the diagnostic
  // environment variables are present; the production decision path below is
  // deliberately unchanged.
  bool diagnostic_enabled_ = false;
  std::ofstream diagnostic_events_;
  std::ofstream diagnostic_predictions_;
  uint64_t diagnostic_event_index_ = 0;
  uint64_t diagnostic_eligible_accesses_ = 0;
  uint64_t diagnostic_history_updates_ = 0;
  uint64_t diagnostic_contexts_formed_ = 0;
  uint64_t diagnostic_training_pairs_ = 0;
  uint64_t diagnostic_unique_contexts_ = 0;
  uint64_t diagnostic_context_revisits_ = 0;
  uint64_t diagnostic_prediction_lookups_ = 0;
  uint64_t diagnostic_prediction_hits_ = 0;
  uint64_t diagnostic_prediction_misses_ = 0;
  uint64_t diagnostic_predictions_generated_ = 0;
  uint64_t diagnostic_predictions_above_threshold_ = 0;
  uint64_t diagnostic_predictions_below_threshold_ = 0;
  uint64_t diagnostic_prefetch_requests_generated_ = 0;
  uint64_t diagnostic_prefetch_requests_accepted_ = 0;
  uint64_t diagnostic_prefetches_completed_ = 0;
  uint64_t diagnostic_prefetched_lines_demanded_later_ = 0;
  uint64_t diagnostic_prefetches_useful_ = 0;
  uint64_t diagnostic_pending_max_ = 0;
  uint64_t diagnostic_pending_expired_ = 0;
  uint64_t diagnostic_hash_collisions_ = 0;
  uint64_t diagnostic_entry_insertions_ = 0;
  uint64_t diagnostic_entry_evictions_ = 0;
  uint64_t diagnostic_entry_reuses_before_eviction_ = 0;
  uint64_t diagnostic_hash_alias_misses_ = 0;
  uint64_t diagnostic_samples_written_ = 0;
  uint64_t diagnostic_shadow_predictions_ = 0;
  uint64_t diagnostic_shadow_correct_ = 0;
  uint64_t diagnostic_shadow_unresolved_ = 0;
  std::unordered_map<uint64_t, uint32_t> diagnostic_context_support_;
  std::array<uint64_t, CONFIDENCE_MAX + 1> diagnostic_confidence_hist_{};
  std::array<uint64_t, 17> diagnostic_support_hist_{};
  std::array<uint64_t, TABLE_ENTRIES> diagnostic_slot_keys_{};
  std::array<uint64_t, TABLE_ENTRIES> diagnostic_slot_insert_events_{};
  std::array<uint64_t, TABLE_ENTRIES> diagnostic_slot_reuses_{};
  std::deque<std::pair<uint64_t, uint64_t>> diagnostic_shadow_pending_;

  static uint64_t mix(uint64_t value);
  uint64_t line(champsim::address addr) const;
  uint64_t context_key() const;
  entry& slot(uint64_t key);
  void train(uint64_t key, uint64_t target_line);
  void diagnostic_reset();
  void diagnostic_record_event(champsim::address addr, uint64_t current_line, uint8_t cache_hit, access_type type);
  void diagnostic_record_lookup(uint64_t key, const entry& candidate, bool context_hit, bool above_threshold, bool generated, uint64_t predicted_line);
  void diagnostic_resolve_shadow(uint64_t target_line);
  uint64_t diagnostic_cycle() const;
};

#endif

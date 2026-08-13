#include "cache.h"
#include "set_associative_wam.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

uint64_t set_associative_wam::mix(uint64_t value)
{
  // This is intentionally byte-for-byte the DirectMappedWAM mix function.
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

uint64_t set_associative_wam::line(champsim::address addr) const
{
  return champsim::block_number{addr}.to<uint64_t>();
}

uint64_t set_associative_wam::context_key() const
{
  uint64_t key = 0x9e3779b97f4a7c15ULL;
  for (uint64_t value : history_)
    key = mix(key ^ (value + 0x9e3779b97f4a7c15ULL + (key << 6) + (key >> 2)));
  return key;
}

std::size_t set_associative_wam::set_index(uint64_t key) const
{
  return mix(key) % SETS;
}

int set_associative_wam::find_way(uint64_t key) const
{
  const auto set = set_index(key);
  for (std::size_t way = 0; way < WAYS; ++way) {
    if (table_[set][way].valid && table_[set][way].key == key)
      return static_cast<int>(way);
  }
  return -1;
}

set_associative_wam::entry& set_associative_wam::slot(uint64_t key, std::size_t& selected_way)
{
  const auto set = set_index(key);
  const auto matched_way = find_way(key);
  if (matched_way >= 0) {
    selected_way = static_cast<std::size_t>(matched_way);
    table_[set][selected_way].age = 0xffff;
    return table_[set][selected_way];
  }

  for (std::size_t way = 0; way < WAYS; ++way) {
    if (!table_[set][way].valid) {
      selected_way = way;
      if (diagnostic_enabled_) {
        ++diagnostic_entry_insertions_;
        ++diagnostic_empty_way_insertions_;
      }
      table_[set][way] = entry{};
      table_[set][way].key = key;
      table_[set][way].valid = true;
      table_[set][way].age = 0xffff;
      return table_[set][way];
    }
  }

  selected_way = next_way_[set] % WAYS;
  if (diagnostic_enabled_) {
    ++diagnostic_entry_insertions_;
    ++diagnostic_conflict_insertions_;
    ++diagnostic_replacement_events_;
    const auto flat = set * WAYS + selected_way;
    if (diagnostic_slot_reuses_[flat] > 0)
      ++diagnostic_reused_before_replacement_;
    diagnostic_set_ever_full_[set] = true;
  }
  table_[set][selected_way] = entry{};
  table_[set][selected_way].key = key;
  table_[set][selected_way].valid = true;
  table_[set][selected_way].age = 0xffff;
  next_way_[set] = static_cast<uint8_t>((selected_way + 1) % WAYS);
  return table_[set][selected_way];
}

void set_associative_wam::direct_shadow_train(uint64_t key, uint64_t target_line)
{
  auto& candidate = diagnostic_direct_shadow_[mix(key) % TABLE_ENTRIES];
  if (!candidate.valid || candidate.key == key || candidate.age == 0) {
    candidate.key = key;
    candidate.valid = true;
    candidate.age = 0xffff;
  } else {
    candidate.age = static_cast<uint16_t>(candidate.age - 1);
  }
  const int64_t target_delta = static_cast<int64_t>(target_line) - static_cast<int64_t>(pending_.front().line);
  if (candidate.key == key && candidate.delta == target_delta) {
    candidate.confidence = std::min<uint8_t>(CONFIDENCE_MAX, static_cast<uint8_t>(candidate.confidence + 1));
  } else if (candidate.key == key && candidate.confidence > 0) {
    candidate.confidence = static_cast<uint8_t>(candidate.confidence - 1);
  } else {
    candidate.key = key;
    candidate.delta = target_delta;
    candidate.confidence = 1;
    candidate.valid = true;
  }
  candidate.age = 0xffff;
}

void set_associative_wam::train(uint64_t key, uint64_t target_line)
{
  std::size_t selected_way = 0;
  auto& candidate = slot(key, selected_way);
  const int64_t target_delta = static_cast<int64_t>(target_line) - static_cast<int64_t>(pending_.front().line);
  if (candidate.key == key && candidate.delta == target_delta) {
    candidate.confidence = std::min<uint8_t>(CONFIDENCE_MAX, static_cast<uint8_t>(candidate.confidence + 1));
  } else if (candidate.key == key && candidate.confidence > 0) {
    candidate.confidence = static_cast<uint8_t>(candidate.confidence - 1);
  } else {
    candidate.key = key;
    candidate.delta = target_delta;
    candidate.confidence = 1;
    candidate.valid = true;
  }
  candidate.age = 0xffff;
  ++training_updates_;
  if (diagnostic_enabled_) {
    ++diagnostic_training_pairs_;
    diagnostic_slot_reuses_[set_index(key) * WAYS + selected_way] = 0;
  }
  if (diagnostic_enabled_)
    direct_shadow_train(key, target_line);
}

void set_associative_wam::prefetcher_initialize()
{
  for (auto& set : table_)
    set.fill(entry{});
  next_way_.fill(0);
  pending_.clear();
  history_.clear();
  accesses_ = predictions_ = issued_ = useful_ = training_updates_ = leaked_future_updates_ = 0;
  diagnostic_reset();
}

void set_associative_wam::diagnostic_reset()
{
  diagnostic_enabled_ = false;
  diagnostic_event_index_ = 0;
  diagnostic_eligible_accesses_ = diagnostic_history_updates_ = diagnostic_contexts_formed_ = diagnostic_training_pairs_ = 0;
  diagnostic_unique_contexts_ = diagnostic_context_revisits_ = diagnostic_prediction_lookups_ = diagnostic_prediction_hits_ = 0;
  diagnostic_prediction_misses_ = diagnostic_predictions_generated_ = diagnostic_predictions_above_threshold_ = 0;
  diagnostic_predictions_below_threshold_ = diagnostic_prefetch_requests_generated_ = diagnostic_prefetch_requests_accepted_ = 0;
  diagnostic_prefetches_completed_ = diagnostic_prefetched_lines_demanded_later_ = diagnostic_prefetches_useful_ = 0;
  diagnostic_pending_max_ = diagnostic_pending_expired_ = diagnostic_entry_insertions_ = diagnostic_empty_way_insertions_ = 0;
  diagnostic_conflict_insertions_ = diagnostic_replacement_events_ = diagnostic_reused_before_replacement_ = 0;
  diagnostic_set_lookups_ = diagnostic_tag_hits_ = diagnostic_tag_misses_ = diagnostic_set_conflict_lookups_ = 0;
  diagnostic_unresolved_conflict_events_ = diagnostic_same_set_distinct_contexts_ = diagnostic_direct_map_alias_equivalent_events_ = 0;
  diagnostic_samples_written_ = diagnostic_shadow_predictions_ = diagnostic_shadow_correct_ = diagnostic_shadow_unresolved_ = 0;
  diagnostic_context_support_.clear();
  diagnostic_confidence_hist_.fill(0);
  diagnostic_support_hist_.fill(0);
  diagnostic_way_hits_.fill(0);
  diagnostic_slot_reuses_.fill(0);
  diagnostic_set_ever_full_.fill(false);
  diagnostic_set_contexts_ = {};
  diagnostic_shadow_pending_.clear();
  diagnostic_direct_shadow_.fill(entry{});
  if (diagnostic_events_.is_open())
    diagnostic_events_.close();
  if (diagnostic_predictions_.is_open())
    diagnostic_predictions_.close();

  const char* event_path = std::getenv("WAM_DIAGNOSTIC_EVENT_PATH");
  const char* prediction_path = std::getenv("WAM_DIAGNOSTIC_PREDICTION_PATH");
  if (event_path == nullptr || prediction_path == nullptr)
    return;
  diagnostic_events_.open(event_path, std::ios::binary | std::ios::trunc);
  diagnostic_predictions_.open(prediction_path, std::ios::binary | std::ios::trunc);
  diagnostic_enabled_ = diagnostic_events_.good() && diagnostic_predictions_.good();
}

uint64_t set_associative_wam::diagnostic_cycle() const
{
  if (intern_ == nullptr || intern_->clock_period.count() == 0)
    return 0;
  return static_cast<uint64_t>(intern_->current_time.time_since_epoch() / intern_->clock_period);
}

void set_associative_wam::diagnostic_record_event(champsim::address addr, uint64_t current_line, uint8_t cache_hit, access_type type)
{
  if (!diagnostic_enabled_)
    return;
  diagnostic_event record{};
  record.index = diagnostic_event_index_;
  record.raw_address = addr.to<uint64_t>();
  record.normalized_line = current_line;
  record.cycle = diagnostic_cycle();
  record.warmup = intern_->warmup ? 1 : 0;
  record.cache_hit = cache_hit;
  record.access_type = static_cast<uint8_t>(champsim::to_underlying(type));
  diagnostic_events_.write(reinterpret_cast<const char*>(&record), sizeof(record));
  ++diagnostic_event_index_;
  ++diagnostic_eligible_accesses_;
}

void set_associative_wam::diagnostic_record_lookup(uint64_t key, const entry* candidate, bool context_hit, bool above_threshold, bool generated,
                                                  uint64_t predicted_line)
{
  if (!diagnostic_enabled_)
    return;
  ++diagnostic_prediction_lookups_;
  if (context_hit && candidate != nullptr) {
    ++diagnostic_prediction_hits_;
    const auto support = diagnostic_context_support_[key];
    diagnostic_confidence_hist_[std::min<std::size_t>(candidate->confidence, CONFIDENCE_MAX)]++;
    diagnostic_support_hist_[std::min<uint32_t>(support, 16)]++;
    ++diagnostic_slot_reuses_[set_index(key) * WAYS + static_cast<std::size_t>(find_way(key))];
    diagnostic_prediction record{};
    record.event_index = diagnostic_event_index_ - 1;
    record.context_key = key;
    record.predicted_line = predicted_line;
    record.support = support;
    record.confidence = candidate->confidence;
    record.above_threshold = above_threshold ? 1 : 0;
    record.generated = generated ? 1 : 0;
    diagnostic_predictions_.write(reinterpret_cast<const char*>(&record), sizeof(record));
    if (above_threshold)
      ++diagnostic_predictions_above_threshold_;
    else
      ++diagnostic_predictions_below_threshold_;
  } else {
    ++diagnostic_prediction_misses_;
  }
}

void set_associative_wam::diagnostic_resolve_shadow(uint64_t target_line)
{
  if (!diagnostic_enabled_)
    return;
  while (!diagnostic_shadow_pending_.empty() && diagnostic_shadow_pending_.front().first <= diagnostic_event_index_ - 1) {
    const auto [due_event, predicted_line] = diagnostic_shadow_pending_.front();
    diagnostic_shadow_pending_.pop_front();
    if (due_event == diagnostic_event_index_ - 1) {
      ++diagnostic_shadow_predictions_;
      if (predicted_line == target_line)
        ++diagnostic_shadow_correct_;
    }
  }
}

uint32_t set_associative_wam::prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit,
                                                      bool useful_prefetch, access_type type, uint32_t metadata_in)
{
  (void)ip;
  (void)metadata_in;
  if (useful_prefetch) {
    ++useful_;
    if (diagnostic_enabled_) {
      ++diagnostic_prefetched_lines_demanded_later_;
      ++diagnostic_prefetches_useful_;
    }
  }
  if (type == access_type::PREFETCH)
    return 0;

  const uint64_t current_line = line(addr);
  ++accesses_;
  if (diagnostic_enabled_) {
    diagnostic_record_event(addr, current_line, cache_hit, type);
    ++diagnostic_history_updates_;
    diagnostic_resolve_shadow(current_line);
  }

  if (pending_.size() >= HORIZON) {
    train(pending_.front().key, current_line);
    pending_.pop_front();
  }

  history_.push_back(current_line);
  if (history_.size() > CONTEXT_DEPTH)
    history_.pop_front();
  if (history_.size() != CONTEXT_DEPTH)
    return 0;

  const uint64_t key = context_key();
  const auto set = set_index(key);
  bool context_hit = false;
  bool above_threshold = false;
  bool generated = false;
  uint64_t predicted_line = 0;
  uint32_t support = 0;
  entry* candidate = nullptr;
  if (diagnostic_enabled_) {
    ++diagnostic_contexts_formed_;
    support = diagnostic_context_support_[key];
    if (support == 0)
      ++diagnostic_unique_contexts_;
    else
      ++diagnostic_context_revisits_;
    diagnostic_context_support_[key] = support + 1;
    auto& contexts = diagnostic_set_contexts_[set];
    const auto inserted = contexts.insert(key).second;
    if (inserted && contexts.size() > 1)
      ++diagnostic_same_set_distinct_contexts_;
    if (diagnostic_samples_written_ < 8) {
      std::cout << "WAM-SA-DIAG-SAMPLE event=" << (diagnostic_event_index_ - 1) << " raw=" << addr.to<uint64_t>() << " line=" << current_line
                << " key=" << key << " set=" << set << "\n";
      ++diagnostic_samples_written_;
    }
  }
  pending_.push_back({key, current_line});
  if (diagnostic_enabled_)
    diagnostic_pending_max_ = std::max<uint64_t>(diagnostic_pending_max_, pending_.size());
  if (pending_.size() > HORIZON)
    ++leaked_future_updates_;

  if (diagnostic_enabled_)
    ++diagnostic_set_lookups_;
  const int way = find_way(key);
  if (way >= 0) {
    context_hit = true;
    candidate = &table_[set][static_cast<std::size_t>(way)];
    if (diagnostic_enabled_) {
      ++diagnostic_tag_hits_;
      ++diagnostic_way_hits_[static_cast<std::size_t>(way)];
      ++diagnostic_slot_reuses_[set * WAYS + static_cast<std::size_t>(way)];
      const auto direct_slot = mix(key) % TABLE_ENTRIES;
      if (diagnostic_direct_shadow_[direct_slot].valid && diagnostic_direct_shadow_[direct_slot].key != key)
        ++diagnostic_direct_map_alias_equivalent_events_;
    }
  } else {
    if (diagnostic_enabled_) {
      ++diagnostic_tag_misses_;
      bool any_valid = false;
      bool all_valid = true;
      for (const auto& way_entry : table_[set]) {
        any_valid = any_valid || way_entry.valid;
        all_valid = all_valid && way_entry.valid;
      }
      if (any_valid)
        ++diagnostic_set_conflict_lookups_;
      if (all_valid)
        ++diagnostic_unresolved_conflict_events_;
    }
  }
  if (candidate != nullptr) {
    above_threshold = candidate->confidence >= CONFIDENCE_THRESHOLD;
    if (candidate->delta != 0) {
      const int64_t target = static_cast<int64_t>(current_line) + candidate->delta;
      if (target > 0)
        predicted_line = static_cast<uint64_t>(target);
    }
  }
  if (candidate != nullptr && above_threshold && candidate->delta != 0) {
    const int64_t target = static_cast<int64_t>(current_line) + candidate->delta;
    if (target > 0) {
      ++predictions_;
      generated = true;
      predicted_line = static_cast<uint64_t>(target);
      if (diagnostic_enabled_) {
        ++diagnostic_predictions_generated_;
        ++diagnostic_prefetch_requests_generated_;
        diagnostic_shadow_pending_.push_back({(diagnostic_event_index_ - 1) + HORIZON, predicted_line});
      }
      if (prefetch_line(champsim::address{champsim::block_number{static_cast<uint64_t>(target)}}, true, 0)) {
        ++issued_;
        if (diagnostic_enabled_)
          ++diagnostic_prefetch_requests_accepted_;
      }
    }
  }
  if (diagnostic_enabled_)
    diagnostic_record_lookup(key, candidate, context_hit, above_threshold, generated, predicted_line);
  return 0;
}

uint32_t set_associative_wam::prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch,
                                                   champsim::address evicted_addr, uint32_t metadata_in)
{
  (void)addr;
  (void)set;
  (void)way;
  (void)prefetch;
  (void)evicted_addr;
  if (diagnostic_enabled_ && prefetch != 0)
    ++diagnostic_prefetches_completed_;
  return metadata_in;
}

void set_associative_wam::prefetcher_final_stats()
{
  if (diagnostic_enabled_) {
    diagnostic_pending_expired_ = pending_.size();
    diagnostic_shadow_unresolved_ = diagnostic_shadow_pending_.size();
    diagnostic_events_.flush();
    diagnostic_predictions_.flush();
  }
  std::cout << "SetAssociativeWAM-H16 accesses: " << accesses_ << " predictions: " << predictions_ << " issued: " << issued_
            << " useful_callbacks: " << useful_ << " delayed_training_updates: " << training_updates_
            << " future_leakage_guard: " << leaked_future_updates_ << " state_bytes: " << LOGICAL_STATE_BYTES << "\n";
  if (!diagnostic_enabled_)
    return;
  std::size_t occupancy[WAYS + 1] = {};
  std::vector<std::size_t> pressure;
  pressure.reserve(SETS);
  std::size_t occupied_sets = 0;
  for (std::size_t set = 0; set < SETS; ++set) {
    std::size_t occupied = 0;
    for (const auto& way : table_[set])
      occupied += way.valid ? 1U : 0U;
    ++occupancy[occupied];
    if (occupied > 0)
      ++occupied_sets;
    pressure.push_back(diagnostic_set_contexts_[set].size());
    if (occupied == WAYS)
      diagnostic_set_ever_full_[set] = true;
  }
  std::sort(pressure.begin(), pressure.end());
  const auto quantile = [&pressure](double fraction) -> std::size_t {
    if (pressure.empty())
      return 0;
    const auto index = static_cast<std::size_t>(fraction * static_cast<double>(pressure.size() - 1));
    return pressure[index];
  };
  std::cout << "WAM-DIAG eligible_accesses_seen=" << diagnostic_eligible_accesses_ << " history_updates=" << diagnostic_history_updates_
            << " contexts_formed=" << diagnostic_contexts_formed_ << " training_pairs_created=" << diagnostic_training_pairs_
            << " unique_contexts=" << diagnostic_unique_contexts_ << " context_revisits=" << diagnostic_context_revisits_
            << " prediction_lookups=" << diagnostic_prediction_lookups_ << " prediction_context_hits=" << diagnostic_prediction_hits_
            << " prediction_context_misses=" << diagnostic_prediction_misses_ << " predictions_generated=" << diagnostic_predictions_generated_
            << " predictions_above_threshold=" << diagnostic_predictions_above_threshold_ << " predictions_below_threshold=" << diagnostic_predictions_below_threshold_
            << " prefetch_requests_generated=" << diagnostic_prefetch_requests_generated_ << " prefetch_requests_accepted=" << diagnostic_prefetch_requests_accepted_
            << " prefetches_completed=" << diagnostic_prefetches_completed_ << " prefetched_lines_demanded_later=" << diagnostic_prefetched_lines_demanded_later_
            << " prefetches_useful=" << diagnostic_prefetches_useful_ << " pending_max=" << diagnostic_pending_max_ << " pending_expired=" << diagnostic_pending_expired_
            << " entry_insertions=" << diagnostic_entry_insertions_ << " entry_evictions=" << diagnostic_replacement_events_
            << " entry_reuses_before_eviction=" << diagnostic_reused_before_replacement_ << " shadow_predictions=" << diagnostic_shadow_predictions_
            << " shadow_correct=" << diagnostic_shadow_correct_ << " shadow_unresolved=" << diagnostic_shadow_unresolved_
            << " request_rejected_or_duplicate=" << (diagnostic_prefetch_requests_generated_ - diagnostic_prefetch_requests_accepted_) << "\n";
  std::cout << "WAM-SA-DIAG set_lookups=" << diagnostic_set_lookups_ << " tag_hits=" << diagnostic_tag_hits_ << " tag_misses=" << diagnostic_tag_misses_
            << " empty_way_insertions=" << diagnostic_empty_way_insertions_ << " conflict_insertions=" << diagnostic_conflict_insertions_
            << " replacement_events=" << diagnostic_replacement_events_ << " reused_before_replacement=" << diagnostic_reused_before_replacement_
            << " way0_hits=" << diagnostic_way_hits_[0] << " way1_hits=" << diagnostic_way_hits_[1]
            << " way2_hits=" << diagnostic_way_hits_[2] << " way3_hits=" << diagnostic_way_hits_[3]
            << " set_conflict_lookups=" << diagnostic_set_conflict_lookups_ << " unresolved_conflict_events=" << diagnostic_unresolved_conflict_events_
            << " same_set_distinct_contexts=" << diagnostic_same_set_distinct_contexts_
            << " direct_map_alias_equivalent_events=" << diagnostic_direct_map_alias_equivalent_events_ << " occupied_sets=" << occupied_sets
            << " sets_ever_reaching_4way=" << std::count(diagnostic_set_ever_full_.begin(), diagnostic_set_ever_full_.end(), true)
            << " occupancy0=" << occupancy[0] << " occupancy1=" << occupancy[1] << " occupancy2=" << occupancy[2]
            << " occupancy3=" << occupancy[3] << " occupancy4=" << occupancy[4] << " active_contexts_per_occupied_set_mean="
            << (occupied_sets == 0 ? 0.0 : static_cast<double>(diagnostic_contexts_formed_) / static_cast<double>(occupied_sets))
            << " set_pressure_median=" << quantile(0.50) << " set_pressure_p90=" << quantile(0.90) << " set_pressure_p95=" << quantile(0.95)
            << " set_pressure_p99=" << quantile(0.99) << " set_pressure_max=" << pressure.back() << "\n";
  for (std::size_t i = 0; i < diagnostic_confidence_hist_.size(); ++i)
    std::cout << "WAM-DIAG-CONFIDENCE bin=" << i << " count=" << diagnostic_confidence_hist_[i] << "\n";
  for (std::size_t i = 0; i < diagnostic_support_hist_.size(); ++i)
    std::cout << "WAM-DIAG-SUPPORT bin=" << i << " count=" << diagnostic_support_hist_[i] << "\n";
  std::cout << "WAM-DIAG-LIMIT prefetch_duplicate_exact=not_exposed prefetch_rejection_reason=not_exposed cache_state_at_prediction=not_exposed prefetch_timeliness=not_exposed instruction_distance=not_exposed\n";
}

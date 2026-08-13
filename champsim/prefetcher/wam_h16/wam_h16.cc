#include "cache.h"
#include "wam_h16.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>

uint64_t wam_h16::mix(uint64_t value)
{
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

uint64_t wam_h16::line(champsim::address addr) const
{
  return champsim::block_number{addr}.to<uint64_t>();
}

uint64_t wam_h16::context_key() const
{
  uint64_t key = 0x9e3779b97f4a7c15ULL;
  for (uint64_t value : history_) {
    key = mix(key ^ (value + 0x9e3779b97f4a7c15ULL + (key << 6) + (key >> 2)));
  }
  return key;
}

wam_h16::entry& wam_h16::slot(uint64_t key)
{
  const auto bucket = mix(key) % TABLE_ENTRIES;
  auto& candidate = table_[bucket];
  if (!candidate.valid || candidate.key == key || candidate.age == 0) {
    if (candidate.valid && candidate.key != key) {
      if (diagnostic_enabled_) {
        ++diagnostic_entry_evictions_;
        if (diagnostic_slot_reuses_[bucket] > 0)
          ++diagnostic_entry_reuses_before_eviction_;
      }
      candidate = entry{};
    }
    if (diagnostic_enabled_ && (!candidate.valid || candidate.key != key)) {
      ++diagnostic_entry_insertions_;
      diagnostic_slot_keys_[bucket] = key;
      diagnostic_slot_insert_events_[bucket] = diagnostic_event_index_;
      diagnostic_slot_reuses_[bucket] = 0;
    }
    candidate.key = key;
    candidate.valid = true;
    candidate.age = 0xffff;
    return candidate;
  }
  candidate.age = static_cast<uint16_t>(candidate.age - 1);
  return candidate;
}

void wam_h16::train(uint64_t key, uint64_t target_line)
{
  auto& candidate = slot(key);
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
  if (diagnostic_enabled_)
    ++diagnostic_training_pairs_;
}

void wam_h16::prefetcher_initialize()
{
  table_.fill(entry{});
  pending_.clear();
  history_.clear();
  accesses_ = predictions_ = issued_ = useful_ = training_updates_ = leaked_future_updates_ = 0;
  diagnostic_reset();
}

void wam_h16::diagnostic_reset()
{
  diagnostic_enabled_ = false;
  diagnostic_event_index_ = 0;
  diagnostic_eligible_accesses_ = diagnostic_history_updates_ = diagnostic_contexts_formed_ = diagnostic_training_pairs_ = 0;
  diagnostic_unique_contexts_ = diagnostic_context_revisits_ = diagnostic_prediction_lookups_ = diagnostic_prediction_hits_ = 0;
  diagnostic_prediction_misses_ = diagnostic_predictions_generated_ = diagnostic_predictions_above_threshold_ = 0;
  diagnostic_predictions_below_threshold_ = diagnostic_prefetch_requests_generated_ = diagnostic_prefetch_requests_accepted_ = 0;
  diagnostic_prefetches_completed_ = diagnostic_prefetched_lines_demanded_later_ = diagnostic_prefetches_useful_ = 0;
  diagnostic_pending_max_ = diagnostic_pending_expired_ = diagnostic_hash_collisions_ = diagnostic_entry_insertions_ = 0;
  diagnostic_entry_evictions_ = diagnostic_entry_reuses_before_eviction_ = diagnostic_hash_alias_misses_ = diagnostic_samples_written_ = 0;
  diagnostic_shadow_predictions_ = diagnostic_shadow_correct_ = diagnostic_shadow_unresolved_ = 0;
  diagnostic_context_support_.clear();
  diagnostic_confidence_hist_.fill(0);
  diagnostic_support_hist_.fill(0);
  diagnostic_slot_keys_.fill(0);
  diagnostic_slot_insert_events_.fill(0);
  diagnostic_slot_reuses_.fill(0);
  diagnostic_shadow_pending_.clear();
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

uint64_t wam_h16::diagnostic_cycle() const
{
  if (intern_ == nullptr || intern_->clock_period.count() == 0)
    return 0;
  return static_cast<uint64_t>(intern_->current_time.time_since_epoch() / intern_->clock_period);
}

void wam_h16::diagnostic_record_event(champsim::address addr, uint64_t current_line, uint8_t cache_hit, access_type type)
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

void wam_h16::diagnostic_record_lookup(uint64_t key, const entry& candidate, bool context_hit, bool above_threshold, bool generated,
                                       uint64_t predicted_line)
{
  if (!diagnostic_enabled_)
    return;
  ++diagnostic_prediction_lookups_;
  if (context_hit) {
    ++diagnostic_prediction_hits_;
    const auto support = diagnostic_context_support_[key];
    diagnostic_confidence_hist_[std::min<std::size_t>(candidate.confidence, CONFIDENCE_MAX)]++;
    diagnostic_support_hist_[std::min<uint32_t>(support, 16)]++;
    ++diagnostic_slot_reuses_[mix(key) % TABLE_ENTRIES];
    diagnostic_prediction record{};
    record.event_index = diagnostic_event_index_ - 1;
    record.context_key = key;
    record.predicted_line = predicted_line;
    record.support = support;
    record.confidence = candidate.confidence;
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

void wam_h16::diagnostic_resolve_shadow(uint64_t target_line)
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

uint32_t wam_h16::prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit,
                                           bool useful_prefetch, access_type type, uint32_t metadata_in)
{
  (void)ip;
  (void)cache_hit;
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

  // This is the only training point: the target is revealed after exactly H16
  // qualifying accesses have arrived. No target address is read during issue.
  if (pending_.size() >= HORIZON) {
    train(pending_.front().key, current_line);
    pending_.pop_front();
  }

  history_.push_back(current_line);
  if (history_.size() > CONTEXT_DEPTH)
    history_.pop_front();
  if (history_.size() == CONTEXT_DEPTH) {
    const uint64_t key = context_key();
    bool context_hit = false;
    bool above_threshold = false;
    bool generated = false;
    uint64_t predicted_line = 0;
    uint32_t support = 0;
    if (diagnostic_enabled_) {
      ++diagnostic_contexts_formed_;
      support = diagnostic_context_support_[key];
      if (support == 0)
        ++diagnostic_unique_contexts_;
      else
        ++diagnostic_context_revisits_;
      diagnostic_context_support_[key] = support + 1;
      if (diagnostic_samples_written_ < 8) {
        std::cout << "WAM-DIAG-SAMPLE event=" << (diagnostic_event_index_ - 1) << " raw=" << addr.to<uint64_t>() << " line=" << current_line
                  << " key=" << key << "\n";
        ++diagnostic_samples_written_;
      }
    }
    pending_.push_back({key, current_line});
    if (diagnostic_enabled_)
      diagnostic_pending_max_ = std::max<uint64_t>(diagnostic_pending_max_, pending_.size());
    if (pending_.size() > HORIZON)
      ++leaked_future_updates_;

    auto& candidate = table_[mix(key) % TABLE_ENTRIES];
    if (diagnostic_enabled_) {
      context_hit = candidate.valid && candidate.key == key;
      if (candidate.valid && candidate.key != key) {
        ++diagnostic_hash_collisions_;
        ++diagnostic_hash_alias_misses_;
      }
      above_threshold = context_hit && candidate.confidence >= CONFIDENCE_THRESHOLD;
      if (context_hit && candidate.delta != 0) {
        const int64_t diagnostic_target = static_cast<int64_t>(current_line) + candidate.delta;
        if (diagnostic_target > 0)
          predicted_line = static_cast<uint64_t>(diagnostic_target);
      }
    }
    if (candidate.valid && candidate.key == key && candidate.confidence >= CONFIDENCE_THRESHOLD && candidate.delta != 0) {
      const int64_t target = static_cast<int64_t>(current_line) + candidate.delta;
      if (target > 0) {
        ++predictions_;
        predicted_line = static_cast<uint64_t>(target);
        generated = true;
        if (diagnostic_enabled_) {
          ++diagnostic_predictions_generated_;
          ++diagnostic_prefetch_requests_generated_;
          diagnostic_shadow_pending_.push_back({(diagnostic_event_index_ - 1) + HORIZON, predicted_line});
        }
        if (prefetch_line(champsim::address{champsim::block_number{static_cast<uint64_t>(target)}}, true, 0))
          {
            ++issued_;
            if (diagnostic_enabled_)
              ++diagnostic_prefetch_requests_accepted_;
          }
      }
    }
    if (diagnostic_enabled_)
      diagnostic_record_lookup(key, candidate, context_hit, above_threshold, generated, predicted_line);
  }
  return 0;
}

uint32_t wam_h16::prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch,
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

void wam_h16::prefetcher_final_stats()
{
  if (diagnostic_enabled_) {
    diagnostic_pending_expired_ = pending_.size();
    diagnostic_shadow_unresolved_ = diagnostic_shadow_pending_.size();
    diagnostic_events_.flush();
    diagnostic_predictions_.flush();
  }
  std::cout << "WAM-H16 accesses: " << accesses_ << " predictions: " << predictions_ << " issued: " << issued_
            << " useful_callbacks: " << useful_ << " delayed_training_updates: " << training_updates_
            << " future_leakage_guard: " << leaked_future_updates_ << " state_bytes: " << sizeof(table_) + sizeof(pending_context) * HORIZON
            << "\n";
  if (diagnostic_enabled_) {
    std::cout << "WAM-DIAG eligible_accesses_seen=" << diagnostic_eligible_accesses_
              << " history_updates=" << diagnostic_history_updates_ << " contexts_formed=" << diagnostic_contexts_formed_
              << " training_pairs_created=" << diagnostic_training_pairs_ << " unique_contexts=" << diagnostic_unique_contexts_
              << " context_revisits=" << diagnostic_context_revisits_ << " prediction_lookups=" << diagnostic_prediction_lookups_
              << " prediction_context_hits=" << diagnostic_prediction_hits_ << " prediction_context_misses=" << diagnostic_prediction_misses_
              << " predictions_generated=" << diagnostic_predictions_generated_ << " predictions_above_threshold=" << diagnostic_predictions_above_threshold_
              << " predictions_below_threshold=" << diagnostic_predictions_below_threshold_ << " prefetch_requests_generated="
              << diagnostic_prefetch_requests_generated_ << " prefetch_requests_accepted=" << diagnostic_prefetch_requests_accepted_
              << " prefetches_completed=" << diagnostic_prefetches_completed_ << " prefetched_lines_demanded_later="
              << diagnostic_prefetched_lines_demanded_later_ << " prefetches_useful=" << diagnostic_prefetches_useful_
              << " pending_max=" << diagnostic_pending_max_ << " pending_expired=" << diagnostic_pending_expired_
              << " hash_collisions=" << diagnostic_hash_collisions_ << " hash_alias_misses=" << diagnostic_hash_alias_misses_
              << " entry_insertions=" << diagnostic_entry_insertions_ << " entry_evictions=" << diagnostic_entry_evictions_
              << " entry_reuses_before_eviction=" << diagnostic_entry_reuses_before_eviction_ << " shadow_predictions="
              << diagnostic_shadow_predictions_ << " shadow_correct=" << diagnostic_shadow_correct_ << " shadow_unresolved="
              << diagnostic_shadow_unresolved_ << " request_rejected_or_duplicate="
              << (diagnostic_prefetch_requests_generated_ - diagnostic_prefetch_requests_accepted_) << "\n";
    for (std::size_t i = 0; i < diagnostic_confidence_hist_.size(); ++i)
      std::cout << "WAM-DIAG-CONFIDENCE bin=" << i << " count=" << diagnostic_confidence_hist_[i] << "\n";
    for (std::size_t i = 0; i < diagnostic_support_hist_.size(); ++i)
      std::cout << "WAM-DIAG-SUPPORT bin=" << i << " count=" << diagnostic_support_hist_[i] << "\n";
    std::cout << "WAM-DIAG-LIMIT prefetch_duplicate_exact=not_exposed prefetch_rejection_reason=not_exposed cache_state_at_prediction=not_exposed prefetch_timeliness=not_exposed instruction_distance=not_exposed\n";
  }
}

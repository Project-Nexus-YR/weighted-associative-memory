#include "wam_h16.h"

#include <algorithm>
#include <iostream>

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
      candidate = entry{};
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
}

void wam_h16::prefetcher_initialize()
{
  table_.fill(entry{});
  pending_.clear();
  history_.clear();
  accesses_ = predictions_ = issued_ = useful_ = training_updates_ = leaked_future_updates_ = 0;
}

uint32_t wam_h16::prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit,
                                           bool useful_prefetch, access_type type, uint32_t metadata_in)
{
  (void)ip;
  (void)cache_hit;
  (void)metadata_in;
  if (useful_prefetch)
    ++useful_;
  if (type == access_type::PREFETCH)
    return 0;

  const uint64_t current_line = line(addr);
  ++accesses_;

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
    pending_.push_back({key, current_line});
    if (pending_.size() > HORIZON)
      ++leaked_future_updates_;

    auto& candidate = table_[mix(key) % TABLE_ENTRIES];
    if (candidate.valid && candidate.key == key && candidate.confidence >= CONFIDENCE_THRESHOLD && candidate.delta != 0) {
      const int64_t target = static_cast<int64_t>(current_line) + candidate.delta;
      if (target > 0) {
        ++predictions_;
        if (prefetch_line(champsim::address{champsim::block_number{static_cast<uint64_t>(target)}}, true, 0))
          ++issued_;
      }
    }
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
  return metadata_in;
}

void wam_h16::prefetcher_final_stats()
{
  std::cout << "WAM-H16 accesses: " << accesses_ << " predictions: " << predictions_ << " issued: " << issued_
            << " useful_callbacks: " << useful_ << " delayed_training_updates: " << training_updates_
            << " future_leakage_guard: " << leaked_future_updates_ << " state_bytes: " << sizeof(table_) + sizeof(pending_context) * HORIZON
            << "\n";
}

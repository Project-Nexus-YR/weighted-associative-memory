from wam.trie import WeightedTrie


def test_frequency_weights_and_ranking():
    trie = WeightedTrie(context_depth=1, strategy="frequency")
    for next_address in (2, 2, 2, 3):
        trie.update([1], next_address)
    assert trie.predict([1], k=2) == [(2, 0.75), (3, 0.25)]


def test_ema_responds_to_recent_transition():
    trie = WeightedTrie(context_depth=1, strategy="ema", alpha=0.8)
    trie.update([1], 2)
    trie.update([1], 3)
    assert trie.predict([1], k=1)[0][0] == 3


def test_context_depth_disambiguates_same_last_address():
    trace = [0, 1, 10, 2, 1, 11] * 20
    depth_one = WeightedTrie(context_depth=1).fit(trace)
    depth_two = WeightedTrie(context_depth=2).fit(trace)
    assert depth_one.predict([0, 1], k=1)[0][0] != 10 or depth_one.predict([2, 1], k=1)[0][0] != 11
    assert depth_two.predict([0, 1], k=1)[0][0] == 10
    assert depth_two.predict([2, 1], k=1)[0][0] == 11


def test_threshold_and_storage_stats():
    trie = WeightedTrie(context_depth=2).fit([1, 2, 3, 1, 2, 4])
    assert trie.predict([1, 2], threshold=0.75) == []
    stats = trie.storage_stats()
    assert stats["nodes"] >= 1
    assert stats["edges"] > 0
    assert stats["estimated_bytes"] > 0

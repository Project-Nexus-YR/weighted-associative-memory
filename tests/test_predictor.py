from wam.predictor import LastTransitionPredictor, NextLinePredictor, WeightedTriePredictor


def test_weighted_predictor_returns_ranked_predictions():
    predictor = WeightedTriePredictor(context_depth=2).fit([0, 1, 10, 0, 1, 10, 0, 1, 11])
    predictions = predictor.predict([0, 1], k=2)
    assert predictions[0].address == 10
    assert predictions[0].weight > predictions[1].weight


def test_markov_and_next_line_baselines():
    markov = LastTransitionPredictor().fit([1, 2, 1, 2])
    assert markov.predict([1])[0].address == 2
    assert NextLinePredictor().predict([41])[0].address == 42

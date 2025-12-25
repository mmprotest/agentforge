from agentforge.scoring import score_exact, score_numeric_tolerance


def test_scorer_exact_match():
    result = score_exact("ok", "ok")
    assert result.passed
    assert result.score == 1.0


def test_scorer_numeric_tolerance():
    result = score_numeric_tolerance("1.001", "1.0", tolerance=0.01)
    assert result.passed
    assert result.score == 1.0

from agentforge.eval.normalize import NormalizationOptions, normalize_answer


def test_normalize_strips_whitespace():
    options = NormalizationOptions(strip_whitespace=True)
    assert normalize_answer("  hello   world \n", options) == "hello world"


def test_normalize_extracts_answer_letter():
    options = NormalizationOptions(extract_answer_letter=True)
    assert normalize_answer("The answer is C.", options) == "C"


def test_normalize_uppercases_single_letter():
    options = NormalizationOptions(uppercase_single_letter=True)
    assert normalize_answer("b", options) == "B"

from agentforge.eval.normalize import (
    NormalizationOptions,
    compare_tabular_outputs,
    normalize_answer,
    normalize_tabular,
)


def test_normalize_strips_whitespace():
    options = NormalizationOptions(strip_whitespace=True)
    assert normalize_answer("  hello   world \n", options) == "hello world"


def test_normalize_extracts_answer_letter():
    options = NormalizationOptions(extract_answer_letter=True)
    assert normalize_answer("The answer is C.", options) == "C"


def test_normalize_uppercases_single_letter():
    options = NormalizationOptions(uppercase_single_letter=True)
    assert normalize_answer("b", options) == "B"


def test_normalize_tabular_commas():
    assert normalize_tabular("a, b ,c\n1,2 , 3") == ["a,b,c", "1,2,3"]


def test_compare_tabular_outputs_allows_optional_header():
    expected = "a,b\n1,2"
    actual = "col1,col2\n1,2"
    assert compare_tabular_outputs(expected, actual, allow_optional_header=True)


def test_compare_tabular_outputs_requires_header_when_forbidden():
    expected = "a,b\n1,2"
    actual = "col1,col2\n1,2"
    assert not compare_tabular_outputs(expected, actual, allow_optional_header=False)

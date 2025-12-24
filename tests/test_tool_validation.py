from agentforge.tools.validation import validate_source


def test_validation_rejects_forbidden_imports():
    source = "import subprocess\n"
    errors = validate_source(source)
    assert any("subprocess" in error for error in errors)


def test_validation_rejects_eval():
    source = "eval('1+1')\n"
    errors = validate_source(source)
    assert any("eval" in error for error in errors)

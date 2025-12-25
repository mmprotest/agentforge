from agentforge.tools.builtins.unit_convert import convert_units


def test_convert_units_accepts_degree_symbols():
    assert convert_units(0, "°c", "°f") == 32
    assert convert_units(100, "°c", "°f") == 212

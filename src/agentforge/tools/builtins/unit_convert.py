"""Unit conversion tool."""

from __future__ import annotations

from pydantic import BaseModel

from agentforge.tools.base import Tool, ToolResult


class UnitConvertInput(BaseModel):
    value: float
    from_unit: str
    to_unit: str


class UnitConvertTool(Tool):
    name = "unit_convert"
    description = "Convert between common units of length, mass, temperature, and time."
    input_schema = UnitConvertInput

    def run(self, data: BaseModel) -> ToolResult:
        payload = UnitConvertInput.model_validate(data)
        value = payload.value
        from_unit = payload.from_unit.lower()
        to_unit = payload.to_unit.lower()
        converted = convert_units(value, from_unit, to_unit)
        return ToolResult(output={"value": converted, "unit": to_unit})


_LENGTH = {
    "m": 1.0,
    "km": 1000.0,
    "cm": 0.01,
    "mm": 0.001,
    "in": 0.0254,
    "ft": 0.3048,
    "yd": 0.9144,
    "mi": 1609.344,
}
_MASS = {
    "kg": 1.0,
    "g": 0.001,
    "lb": 0.45359237,
    "oz": 0.028349523125,
}
_TIME = {
    "s": 1.0,
    "sec": 1.0,
    "min": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "day": 86400.0,
}


def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    if from_unit in _LENGTH and to_unit in _LENGTH:
        return value * _LENGTH[from_unit] / _LENGTH[to_unit]
    if from_unit in _MASS and to_unit in _MASS:
        return value * _MASS[from_unit] / _MASS[to_unit]
    if from_unit in _TIME and to_unit in _TIME:
        return value * _TIME[from_unit] / _TIME[to_unit]
    if from_unit in {"c", "f", "k"} and to_unit in {"c", "f", "k"}:
        return _convert_temperature(value, from_unit, to_unit)
    raise ValueError(f"Unsupported unit conversion: {from_unit} -> {to_unit}")


def _convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    if from_unit == to_unit:
        return value
    if from_unit == "c":
        celsius = value
    elif from_unit == "f":
        celsius = (value - 32) * 5 / 9
    elif from_unit == "k":
        celsius = value - 273.15
    else:
        raise ValueError("Unsupported temperature unit")
    if to_unit == "c":
        return celsius
    if to_unit == "f":
        return celsius * 9 / 5 + 32
    if to_unit == "k":
        return celsius + 273.15
    raise ValueError("Unsupported temperature unit")

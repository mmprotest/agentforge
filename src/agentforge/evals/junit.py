"""JUnit XML output for eval reports."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any


def report_to_junit_xml(report: dict[str, Any]) -> str:
    testsuites = ET.Element("testsuites")
    packs = report.get("packs")
    if packs is None:
        packs = [
            {
                "eval_pack_id": report.get("eval_pack_id", "pack"),
                "cases": report.get("cases", []),
                "failures": report.get("failures", []),
            }
        ]

    for pack in packs:
        pack_id = str(pack.get("eval_pack_id") or pack.get("pack_id") or "pack")
        cases = pack.get("cases", []) or []
        failures = pack.get("failures", []) or []
        failure_ids = {str(item.get("id")): item for item in failures if "id" in item}
        suite = ET.SubElement(
            testsuites,
            "testsuite",
            {
                "name": pack_id,
                "tests": str(len(cases)),
                "failures": str(len(failure_ids)),
            },
        )
        for case in cases:
            case_id = str(case.get("id", "case"))
            testcase = ET.SubElement(suite, "testcase", {"name": case_id})
            if not case.get("passed", True):
                reason = str(failure_ids.get(case_id, {}).get("reason", "failed"))
                failure = ET.SubElement(testcase, "failure", {"message": reason})
                failure.text = reason
    return ET.tostring(testsuites, encoding="unicode")


def write_junit(path: Path, xml: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml, encoding="utf-8")

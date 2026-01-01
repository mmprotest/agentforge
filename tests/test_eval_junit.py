from __future__ import annotations

import xml.etree.ElementTree as ET

from agentforge.evals.junit import report_to_junit_xml


def test_junit_contains_testcases_and_failures() -> None:
    report = {
        "report_version": "0.1",
        "packs": [
            {
                "eval_pack_id": "sample",
                "cases": [
                    {"id": "case-1", "passed": True},
                    {"id": "case-2", "passed": False},
                ],
                "failures": [{"id": "case-2", "reason": "Score below threshold"}],
            }
        ],
    }
    xml = report_to_junit_xml(report)
    root = ET.fromstring(xml)
    testcases = root.findall(".//testcase")
    failures = root.findall(".//failure")
    assert len(testcases) == 2
    assert len(failures) == 1
    assert failures[0].attrib["message"] == "Score below threshold"

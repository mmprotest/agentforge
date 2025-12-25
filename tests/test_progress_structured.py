from agentforge.util.progress import ProgressTracker


def test_progress_detects_structured_and_intermediates():
    tracker = ProgressTracker()
    assert tracker.update(
        fact_count=0,
        tool_count=0,
        verifier_ok=None,
        issue_count=None,
        structured_fact_count=0,
        constraints_signature="{}",
        intermediates_signature="",
    )
    progressed = tracker.update(
        fact_count=0,
        tool_count=0,
        verifier_ok=None,
        issue_count=None,
        structured_fact_count=1,
        constraints_signature='{"requirements": ["A"]}',
        intermediates_signature="value",
    )
    assert progressed

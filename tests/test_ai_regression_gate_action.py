from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_entrypoint_module():
    entrypoint_path = Path(__file__).resolve().parents[1] / ".github" / "actions" / "ai-regression-gate" / "entrypoint.py"
    spec = spec_from_file_location("ai_regression_gate_entrypoint", entrypoint_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_eval_command_uses_global_workspace_flag() -> None:
    entrypoint = _load_entrypoint_module()
    report_path = Path("agentforge_eval_report.json")
    command = entrypoint._build_eval_command(
        workspace_id="demo",
        pack_name="sample",
        report_path=report_path,
        agentforge_args=["--foo", "bar"],
    )
    assert command[:3] == ["agentforge", "--workspace", "demo"]
    assert command[3:8] == ["eval", "run", "--pack", "sample", "--report"]
    assert command[8] == str(report_path)
    assert command[-2:] == ["--foo", "bar"]

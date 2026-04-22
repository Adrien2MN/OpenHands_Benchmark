import json
from pathlib import Path

from benchmarks.swebench.reporting import generate_instance_reports


def test_generate_instance_reports_creates_report_folder(tmp_path: Path) -> None:
    output_file = tmp_path / "output.jsonl"
    output_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "instance_id": "django__django-1",
                        "attempt": 1,
                        "instruction": "fix bug",
                        "test_result": {"git_patch": "diff --git a/a.py b/a.py"},
                        "metrics": {
                            "accumulated_usage": {
                                "prompt_tokens": 10,
                                "completion_tokens": 5,
                                "total_tokens": 15,
                            }
                        },
                        "history": [
                            {"timestamp": "2026-04-20T10:00:00Z"},
                            {"timestamp": "2026-04-20T10:00:03Z"},
                        ],
                        "error": None,
                    }
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary_path = generate_instance_reports(
        str(output_file),
        model_configs_dir=str(tmp_path / "missing_model_configs"),
        scenarios_dir=str(tmp_path / "missing_scenarios"),
        js_bridge_path=str(tmp_path / "missing_bridge.js"),
    )

    assert summary_path.exists()
    assert summary_path.parent.name == "report"

    instance_report = summary_path.parent / "report_django__django-1.json"
    assert instance_report.exists()

    payload = json.loads(instance_report.read_text(encoding="utf-8"))
    assert payload["usage"]["prompt_tokens"] == 10
    assert payload["usage"]["completion_tokens"] == 5
    assert payload["usage"]["total_tokens"] == 15
    assert payload["generation_seconds"] == 3.0
    assert payload["status"] == "ok"
    assert payload["token_footprint"]["energy_consumption_kwh"] is None


def test_generate_instance_reports_summary_totals(tmp_path: Path) -> None:
    output_file = tmp_path / "output.jsonl"
    rows = [
        {
            "instance_id": "repo__1",
            "test_result": {"git_patch": ""},
            "metrics": {
                "accumulated_usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                }
            },
            "history": [],
            "error": None,
        },
        {
            "instance_id": "repo__2",
            "test_result": {"git_patch": ""},
            "metrics": {
                "accumulated_usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 5,
                    "total_tokens": 9,
                }
            },
            "history": [],
            "error": "boom",
        },
    ]
    output_file.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    summary_path = generate_instance_reports(
        str(output_file),
        model_configs_dir=str(tmp_path / "missing_model_configs"),
        scenarios_dir=str(tmp_path / "missing_scenarios"),
        js_bridge_path=str(tmp_path / "missing_bridge.js"),
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["instances_total"] == 2
    assert summary["totals"]["prompt_tokens"] == 5
    assert summary["totals"]["completion_tokens"] == 7
    assert summary["totals"]["total_tokens"] == 12

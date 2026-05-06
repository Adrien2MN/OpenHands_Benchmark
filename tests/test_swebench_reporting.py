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
                        "test_result": {
                            "git_patch": "diff --git a/a.py b/a.py",
                            "proxy_cost": 2.5,
                        },
                        "metrics": {
                            "accumulated_cost": 1.25,
                            "accumulated_usage": {
                                "prompt_tokens": 10,
                                "completion_tokens": 5,
                                "total_tokens": 15,
                            },
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
    assert payload["cost"]["accumulated_cost_usd"] == 1.25
    assert payload["cost"]["proxy_cost_usd"] == 2.5
    assert payload["cost"]["effective_cost_usd"] == 1.25
    assert payload["token_footprint"]["energy_consumption_kwh"] is None


def test_generate_instance_reports_summary_totals(tmp_path: Path) -> None:
    output_file = tmp_path / "output.jsonl"
    rows = [
        {
            "instance_id": "repo__1",
            "test_result": {"git_patch": ""},
            "metrics": {
                "accumulated_cost": 1.5,
                "accumulated_usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                },
            },
            "history": [],
            "error": None,
        },
        {
            "instance_id": "repo__2",
            "test_result": {"git_patch": ""},
            "metrics": {
                "accumulated_cost": 2.5,
                "accumulated_usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 5,
                    "total_tokens": 9,
                },
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


def test_generate_instance_reports_uses_direct_model_footprint_json(
    tmp_path: Path,
) -> None:
    output_file = tmp_path / "output.jsonl"
    output_file.write_text(
        json.dumps(
            {
                "instance_id": "repo__3",
                "test_result": {"git_patch": "diff --git a/a.py b/a.py"},
                "metrics": {
                    "accumulated_usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                    }
                },
                "history": [],
                "error": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text(
        json.dumps({"llm": {"model": "gpt-4.1"}}),
        encoding="utf-8",
    )

    model_footprint_dir = tmp_path / "model_footprint"
    model_footprint_dir.mkdir(parents=True)
    (model_footprint_dir / "gpt41-ecologits2-token-2.json").write_text(
        json.dumps(
            {
                "scopes": [
                    {
                        "list": [
                            {
                                "consumer": {
                                    "name": "GPU energy",
                                    "consumptions": {"electricity": {"value": "2e-06"}},
                                },
                                "quantity": "1",
                                "source": {"emissions": {"co2e": {"value": "0.05"}}},
                            },
                            {
                                "consumer": {
                                    "name": "Server energy",
                                    "consumptions": {"electricity": {"value": "1e-06"}},
                                },
                                "quantity": "1",
                                "source": {"emissions": {"co2e": {"value": "0.05"}}},
                            },
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    summary_path = generate_instance_reports(
        str(output_file),
        model_configs_dir=str(model_footprint_dir),
        scenarios_dir=str(tmp_path / "unused_scenarios"),
        js_bridge_path=str(tmp_path / "missing_bridge.js"),
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["model_footprint"] is not None
    assert (
        summary["model_footprint"]["scenario_file"] == "gpt41-ecologits2-token-2.json"
    )
    assert summary["totals"]["energy_kwh"] > 0
    assert summary["totals"]["carbon_kg_co2e"] > 0
    assert summary["totals"]["accumulated_cost_usd"] == 4.0
    assert summary["totals"]["effective_cost_usd"] == 4.0

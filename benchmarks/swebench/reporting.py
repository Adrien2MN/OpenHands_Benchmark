from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from openhands.sdk import get_logger


logger = get_logger(__name__)


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_model_configs_dir() -> Path:
    return _workspace_root() / "SWE_bench_agent" / "configs" / "model"


def _default_scenarios_dir() -> Path:
    root = _workspace_root()
    ai_repo_scenarios = root / "AICarbonFootprintScenarios" / "scenarios"
    if ai_repo_scenarios.exists():
        return ai_repo_scenarios
    return root / "carbon-footprint-modeling-tool" / "scenarios"


def _default_js_bridge_path() -> Path:
    return _workspace_root() / "SWE_bench_agent" / "js_carbon_bridge.js"


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def _normalize_model_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _strip_yaml_scalar(value: str) -> str:
    out = value.strip()
    if not out:
        return ""
    if out[0] in {'"', "'"} and out[-1:] == out[0]:
        out = out[1:-1].strip()
    return out


def _load_model_carbon_scenarios(model_configs_dir: Path) -> dict[str, str]:
    scenario_map: dict[str, str] = {}
    if not model_configs_dir.exists():
        logger.warning("Model config directory not found: %s", model_configs_dir)
        return scenario_map

    for cfg_file in sorted(model_configs_dir.glob("*.yaml")):
        content = cfg_file.read_text(encoding="utf-8")
        model_key = _normalize_model_key(cfg_file.stem)
        for line in content.splitlines():
            if "carbon_scenario" not in line:
                continue
            if line.lstrip().startswith("#"):
                continue
            m = re.match(r"\s*carbon_scenario\s*:\s*(.+)\s*$", line)
            if not m:
                continue
            value = _strip_yaml_scalar(m.group(1))
            if not value or value.startswith("${"):
                continue
            scenario_map[model_key] = value
            break

    return scenario_map


def _resolve_model_name(output_jsonl_path: Path) -> str:
    metadata_path = output_jsonl_path.parent / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            llm = metadata.get("llm") or {}
            model = llm.get("model")
            if isinstance(model, str) and model.strip():
                return model.strip()
        except Exception as e:
            logger.warning(
                "Failed to parse metadata model from %s: %s", metadata_path, e
            )

    # Fallback: infer from eval folder naming convention.
    folder = output_jsonl_path.parent.name
    if "_sdk_" in folder:
        return folder.split("_sdk_", 1)[0]
    return folder


def _pick_scenario_for_model(
    model_name: str, scenario_map: dict[str, str]
) -> str | None:
    if not scenario_map:
        return None

    model_key = _normalize_model_key(model_name)
    if model_key in scenario_map:
        return scenario_map[model_key]

    # Fallback: contains-match on normalized keys.
    for key, value in scenario_map.items():
        if key and key in model_key:
            return value

    return None


def _resolve_scenario_path(
    scenarios_dir: Path, scenario_ref: str | None
) -> Path | None:
    if not scenario_ref:
        return None

    ref = scenario_ref.strip()
    if not ref:
        return None

    candidate = Path(ref).expanduser()
    if candidate.is_file():
        return candidate

    direct = scenarios_dir / ref
    if direct.is_file():
        return direct

    if not ref.endswith(".json"):
        with_ext = scenarios_dir / f"{ref}.json"
        if with_ext.is_file():
            return with_ext

    return None


def _extract_metrics_from_scenario(
    scenario_path: Path, js_bridge_path: Path
) -> dict[str, Any] | None:
    if not scenario_path.exists() or not js_bridge_path.exists():
        return None

    try:
        proc = subprocess.run(
            ["node", str(js_bridge_path), "--scenario", str(scenario_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(proc.stdout)
        return {
            "gpu_consumption_kwh": payload.get("gpu_consumption_kwh"),
            "server_consumption_kwh": payload.get("server_consumption_kwh"),
            "total_electricity_kwh": payload.get("total_electricity_kwh"),
            "co2e_operational_kg": payload.get("co2e_operational_kg"),
            "co2e_embodied_kg": payload.get("co2e_embodied_kg"),
            "co2e_total_kg": payload.get("co2e_total_kg"),
            "co2e_factor_per_kwh": payload.get("co2e_factor_per_kwh"),
            "scenario_file": scenario_path.name,
            "scenario_path": str(scenario_path),
            "electricity_grid_source": os.getenv("ELECTRICITY_GRID_SOURCE"),
        }
    except Exception as e:
        logger.warning(
            "Failed to compute scenario metrics from %s: %s", scenario_path, e
        )
        return None


def _extract_usage(metrics: dict[str, Any]) -> dict[str, int]:
    usage = metrics.get("accumulated_usage") or metrics.get("usage") or {}
    prompt_tokens = int(
        usage.get("prompt_tokens") or metrics.get("accumulated_prompt_tokens") or 0
    )
    completion_tokens = int(
        usage.get("completion_tokens")
        or metrics.get("accumulated_completion_tokens")
        or 0
    )
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _compute_generation_seconds(history: list[dict[str, Any]]) -> float:
    timestamps: list[datetime] = []
    for event in history:
        ts = event.get("timestamp")
        if not isinstance(ts, str) or not ts:
            continue
        try:
            timestamps.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
        except ValueError:
            continue

    if len(timestamps) < 2:
        return 0.0
    return round((max(timestamps) - min(timestamps)).total_seconds(), 3)


def _compute_token_footprint(
    completion_tokens: int, model_footprint: dict[str, Any] | None
) -> dict[str, Any]:
    if not model_footprint:
        return {
            "output_tokens_used": completion_tokens,
            "energy_consumption_kwh": None,
            "carbon_footprint_kg_co2e": None,
            "scenario_file": None,
            "scenario_path": None,
            "electricity_grid_source": None,
        }

    per_token_kwh = model_footprint.get("total_electricity_kwh")
    per_token_co2 = model_footprint.get("co2e_total_kg")
    return {
        "output_tokens_used": completion_tokens,
        "energy_consumption_kwh": per_token_kwh * completion_tokens
        if per_token_kwh is not None
        else None,
        "carbon_footprint_kg_co2e": per_token_co2 * completion_tokens
        if per_token_co2 is not None
        else None,
        "scenario_file": model_footprint.get("scenario_file"),
        "scenario_path": model_footprint.get("scenario_path"),
        "electricity_grid_source": model_footprint.get("electricity_grid_source"),
    }


def _read_jsonl(input_file: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def generate_instance_reports(
    input_file: str,
    report_dir: str | None = None,
    model_configs_dir: str | None = None,
    scenarios_dir: str | None = None,
    js_bridge_path: str | None = None,
) -> Path:
    """Generate per-instance and run-level report JSON files.

    The generated schema mirrors the SWE_bench_agent report style with
    token/time totals and optional energy/carbon estimates.
    """
    input_path = Path(input_file).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input output JSONL not found: {input_file}")

    reports_root = (
        Path(report_dir).resolve()
        if report_dir is not None
        else input_path.parent / "report"
    )
    reports_root.mkdir(parents=True, exist_ok=True)

    model_cfg_dir = (
        Path(model_configs_dir).resolve()
        if model_configs_dir is not None
        else _default_model_configs_dir()
    )
    scenario_dir = (
        Path(scenarios_dir).resolve()
        if scenarios_dir is not None
        else _default_scenarios_dir()
    )
    bridge = (
        Path(js_bridge_path).resolve()
        if js_bridge_path is not None
        else _default_js_bridge_path()
    )

    model_name = _resolve_model_name(input_path)
    scenario_map = _load_model_carbon_scenarios(model_cfg_dir)
    scenario_ref = _pick_scenario_for_model(model_name, scenario_map)
    scenario_path = _resolve_scenario_path(scenario_dir, scenario_ref)
    model_footprint = (
        _extract_metrics_from_scenario(scenario_path, bridge) if scenario_path else None
    )

    rows = _read_jsonl(input_path)

    totals: dict[str, Any] = {
        "instances": len(rows),
        "generation_seconds": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "energy_kwh": 0.0,
        "carbon_kg_co2e": 0.0,
    }

    instance_report_paths: list[str] = []
    for row in rows:
        instance_id = str(row.get("instance_id") or "unknown_instance")
        test_result = row.get("test_result") or {}
        metrics = row.get("metrics") or {}
        history = row.get("history") or []
        usage = _extract_usage(metrics)
        generation_seconds = _compute_generation_seconds(history)
        token_footprint = _compute_token_footprint(
            usage["completion_tokens"], model_footprint
        )

        totals["generation_seconds"] += generation_seconds
        totals["prompt_tokens"] += usage["prompt_tokens"]
        totals["completion_tokens"] += usage["completion_tokens"]
        totals["total_tokens"] += usage["total_tokens"]
        totals["energy_kwh"] += float(
            token_footprint.get("energy_consumption_kwh") or 0.0
        )
        totals["carbon_kg_co2e"] += float(
            token_footprint.get("carbon_footprint_kg_co2e") or 0.0
        )

        git_patch = str(test_result.get("git_patch") or "")
        status = (
            "error"
            if row.get("error")
            else ("no_patch" if not git_patch.strip() else "ok")
        )
        instance_report = {
            "instance_id": instance_id,
            "model": model_name,
            "status": status,
            "error": row.get("error"),
            "output": {
                "has_patch": bool(git_patch.strip()),
                "patch_chars": len(git_patch),
            },
            "generation_seconds": generation_seconds,
            "usage": usage,
            "token_footprint": token_footprint,
            "attempt": row.get("attempt"),
            "instruction": row.get("instruction"),
            "test_result": test_result,
        }

        instance_path = reports_root / f"report_{_slug(instance_id)}.json"
        _write_json(instance_path, instance_report)
        instance_report_paths.append(str(instance_path))

    summary = {
        "protocol": {
            "benchmark": "SWE-bench",
            "secondary_metrics": [
                "generation_seconds",
                "token_usage",
                "energy_kwh",
                "carbon_kg_co2e",
            ],
        },
        "model": model_name,
        "instances_total": totals["instances"],
        "totals": totals,
        "model_footprint": model_footprint,
        "artifacts": {
            "input_output_jsonl": str(input_path),
            "reports_dir": str(reports_root),
            "instance_reports": instance_report_paths,
        },
    }
    summary_path = reports_root / "report.json"
    _write_json(summary_path, summary)

    logger.info("Generated %d instance reports under %s", len(rows), reports_root)
    return summary_path

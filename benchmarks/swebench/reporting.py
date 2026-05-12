from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from openhands.sdk import get_logger


logger = get_logger(__name__)


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_model_configs_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "model_footprint"


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


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _load_model_footprint_files(model_configs_dir: Path) -> list[Path]:
    if not model_configs_dir.exists():
        logger.warning("Model footprint directory not found: %s", model_configs_dir)
        return []
    return sorted(model_configs_dir.glob("*.json"))


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


def _pick_footprint_for_model(
    model_name: str, footprint_files: list[Path]
) -> Path | None:
    if not footprint_files:
        return None

    model_key = _normalize_model_key(model_name)
    # Strip common provider prefixes (e.g. openai/, azure/, anthropic/) so
    # footprint filenames which are provider-agnostic (e.g. "claudeopus46-...")
    # match even when metadata contains a provider namespace like
    # "openai/claude-opus-4-6" or "azure_ai/claude-opus-4-6".
    for _pref in ("openai", "azureai", "azure", "anthropic", "openrouter"):
        if model_key.startswith(_pref) and len(model_key) > len(_pref):
            stripped = model_key[len(_pref) :]
            if stripped:
                model_key = stripped
                break
    aliases = {
        "gpt41": "gpt41",
        "gpt41mini": "gpt41mini",
        "gpt41nano": "gpt41nano",
        "gpt53codex": "gpt53",
        "mistralsmall2503": "mistralsmall3",
        "o1": "o1",
    }
    target_key = aliases.get(model_key, model_key)

    stems = [(path, _normalize_model_key(path.stem)) for path in footprint_files]

    for path, stem_key in stems:
        if stem_key == target_key:
            return path

    for path, stem_key in stems:
        if stem_key.startswith(target_key):
            return path

    for path, stem_key in stems:
        if target_key in stem_key or stem_key in target_key:
            return path

    return None


def _extract_metrics_from_scenario(
    scenario_path: Path, js_bridge_path: Path
) -> dict[str, Any] | None:
    if not scenario_path.exists():
        return None

    try:
        payload = json.loads(scenario_path.read_text(encoding="utf-8"))

        gpu_kwh = 0.0
        server_kwh = 0.0
        total_kwh = 0.0
        total_co2_kg = 0.0
        embodied_co2_kg = 0.0

        scopes = payload.get("scopes")
        if isinstance(scopes, list):
            for scope in scopes:
                if not isinstance(scope, dict):
                    continue
                components = scope.get("list")
                if not isinstance(components, list):
                    continue
                for component in components:
                    if not isinstance(component, dict):
                        continue
                    # Handle linked scenario entries which reference another
                    # scenario by id. This allows model footprint files to
                    # include `type: link` entries that point to a detailed
                    # scenario in the central scenarios directory.
                    if component.get("type") == "link" and component.get("scenario_id"):
                        linked_id = str(component.get("scenario_id"))
                        quantity_raw = component.get("quantity")
                        quantity = _coerce_float(quantity_raw) or 1.0
                        # Locate linked scenario JSON in default scenarios dir
                        linked_path = (
                            Path(__file__).resolve().parents[4]
                            / "AICarbonFootprintScenarios"
                            / "scenarios"
                        )
                        if not linked_path.exists():
                            linked_path = (
                                Path(__file__).resolve().parents[4]
                                / "carbon-footprint-modeling-tool"
                                / "scenarios"
                            )
                        linked_file = linked_path / f"{linked_id}.json"
                        submetrics = None
                        if linked_file.exists():
                            submetrics = _extract_metrics_from_scenario(
                                linked_file, js_bridge_path
                            )
                        else:
                            # Attempt to fetch linked scenario from the canonical
                            # GitHub repo if it's not present locally.
                            try:
                                import urllib.request

                                gh_url = (
                                    "https://raw.githubusercontent.com/borisruf/carbon-footprint-modeling-tool/main/scenarios/"
                                    + f"{linked_id}.json"
                                )
                                with urllib.request.urlopen(gh_url, timeout=5) as resp:
                                    if resp.status == 200:
                                        data = resp.read().decode("utf-8")
                                        # write to a temp file in-memory by using
                                        # the existing extractor via a patched
                                        # Path-like object isn't trivial, so
                                        # instead, compute metrics directly here
                                        # by invoking a small helper that mirrors
                                        # the logic in _extract_metrics_from_scenario.
                                        # For simplicity, write to a temp file on disk
                                        import tempfile

                                        tf = tempfile.NamedTemporaryFile(
                                            mode="w", delete=False, suffix=".json"
                                        )
                                        tf.write(data)
                                        tf.flush()
                                        tf.close()
                                        submetrics = _extract_metrics_from_scenario(
                                            Path(tf.name), js_bridge_path
                                        )
                            except Exception:
                                submetrics = None
                            if submetrics:
                                # Scale linked metrics by quantity (e.g., PUE)
                                total_kwh += (
                                    float(
                                        submetrics.get("total_electricity_kwh") or 0.0
                                    )
                                    * quantity
                                )
                                total_co2_kg += (
                                    float(submetrics.get("co2e_operational_kg") or 0.0)
                                    * quantity
                                )
                                gpu_kwh += (
                                    float(submetrics.get("gpu_consumption_kwh") or 0.0)
                                    * quantity
                                )
                                server_kwh += (
                                    float(
                                        submetrics.get("server_consumption_kwh") or 0.0
                                    )
                                    * quantity
                                )
                        continue

                    consumer = component.get("consumer") or {}
                    source = component.get("source") or {}
                    electricity_value = (
                        (consumer.get("consumptions") or {})
                        .get("electricity", {})
                        .get("value")
                    )
                    per_unit_kwh = _coerce_float(electricity_value)

                    quantity_raw = component.get("quantity")
                    quantity = _coerce_float(quantity_raw)
                    if quantity is None:
                        quantity = 1.0

                    # If electricity consumption is provided, compute kWh and
                    # operational CO2 (via co2 per kWh if available).
                    if per_unit_kwh is not None:
                        component_kwh = per_unit_kwh * quantity
                        total_kwh += component_kwh

                        co2_factor = (
                            (source.get("emissions") or {}).get("co2e", {}).get("value")
                        )
                        co2_per_kwh = _coerce_float(co2_factor)
                        if co2_per_kwh is None:
                            co2_per_kwh = 0.0

                        total_co2_kg += component_kwh * co2_per_kwh

                        consumer_name = str(consumer.get("name") or "").lower()
                        if "gpu" in consumer_name:
                            gpu_kwh += component_kwh
                        elif "server" in consumer_name:
                            server_kwh += component_kwh
                    else:
                        # No electricity specified. Try to read per-unit CO2
                        # (e.g. embodied emissions given as kg per token).
                        co2_entry = (source.get("emissions") or {}).get("co2e", {})
                        co2_per_unit = _coerce_float(co2_entry.get("value"))
                        base_unit = (co2_entry.get("base_unit") or "").lower()
                        if co2_per_unit is not None:
                            if base_unit in ("token", "tokens", "per_token"):
                                # Embodied / per-token emissions
                                embodied_co2_kg += co2_per_unit * quantity
                                total_co2_kg += co2_per_unit * quantity
                            elif base_unit in ("kwh", "per_kwh"):
                                # CO2 expressed per kWh but no electricity given
                                # can't convert, skip
                                pass
                            else:
                                # Unknown base unit: treat as per-unit CO2
                                embodied_co2_kg += co2_per_unit * quantity
                                total_co2_kg += co2_per_unit * quantity

        return {
            "gpu_consumption_kwh": gpu_kwh,
            "server_consumption_kwh": server_kwh,
            "total_electricity_kwh": total_kwh,
            "co2e_operational_kg": total_co2_kg,
            "co2e_embodied_kg": None,
            "co2e_total_kg": total_co2_kg,
            "co2e_factor_per_kwh": (total_co2_kg / total_kwh) if total_kwh else None,
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
    usage = (
        metrics.get("accumulated_usage")
        or metrics.get("accumulated_token_usage")
        or metrics.get("usage")
        or {}
    )
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


def _extract_iteration_token_usage(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    usages = metrics.get("token_usages") or []
    normalized: list[dict[str, Any]] = []
    for usage in usages:
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump(exclude_none=True)
        if isinstance(usage, dict):
            normalized.append(usage)
    return normalized


def _extract_timings(test_result: dict[str, Any]) -> dict[str, float | None]:
    timings = test_result.get("timings") or {}
    return {
        "workspace_generation_seconds": timings.get("workspace_generation_seconds"),
        "agent_generation_seconds": timings.get("agent_generation_seconds"),
        "total_generation_seconds": timings.get("total_generation_seconds"),
    }


def _extract_costs(
    metrics: dict[str, Any], test_result: dict[str, Any]
) -> dict[str, Any]:
    accumulated_cost = metrics.get("accumulated_cost")
    proxy_cost = test_result.get("proxy_cost")
    effective_cost = accumulated_cost if accumulated_cost is not None else proxy_cost

    return {
        "accumulated_cost_usd": float(accumulated_cost)
        if accumulated_cost is not None
        else None,
        "proxy_cost_usd": float(proxy_cost) if proxy_cost is not None else None,
        "effective_cost_usd": float(effective_cost)
        if effective_cost is not None
        else 0.0,
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
    bridge = (
        Path(js_bridge_path).resolve()
        if js_bridge_path is not None
        else _default_js_bridge_path()
    )

    model_name = _resolve_model_name(input_path)
    footprint_files = _load_model_footprint_files(model_cfg_dir)
    scenario_path = _pick_footprint_for_model(model_name, footprint_files)
    model_footprint = (
        _extract_metrics_from_scenario(scenario_path, bridge) if scenario_path else None
    )

    rows = _read_jsonl(input_path)

    totals: dict[str, Any] = {
        "instances": len(rows),
        "generation_seconds": 0.0,
        "workspace_generation_seconds": 0.0,
        "agent_generation_seconds": 0.0,
        "iterations": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "accumulated_cost_usd": 0.0,
        "proxy_cost_usd": 0.0,
        "effective_cost_usd": 0.0,
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
        iteration_token_usage = _extract_iteration_token_usage(metrics)
        timings = _extract_timings(test_result)
        costs = _extract_costs(metrics, test_result)
        generation_seconds = _compute_generation_seconds(history)
        iteration_count = int(
            test_result.get("iteration_count") or len(iteration_token_usage)
        )
        token_footprint = _compute_token_footprint(
            usage["completion_tokens"], model_footprint
        )

        totals["generation_seconds"] += generation_seconds
        totals["workspace_generation_seconds"] += float(
            timings.get("workspace_generation_seconds") or 0.0
        )
        totals["agent_generation_seconds"] += float(
            timings.get("agent_generation_seconds") or 0.0
        )
        totals["iterations"] += iteration_count
        totals["prompt_tokens"] += usage["prompt_tokens"]
        totals["completion_tokens"] += usage["completion_tokens"]
        totals["total_tokens"] += usage["total_tokens"]
        totals["accumulated_cost_usd"] += float(
            costs.get("accumulated_cost_usd") or 0.0
        )
        totals["proxy_cost_usd"] += float(costs.get("proxy_cost_usd") or 0.0)
        totals["effective_cost_usd"] += float(costs.get("effective_cost_usd") or 0.0)
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
            "cost": costs,
            "generation_seconds": generation_seconds,
            "timings": timings,
            "iterations": iteration_count,
            "iteration_token_usage": iteration_token_usage,
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
                "iterations",
                "workspace_generation_seconds",
                "agent_generation_seconds",
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

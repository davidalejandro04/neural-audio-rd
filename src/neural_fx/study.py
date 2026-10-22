"""Prospective matched studies and review of existing engineering evidence."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .models import build_model
from .utils import canonical_hash, read_json, sha256, write_json


def parameter_count(spec):
    # Planning must not change training's random state or allocate huge candidates.
    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        return sum(p.numel() for p in build_model(spec).parameters())


def matched_baseline(spec, tolerance=0.1):
    target = parameter_count({**spec, "name": "hybrid"})
    candidates = [{**spec, "name": "conditional_tcn", "channels": channels} for channels in range(1, max(128, spec["channels"] * 4) + 1)]
    baseline = min(candidates, key=lambda candidate: abs(parameter_count(candidate) - target))
    if abs(parameter_count(baseline) / target - 1) > tolerance:
        raise ValueError("Could not match the conditioned TCN within the parameter tolerance")
    return baseline


def review_run(root, output):
    root = Path(root)
    config = read_json(root / "preregistered_config.json")
    rows = []
    for folder in sorted((root / "models").iterdir()):
        if not (folder / "evaluation/metrics.json").exists():
            continue
        metrics = read_json(folder / "evaluation/metrics.json")
        gates = read_json(folder / "gates.json")
        summary = metrics["summary"]["model"]
        rows.append({"model": folder.name, "parameters": metrics["parameters"],
                     "active_source_median": summary["active_source_median"],
                     "by_regime_pair_median": summary["by_regime_pair_median"],
                     "failed_checks": {name: value for name, value in gates["checks"].items() if not value["passed"]},
                     "checkpoint_sha256": metrics["checkpoint_sha256"]})
    if not rows:
        raise ValueError("No evaluated model reports found")
    result = {"run": str(root), "seed": config["seed"], "effect": config["data"]["effect"],
              "config_sha256": canonical_hash(config), "models": rows,
              "phase_status": "single-seed evidence; P2 replication and P3 matched multi-family study remain",
              "interpretation": "Unconditioned models cannot distinguish identical dry input at different drive settings. Compare the hybrid to a conditioned baseline. Validation contains seen control settings; inspect held-out-control results separately."}
    write_json(output, result)
    return result


def plan_study(base_config, output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Study directory already exists: {output}")
    baseline = matched_baseline(base_config["model"])
    hybrid = {**base_config["model"], "name": "hybrid"}
    variants = {"conditional_tcn": {"channels": baseline["channels"]}}
    output.mkdir(parents=True)
    entries = []
    for effect in ("distortion", "cabinet"):
        for seed in (1337, 2027, 3407):
            name = f"{effect}-seed-{seed}"
            config = deepcopy(base_config)
            config.update({"experiment": f"p3_{name}", "seed": seed, "data_seed": 8181,
                           "suite": ["conditional_tcn", "hybrid_no_gru", "hybrid_no_fir", "hybrid"],
                           "model_variants": variants})
            config["data"]["effect"] = effect
            config["model"]["name"] = "hybrid"
            config_path = output / f"{name}.json"
            write_json(config_path, config)
            entries.append({"effect": effect, "seed": seed, "config": config_path.name,
                            "config_sha256": canonical_hash(config), "run": f"runs/{name}"})
    plan = {"schema_version": 1, "purpose": "prospective synthetic P3 study, not a reference-transfer or release test",
            "baseline": baseline, "candidate": hybrid,
            "parameters": {"conditional_tcn": parameter_count(baseline), "hybrid": parameter_count(hybrid)},
            "criteria": {"minimum_effect_families": 2, "minimum_seeds": 3, "minimum_test_sources": 20,
                         "parameter_tolerance": 0.1, "relative_improvement": 0.1,
                         "bootstrap_draws": 5000, "bootstrap_seed": 1871,
                         "held_out_controls_must_not_worsen": True},
            "data_seed": 8181, "entries": entries,
            "statistics": "Mean presets within source, mean across training seeds for each source, median across sources. Paired crossed bootstrap resamples seeds and sources independently; identical resamples for both systems. Percentile 95% intervals. Only three seeds: uncertainty estimates remain exploratory.",
            "limitations": ["synthetic targets only", "same fixed dataset across training seeds", "no long reverb or hardware inference", "ablations retain hybrid branch widths and are not independently parameter matched", "Python runtime screening is not native callback validation"]}
    write_json(output / "plan.json", plan)
    # Resolve script-relative paths so invocation works from any working directory.
    lines = ["$ErrorActionPreference = 'Stop'", "$PSNativeCommandUseErrorActionPreference = $false", "$studyRoot = $PSScriptRoot", "foreach ($entry in (Get-Content (Join-Path $studyRoot 'plan.json') -Raw | ConvertFrom-Json).entries) {",
             "    $configPath = Join-Path $studyRoot $entry.config",
             "    $runPath = Join-Path $studyRoot $entry.run",
             "    if (Test-Path $runPath) {",
             "        $statusPath = Join-Path $runPath 'status.json'",
             "        if (-not (Test-Path $statusPath)) { throw ('Existing run has no status: ' + $runPath) }",
             "        $existing = Get-Content $statusPath -Raw | ConvertFrom-Json",
             "        if ($existing.state -ne 'completed' -or $existing.config_sha256 -ne $entry.config_sha256) { throw ('Existing run is incomplete or differs from this plan: ' + $runPath) }",
             "        Write-Host ('Skipping completed run: ' + $runPath)", "        continue", "    }",
             "    python -m neural_fx run --config $configPath --output $runPath --suite --export",
             "    $runExit = $LASTEXITCODE",
             "    $status = Get-Content (Join-Path $runPath 'status.json') -Raw | ConvertFrom-Json",
             "    if ($status.state -ne 'completed') { throw ('Execution failed: ' + $runPath) }",
             "    Write-Host ($entry.effect + ' seed=' + $entry.seed + ' completed, exit=' + $runExit)", "}",
             "python -m neural_fx analyze-study --plan (Join-Path $studyRoot 'plan.json') --output (Join-Path $studyRoot 'analysis.json')"]
    (output / "run.ps1").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"plan": str(output / "plan.json"), "runner": str(output / "run.ps1"),
            "parameters": plan["parameters"], "training_runs": len(entries) * 4,
            "note": "Prepared only; training has not been launched."}


def source_values(rows, regime=None):
    selected = [r for r in rows if r["system"] == "model" and not r["bypass"]
                and (regime is None or r["regime"] == regime)]
    result = {}
    for row in selected:
        key = (row["source_id"], row["pair_id"])
        if key in result:
            raise ValueError("Duplicate model pair in evaluation")
        result[key] = row
    if not result:
        raise ValueError("No active examples for requested regime")
    for row in result.values():
        if any(row[key] is None or not np.isfinite(row[key]) or row[key] < 0 for key in ("esr", "mrstft")):
            raise ValueError("Paired analysis requires finite, nonnegative ESR and MR-STFT")
    return result


def paired_effect(seed_pairs, criteria, regime=None):
    if not seed_pairs:
        raise ValueError("At least one paired seed is required")
    arrays = {metric: {"baseline": [], "candidate": []} for metric in ("esr", "mrstft")}
    source_order = None
    for baseline_rows, candidate_rows in seed_pairs:
        a, b = source_values(baseline_rows, regime), source_values(candidate_rows, regime)
        if a.keys() != b.keys():
            raise ValueError("Baseline and candidate pair IDs/source IDs differ")
        if any(a[key]["regime"] != b[key]["regime"] for key in a):
            raise ValueError("Baseline and candidate control regimes differ")
        sources = sorted({key[0] for key in a})
        if source_order is not None and sources != source_order:
            raise ValueError("Study seeds must evaluate the same fixed test sources")
        source_order = sources
        for metric in arrays:
            for side, rows in (("baseline", a), ("candidate", b)):
                arrays[metric][side].append([np.mean([row[metric] for key, row in rows.items() if key[0] == source]) for source in sources])
    rng = np.random.default_rng(criteria["bootstrap_seed"])
    seeds, sources = len(seed_pairs), len(source_order)
    seed_draws = rng.integers(0, seeds, (criteria["bootstrap_draws"], seeds))
    source_draws = rng.integers(0, sources, (criteria["bootstrap_draws"], sources))
    output = {}
    for metric, sides in arrays.items():
        a, b = np.asarray(sides["baseline"]), np.asarray(sides["candidate"])
        reference = float(np.median(a.mean(axis=0)))
        candidate = float(np.median(b.mean(axis=0)))
        if reference <= 0:
            raise ValueError("Relative improvement is undefined for zero baseline error")
        # Limit peak memory: draw loop retains only one seed/source matrix.
        changes = np.array([np.median(a[sd][:, so].mean(axis=0)) - np.median(b[sd][:, so].mean(axis=0))
                            for sd, so in zip(seed_draws, source_draws)])
        interval = np.percentile(changes, [2.5, 97.5]).tolist()
        relative = (reference - candidate) / reference
        output[metric] = {"baseline": reference, "candidate": candidate,
                          "relative_improvement": relative, "absolute_improvement_95_ci": interval,
                          "improvement_supported": relative >= criteria["relative_improvement"] and interval[0] > 0}
    return {"sources": sources, "seeds": seeds, "metrics": output}


def analyze_study(plan_path, output):
    plan_path = Path(plan_path)
    plan = read_json(plan_path)
    grouped, missing = {}, []
    for entry in plan["entries"]:
        folder = plan_path.parent / entry["run"]
        expected = [folder / "status.json"] + [folder / "models" / name / "evaluation/per_pair.json" for name in ("conditional_tcn", "hybrid")]
        if any(not path.exists() for path in expected):
            missing.append(entry["run"])
            continue
        config = read_json(folder / "preregistered_config.json")
        if canonical_hash(config) != entry["config_sha256"]:
            raise ValueError(f"Run config differs from the study plan: {folder}")
        if read_json(folder / "status.json")["state"] != "completed":
            missing.append(entry["run"])
            continue
        grouped.setdefault(entry["effect"], []).append((entry, folder))
    result = {"state": "incomplete" if missing else "analyzed", "plan_sha256": sha256(plan_path),
              "missing_runs": missing, "effects": {}, "p3_accepted": False, "production_ready": False}
    if missing:
        write_json(output, result)
        return result
    criteria = plan["criteria"]
    for effect, entries in grouped.items():
        pairs, parameter_ok, runtime_ok, streaming_ok, quality_ok = [], True, True, True, True
        manifests = []
        for entry, folder in entries:
            evaluations = []
            for name in ("conditional_tcn", "hybrid"):
                root = folder / "models" / name
                metric = read_json(root / "evaluation/metrics.json")
                if metric["checkpoint_sha256"] != sha256(root / "best.pt"):
                    raise ValueError("Evaluation checkpoint checksum mismatch")
                gates = read_json(root / "gates.json")["checks"]
                runtime_ok &= gates.get("runtime_screening", {}).get("passed", False)
                streaming_ok &= gates["streaming"]["passed"]
                quality_ok &= gates["quality_esr"]["passed"] and gates["beats_identity"]["passed"]
                evaluations.append(metric)
            a, b = evaluations
            if a["manifest_sha256"] != b["manifest_sha256"]:
                raise ValueError("Paired models used different datasets")
            manifests.append(a["manifest_sha256"])
            parameter_ok &= abs(a["parameters"] / b["parameters"] - 1) <= criteria["parameter_tolerance"]
            pairs.append(tuple(read_json(folder / "models" / name / "evaluation/per_pair.json") for name in ("conditional_tcn", "hybrid")))
        if any(item != manifests[0] for item in manifests):
            raise ValueError("Training seeds used different fixed datasets")
        scores = paired_effect(pairs, criteria)
        held_out = paired_effect(pairs, criteria, "held_out_controls")
        supported = all(row["improvement_supported"] for row in scores["metrics"].values())
        nonworsening = all(row["relative_improvement"] >= 0 for row in held_out["metrics"].values())
        enough = len({entry["seed"] for entry, _ in entries}) >= criteria["minimum_seeds"] and scores["sources"] >= criteria["minimum_test_sources"]
        result["effects"][effect] = {"primary": scores, "held_out_controls": held_out,
                                       "parameter_match": bool(parameter_ok), "runtime_screening": bool(runtime_ok),
                                       "streaming": bool(streaming_ok), "engineering_quality": bool(quality_ok),
                                       "quality_hypothesis_supported": bool(supported and nonworsening and enough and parameter_ok),
                                       "accepted": bool(supported and nonworsening and enough and parameter_ok and runtime_ok and streaming_ok and quality_ok)}
    result["p3_accepted"] = sum(item["accepted"] for item in result["effects"].values()) >= criteria["minimum_effect_families"]
    result["interpretation"] = "Negative or inconclusive results remain valid. Synthetic acceptance does not establish hardware fidelity, native real-time behavior, or reference transfer. Bootstrap intervals are exploratory with three seeds."
    write_json(output, result)
    return result

"""An ordered pipeline whose reports distinguish execution from gate acceptance."""
from copy import deepcopy
from pathlib import Path

from .data import generate, verify
from .experiment import compare, evaluate, train
from .runtime import benchmark, export_onnx
from .utils import canonical_hash, read_json, write_json


def assess(config, metrics, runtime=None, export=None):
    gates = config["gates"]
    neural = metrics["summary"]["model"]["active_source_median"]["esr"]
    identity = metrics["summary"]["identity"]["active_source_median"]["esr"]
    checks = {
        "quality_esr": {"passed": neural is not None and neural <= gates["max_esr"], "observed": neural, "maximum": gates["max_esr"]},
        "beats_identity": {"passed": neural is not None and identity is not None and neural <= identity * gates["identity_esr_ratio"],
                           "observed": neural / identity if identity else None, "maximum": gates["identity_esr_ratio"]},
        "streaming": {"passed": metrics["max_stream_error"] <= gates["stream_max_error"], "observed": metrics["max_stream_error"], "maximum": gates["stream_max_error"]},
    }
    if runtime is not None:
        worst = max(block["p99_budget_fraction"] for block in runtime["blocks"])
        checks["runtime_screening"] = {"passed": worst <= gates["p99_budget_fraction"], "observed": worst, "maximum": gates["p99_budget_fraction"]}
    if export is not None:
        checks["onnx_parity"] = {"passed": export["passed"], "observed": export["max_error"], "maximum": export["tolerance"]}
    return {"passed": all(check["passed"] for check in checks.values()), "checks": checks,
            "production_ready": False,
            "remaining_release_requirements": ["real capture evaluation", "reference transfer study", "native runtime parity", "listening study", "VST3 validator and host tests"]}


def run(config, output, suite=False, with_export=False):
    output = Path(output)
    # Refuse silent reuse/overwrite of any experiment evidence.
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output already exists: {output}; choose a new directory")
    output.mkdir(parents=True)
    write_json(output / "preregistered_config.json", config)
    root = output / "dataset"
    write_json(output / "status.json", {"state": "running", "stage": "generate", "completed_models": []})
    try:
        audit = generate(config, root)
    except Exception as error:
        write_json(output / "status.json", {"state": "failed", "stage": "generate", "error": str(error)})
        raise
    write_json(output / "data_audit.json", audit)
    models = config["suite"] if suite else [config["model"]["name"]]
    results = []
    write_json(output / "status.json", {"state": "running", "completed_models": []})
    for name in models:
        selected = deepcopy(config)
        selected["model"]["name"] = name
        folder = output / "models" / name
        try:
            training = train(selected, root, folder)
            metrics = evaluate(folder / "best.pt", root, folder / "evaluation")
            runtime = benchmark(folder / "best.pt", folder / "runtime.json", config["evaluation"]["block_sizes"])
            exported = None
            if with_export and name not in ("gru", "hybrid", "hybrid_no_fir"):
                exported = export_onnx(folder / "best.pt", folder / "model.onnx")
            gates = assess(selected, metrics, runtime, exported)
            write_json(folder / "gates.json", gates)
            results.append({"model": name, "training": training, "gates_passed": gates["passed"]})
            print(f"{name}: validation ESR={training['best_valid_esr']:.4f}, gates={'PASS' if gates['passed'] else 'FAIL'}", flush=True)
            write_json(output / "status.json", {"state": "running", "completed_models": results})
        except Exception as error:
            write_json(output / "status.json", {"state": "failed", "failed_model": name,
                                                "error": str(error), "completed_models": results})
            raise
    comparison = compare(output / "models", output)
    status = {"state": "completed", "config_sha256": canonical_hash(config), "completed_models": results,
              "all_engineering_gates_passed": all(row["gates_passed"] for row in results),
              "research_claim": "not tested; smoke experiments do not establish hybrid superiority",
              "comparison": comparison}
    write_json(output / "status.json", status)
    return status

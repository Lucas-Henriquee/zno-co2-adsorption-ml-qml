"""Route explicitly selected configurations through the existing stage launchers."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[2]


def resolve(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-results", type=Path, help="Reuse explicitly selected scientific results.")
    source.add_argument("--simulate", action="store_true", help="Include new simulations with the supplied YAML.")
    parser.add_argument("--simulation-config", type=Path)
    for stage in ["dataset", "model", "analysis"]:
        parser.add_argument(f"--{stage}-config", type=Path, required=True)
    parser.add_argument("--model-dataset", type=Path, required=True,
                        help="Dataset path relative to this run's generated dataset directory.")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true", help="Execute the printed plan. Otherwise only display it.")
    action.add_argument("--dry-run", action="store_true", help="Display the plan without executing any stage (default).")
    return parser.parse_args(argv)


def build_plan(args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        raise ValueError("Invalid run ID.")
    if args.simulate and args.simulation_config is None:
        raise ValueError("--simulate requires --simulation-config; no experiment is chosen implicitly.")
    if not args.simulate and args.simulation_config is not None:
        raise ValueError("--simulation-config applies only to --simulate.")
    configs, outputs = {}, {}
    for stage in (["simulation"] if args.simulate else []) + ["dataset", "model", "analysis"]:
        path = resolve(getattr(args, f"{stage}_config"))
        config = yaml.safe_load(path.read_text())
        if not isinstance(config, dict) or config.get("schema_version") != 1:
            raise ValueError(f"Invalid configuration: {path}")
        configs[stage] = path
        outputs[stage] = resolve(config["output"]["root"]) / args.run_id
        area = ROOT / ("data/generated" if stage == "dataset" else "results")
        if not outputs[stage].resolve().is_relative_to(area):
            raise ValueError(f"Output outside the allowed project area: {outputs[stage]}")
    relative = args.model_dataset
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise ValueError("--model-dataset must be a file path within the generated dataset directory.")
    science = outputs["simulation"] if args.simulate else resolve(args.source_results)
    if not args.simulate and not science.is_dir():
        raise FileNotFoundError(f"Scientific results directory does not exist: {science}")
    science_analysis = outputs["analysis"].with_name(args.run_id + "_science")
    model_analysis = outputs["analysis"].with_name(args.run_id + "_models")
    commands = []
    if args.simulate:
        commands.append(("simulations", [str(ROOT / "run_simulations.sh"), "all", "--config",
                                         str(configs["simulation"]), "--run-id", args.run_id]))
    commands.extend([
        ("science", [str(ROOT / "run_analysis.sh"), "science", "--config", str(configs["analysis"]),
                     "--scientific-results", str(science), "--run-id", science_analysis.name]),
        ("datasets", [str(ROOT / "run_datasets.sh"), "all", "--config", str(configs["dataset"]),
                      "--source-results", str(science), "--run-id", args.run_id]),
        ("models", [str(ROOT / "run_models.sh"), "local", "--config", str(configs["model"]),
                    "--dataset", str(outputs["dataset"] / relative), "--run-id", args.run_id]),
        ("model_analysis", [str(ROOT / "run_analysis.sh"), "models", "--config", str(configs["analysis"]),
                            "--model-run", str(outputs["model"]), "--run-id", model_analysis.name]),
    ])
    destinations = [outputs["dataset"], outputs["model"], science_analysis, model_analysis]
    if args.simulate:
        destinations.append(science)
    if len(set(destinations)) != len(destinations):
        raise ValueError("Stage output directories must be distinct.")
    for destination in destinations:
        if destination.exists():
            raise FileExistsError(f"Choose a new run ID; output already exists: {destination}")
    return commands, configs


def execute_plan(commands, configs, run_id):
    directory = ROOT / "results/workflows" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    state = {"run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
             "status": "running", "configs": {}, "stages": {}}
    for name, path in configs.items():
        content = path.read_bytes()
        state["configs"][name] = {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}

    def save():
        (directory / "run_manifest.json").write_text(json.dumps(state, indent=2) + "\n")

    try:
        for name, command in commands:
            for item in state["configs"].values():
                if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
                    raise RuntimeError("A selected configuration changed during execution; stopping.")
            record = {"status": "running", "command": command}
            state["stages"][name] = record
            save()
            print(f"[start] {name}; log: {directory / (name + '.log')}", flush=True)
            with (directory / (name + ".log")).open("x") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            record.update(exit_code=result.returncode, status="completed" if result.returncode == 0 else "failed")
            save()
            result.check_returncode()
        state["status"] = "completed"
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()


def main():
    args = parse_args()
    commands, configs = build_plan(args)
    for name, command in commands:
        print(f"[{name}] {shlex.join(command)}")
    print("Parameters and seeds come unchanged from the selected configs. IBM submission is excluded.")
    if not args.execute:
        print("Plan only. Future generated inputs have not been validated. Add --execute to run.")
        return
    execute_plan(commands, configs, args.run_id)


if __name__ == "__main__":
    main()

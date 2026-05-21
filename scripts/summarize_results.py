from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


METRIC_FILES = (
    "metrics.csv",
    "metrics_adam_lora_spsa.csv",
    "metrics_agentwise_spsa.csv",
)


def to_float(value):
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def read_rows(run_dir: Path):
    for name in METRIC_FILES:
        path = run_dir / name
        if path.exists():
            with path.open(newline="", encoding="utf-8", errors="replace") as f:
                return name, list(csv.DictReader(f))
    return None, []


def read_meta(run_dir: Path):
    for path in sorted(run_dir.glob("meta*.json")):
        try:
            return json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            pass
    return {}


def classify_run(name: str):
    if "adam_no_consensus" in name:
        return "adam_no_consensus"
    if "adam_confidence_consensus" in name:
        return "adam_consensus"
    if "adam_lora" in name:
        return "adam_pretrain_lora_spsa"
    if "agentwise" in name:
        return "sudoku9_agentwise_spsa"
    if "param_consensus" in name:
        return "parameter_consensus_spsa"
    if "single_spsa_boardacc_warmup" in name:
        return "single_spsa_warmup_then_board_acc"
    if "single_spsa_boardacc" in name:
        return "single_spsa_board_acc"
    if "single_spsa_boardcell" in name:
        return "single_spsa_board_cell"
    if "single_spsa_quality" in name:
        return "single_spsa_quality"
    return "other"


def summarize_run(run_dir: Path):
    metric_file, rows = read_rows(run_dir)
    meta = read_meta(run_dir)
    if not rows:
        return None

    def best_by(primary, secondary="cell_acc"):
        valid = []
        for idx, row in enumerate(rows):
            p = to_float(row.get(primary))
            if p is None:
                continue
            s = to_float(row.get(secondary)) or 0.0
            loss = to_float(row.get("loss"))
            valid.append((p, s, -(loss if loss is not None else 1e9), idx, row))
        return max(valid, default=(None, None, None, -1, rows[-1]))[-1]

    last = rows[-1]
    best_board = best_by("board_acc")
    best_cell = best_by("cell_acc", "board_acc")
    best_missing = best_by("cell_acc_missing", "board_acc")

    result = {
        "run": run_dir.name,
        "family": classify_run(run_dir.name),
        "metric_file": metric_file,
        "rows": len(rows),
        "task": meta.get("task", infer_task(run_dir.name)),
        "mode": meta.get("mode", ""),
        "agents": meta.get("agents", meta.get("num_agents", "")),
        "consensus": meta.get("consensus", ""),
        "hidden": meta.get("hidden", ""),
        "time_steps": meta.get("time_steps", ""),
        "params": meta.get("params", ""),
        "last_step": last.get("step", ""),
        "last_phase": last.get("phase", ""),
        "last_loss": last.get("loss", ""),
        "last_cell_acc": last.get("cell_acc", last.get("cell_acc_missing", "")),
        "last_board_acc": last.get("board_acc", ""),
        "best_board_step": best_board.get("step", ""),
        "best_board_acc": best_board.get("board_acc", ""),
        "best_board_cell_acc": best_board.get("cell_acc", best_board.get("cell_acc_missing", "")),
        "best_cell_step": best_cell.get("step", ""),
        "best_cell_acc": best_cell.get("cell_acc", ""),
        "best_missing_step": best_missing.get("step", ""),
        "best_cell_acc_missing": best_missing.get("cell_acc_missing", ""),
    }
    return result


def infer_task(name: str):
    for task in ("maze12", "sudoku4", "nonogram5"):
        if task in name:
            return task
    if "50k" in name or "sudoku9" in name or "agentwise" in name:
        return "sudoku9"
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/summary.csv"))
    args = parser.parse_args()

    rows = []
    for run_dir in sorted(args.raw_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        summary = summarize_run(run_dir)
        if summary:
            rows.append(summary)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run", "family", "task", "mode", "agents", "consensus", "hidden", "time_steps", "params",
        "metric_file", "rows", "last_step", "last_phase", "last_loss", "last_cell_acc", "last_board_acc",
        "best_board_step", "best_board_acc", "best_board_cell_acc",
        "best_cell_step", "best_cell_acc", "best_missing_step", "best_cell_acc_missing",
    ]
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} summaries to {args.out}")


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from task_bench_agents_spsa import (
    TinyConsensusNet,
    make_batch,
    task_spec,
    evaluate,
    count_params,
    objective_components_from_logits,
    compose_objective,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_row(path, row):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def flat_params(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


@torch.no_grad()
def add_flat_(model, direction, scale):
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.add_(direction[offset:offset + n].view_as(p), alpha=scale)
        offset += n


@torch.no_grad()
def set_flat_(model, theta):
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.copy_(theta[offset:offset + n].view_as(p))
        offset += n


@torch.no_grad()
def objective(model, task, batch_size, device, objective_name="ce", ce_weight=1.0, prob_weight=1.0, margin_weight=0.1, entropy_weight=0.0, board_weight=1.0, margin_tau=0.5):
    x, y, mask = make_batch(task, batch_size, device)
    return objective_on_batch(model, x, y, mask, objective_name, ce_weight, prob_weight, margin_weight, entropy_weight, board_weight, margin_tau)


@torch.no_grad()
def objective_on_batch(model, x, y, mask, objective_name="ce", ce_weight=1.0, prob_weight=1.0, margin_weight=0.1, entropy_weight=0.0, board_weight=1.0, margin_tau=0.5):
    model.eval()
    logits = model(x)[0]
    comps = objective_components_from_logits(logits, y, mask, margin_tau)
    value = compose_objective(comps, objective_name, ce_weight, prob_weight, margin_weight, entropy_weight, board_weight)
    stats = {k: float(v.detach().cpu()) for k, v in comps.items()}
    stats["objective"] = float(value.detach().cpu())
    return stats["objective"], stats


@torch.no_grad()
def param_stats(theta):
    return {
        "theta_l2": float(theta.float().norm().detach().cpu()),
        "theta_max_abs": float(theta.float().abs().max().detach().cpu()),
        "theta_finite": bool(torch.isfinite(theta).all().item()),
    }


@torch.no_grad()
def directional_derivative(model, task, batch_size, device, delta, beta, estimator, objective_name, ce_weight, prob_weight, margin_weight, entropy_weight, board_weight, margin_tau):
    x, y, mask = make_batch(task, batch_size, device)
    if estimator == "classic":
        add_flat_(model, delta, +beta)
        y_plus, plus_stats = objective_on_batch(model, x, y, mask, objective_name, ce_weight, prob_weight, margin_weight, entropy_weight, board_weight, margin_tau)
        add_flat_(model, delta, -2 * beta)
        y_minus, minus_stats = objective_on_batch(model, x, y, mask, objective_name, ce_weight, prob_weight, margin_weight, entropy_weight, board_weight, margin_tau)
        add_flat_(model, delta, +beta)
        deriv = ((y_plus - y_minus) / (2 * beta)) if math.isfinite(y_plus) and math.isfinite(y_minus) else 0.0
        return deriv, {"j_plus": y_plus, "j_minus": y_minus, **{f"plus_{k}": v for k, v in plus_stats.items()}, **{f"minus_{k}": v for k, v in minus_stats.items()}}

    if estimator != "six_point":
        raise ValueError(f"unknown SPSA estimator: {estimator}")

    weighted = 0.0
    finite = True
    stats = {}
    weights = {1: 1.0 / 14.0, 2: 4.0 / 14.0, 3: 9.0 / 14.0}
    current_scale = 0.0
    for radius, weight in weights.items():
        target_scale = radius * beta
        add_flat_(model, delta, target_scale - current_scale)
        current_scale = target_scale
        y_plus, plus_stats = objective_on_batch(model, x, y, mask, objective_name, ce_weight, prob_weight, margin_weight, entropy_weight, board_weight, margin_tau)

        target_scale = -radius * beta
        add_flat_(model, delta, target_scale - current_scale)
        current_scale = target_scale
        y_minus, minus_stats = objective_on_batch(model, x, y, mask, objective_name, ce_weight, prob_weight, margin_weight, entropy_weight, board_weight, margin_tau)

        stats[f"j_plus_{radius}"] = y_plus
        stats[f"j_minus_{radius}"] = y_minus
        for k, v in plus_stats.items():
            stats[f"plus{radius}_{k}"] = v
        for k, v in minus_stats.items():
            stats[f"minus{radius}_{k}"] = v
        if math.isfinite(y_plus) and math.isfinite(y_minus):
            weighted += weight * ((y_plus - y_minus) / (2 * radius * beta))
        else:
            finite = False

    add_flat_(model, delta, -current_scale)
    if not finite:
        weighted = 0.0
    stats["j_plus"] = stats["j_plus_1"]
    stats["j_minus"] = stats["j_minus_1"]
    return weighted, stats


def run(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = task_spec(args.task)
    models = [
        TinyConsensusNet(**spec, hidden=args.hidden, agents=args.agents, time_steps=args.time_steps, consensus=args.consensus).to(device)
        for _ in range(args.population)
    ]
    # Start near each other, but not exactly identical.
    base = flat_params(models[0])
    for m in models[1:]:
        theta = base + args.init_noise * torch.randn_like(base)
        set_flat_(m, theta)

    meta = vars(args) | {"device": str(device), "params_per_model": count_params(models[0])}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    csv_path = out_dir / "metrics.csv"
    start = time.time()

    n = flat_params(models[0]).numel()
    delta_scale = 1.0 / math.sqrt(3.0 * n)

    for step in range(1, args.steps + 1):
        thetas = [flat_params(m) for m in models]
        theta_mean = torch.stack(thetas).mean(dim=0)
        rows_for_step = []
        for i, model in enumerate(models):
            delta = torch.empty(n, device=device).bernoulli_(0.5).mul_(2).sub_(1).mul_(delta_scale)
            beta = args.beta / (step ** args.beta_decay)
            alpha = args.alpha / (step ** args.alpha_decay)

            deriv, spsa_stats = directional_derivative(
                model, args.task, args.batch_size, device, delta, beta, args.spsa_estimator,
                args.objective, args.ce_weight, args.prob_weight, args.margin_weight, args.entropy_weight,
                args.board_weight, args.margin_tau,
            )

            theta_i = flat_params(model)
            consensus_vec = theta_mean - theta_i
            if math.isfinite(deriv):
                # Formula from slide: theta <- theta - alpha * (Delta * (y+ - y-) / 2 beta + omega * sum(theta_j - theta_i)).
                # Here J = -loss is maximized, so we ascend SPSA and add consensus attraction.
                spsa_dir = delta * deriv
                update = spsa_dir + args.omega * consensus_vec
                norm = update.float().norm().clamp_min(1e-12)
                if args.max_update_norm > 0 and norm > args.max_update_norm:
                    update = update * (args.max_update_norm / norm)
                set_flat_(model, (theta_i + alpha * update).clamp(-args.param_clip, args.param_clip))
                update_norm = float((alpha * update).float().norm().detach().cpu())
            else:
                update_norm = 0.0
            if i == 0:
                rows_for_step.append({**spsa_stats, "update_norm": update_norm})

        if step == 1 or step % args.eval_every == 0:
            metrics = []
            for i, model in enumerate(models):
                m = evaluate(
                    model, args.task, args.eval_batches, args.eval_batch_size, device,
                    args.objective, args.ce_weight, args.prob_weight, args.margin_weight, args.entropy_weight,
                    args.board_weight, args.margin_tau,
                )
                theta = flat_params(model)
                metrics.append((i, m, param_stats(theta)))
            best = max(metrics, key=lambda x: x[1]["board_acc"] * 1000 + x[1]["cell_acc"])
            avg_cell = float(np.mean([x[1]["cell_acc"] for x in metrics]))
            avg_board = float(np.mean([x[1]["board_acc"] for x in metrics]))
            spread = float(torch.stack([flat_params(m) for m in models]).std(dim=0).mean().detach().cpu())
            row = {
                "step": step,
                "elapsed_sec": round(time.time() - start, 3),
                "best_agent": best[0],
                "best_loss": best[1]["loss"],
                "best_cell_acc": best[1]["cell_acc"],
                "best_board_acc": best[1]["board_acc"],
                "avg_cell_acc": avg_cell,
                "avg_board_acc": avg_board,
                "theta_spread": spread,
                **rows_for_step[0],
                **best[2],
            }
            write_row(csv_path, row)
            print(json.dumps(row), flush=True)
        if step % args.ckpt_every == 0:
            torch.save({"models": [m.state_dict() for m in models], "step": step, "meta": meta}, out_dir / f"ckpt_{step}.pt")
    torch.save({"models": [m.state_dict() for m in models], "step": args.steps, "meta": meta}, out_dir / "final.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["sudoku4", "maze12", "nonogram5"], required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--population", type=int, default=4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--agents", type=int, default=2)
    p.add_argument("--time-steps", type=int, default=8)
    p.add_argument("--consensus", choices=["mean", "confidence", "last", "residual_mean"], default="mean")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--ckpt-every", type=int, default=5000)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--omega", type=float, default=0.05)
    p.add_argument("--spsa-estimator", choices=["classic", "six_point"], default="classic")
    p.add_argument("--objective", choices=["ce", "prob", "margin", "soft_acc", "board_acc", "board_cell", "hybrid", "quality"], default="ce")
    p.add_argument("--ce-weight", type=float, default=0.1)
    p.add_argument("--prob-weight", type=float, default=1.0)
    p.add_argument("--margin-weight", type=float, default=0.1)
    p.add_argument("--entropy-weight", type=float, default=0.0)
    p.add_argument("--board-weight", type=float, default=1.0)
    p.add_argument("--margin-tau", type=float, default=0.5)
    p.add_argument("--alpha-decay", type=float, default=0.602)
    p.add_argument("--beta-decay", type=float, default=0.101)
    p.add_argument("--max-update-norm", type=float, default=0.05)
    p.add_argument("--param-clip", type=float, default=3.0)
    p.add_argument("--init-noise", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()

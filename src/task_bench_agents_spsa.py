import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def base_sudoku4():
    return np.array([[(r * 2 + r // 2 + c) % 4 for c in range(4)] for r in range(4)], dtype=np.int64)


def sample_sudoku4(batch, givens, device):
    xs, ys, masks = [], [], []
    for _ in range(batch):
        rng = np.random.default_rng(random.randrange(1 << 30))
        g = base_sudoku4()
        perm = rng.permutation(4)
        g = perm[g]
        rows = []
        for band in rng.permutation(2):
            rows.extend((band * 2 + rng.permutation(2)).tolist())
        cols = []
        for stack in rng.permutation(2):
            cols.extend((stack * 2 + rng.permutation(2)).tolist())
        g = g[rows, :][:, cols].reshape(-1)
        keep = rng.choice(16, size=givens, replace=False)
        x = np.zeros(16, dtype=np.int64)
        x[keep] = g[keep] + 1
        mask = np.ones(16, dtype=np.float32)
        mask[keep] = 0.0
        xs.append(x); ys.append(g); masks.append(mask)
    return (
        torch.tensor(np.stack(xs), dtype=torch.long, device=device),
        torch.tensor(np.stack(ys), dtype=torch.long, device=device),
        torch.tensor(np.stack(masks), dtype=torch.float32, device=device),
    )


def sample_maze(batch, size, wall_p, device):
    xs, ys, masks = [], [], []
    for _ in range(batch):
        rng = np.random.default_rng(random.randrange(1 << 30))
        wall = rng.random((size, size)) < wall_p
        free = np.argwhere(~wall)
        if len(free) < 2:
            wall[:] = False
            free = np.argwhere(~wall)
        sidx, gidx = rng.choice(len(free), size=2, replace=False)
        start = tuple(free[sidx]); goal = tuple(free[gidx])
        reach = np.zeros((size, size), dtype=np.int64)
        q = [start]; reach[start] = 1
        for r, c in q:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < size and 0 <= cc < size and not wall[rr, cc] and not reach[rr, cc]:
                    reach[rr, cc] = 1
                    q.append((rr, cc))
        x = np.zeros((size, size, 3), dtype=np.float32)
        x[..., 0] = wall.astype(np.float32)
        x[start[0], start[1], 1] = 1.0
        x[goal[0], goal[1], 2] = 1.0
        xs.append(x.reshape(size * size, 3))
        ys.append(reach.reshape(-1))
        masks.append(np.ones(size * size, dtype=np.float32))
    return (
        torch.tensor(np.stack(xs), dtype=torch.float32, device=device),
        torch.tensor(np.stack(ys), dtype=torch.long, device=device),
        torch.tensor(np.stack(masks), dtype=torch.float32, device=device),
    )


def clues_for_line(bits):
    out, run = [], 0
    for b in bits:
        if b:
            run += 1
        elif run:
            out.append(run); run = 0
    if run:
        out.append(run)
    return out or [0]


def sample_nonogram5(batch, device):
    xs, ys, masks = [], [], []
    for _ in range(batch):
        rng = np.random.default_rng(random.randrange(1 << 30))
        grid = (rng.random((5, 5)) < 0.45).astype(np.int64)
        row_clues = [clues_for_line(grid[r])[:3] for r in range(5)]
        col_clues = [clues_for_line(grid[:, c])[:3] for c in range(5)]
        row_arr = np.zeros((5, 3), dtype=np.float32)
        col_arr = np.zeros((5, 3), dtype=np.float32)
        for i, cl in enumerate(row_clues):
            row_arr[i, :len(cl)] = cl
        for i, cl in enumerate(col_clues):
            col_arr[i, :len(cl)] = cl
        feats = []
        for r in range(5):
            for c in range(5):
                feats.append(np.concatenate([row_arr[r] / 5.0, col_arr[c] / 5.0]))
        xs.append(np.stack(feats))
        ys.append(grid.reshape(-1))
        masks.append(np.ones(25, dtype=np.float32))
    return (
        torch.tensor(np.stack(xs), dtype=torch.float32, device=device),
        torch.tensor(np.stack(ys), dtype=torch.long, device=device),
        torch.tensor(np.stack(masks), dtype=torch.float32, device=device),
    )


class AgentBlock(nn.Module):
    def __init__(self, hidden, out_dim):
        super().__init__()
        in_dim = hidden * 4 + out_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.state = nn.Linear(hidden, hidden)
        self.proposal = nn.Linear(hidden, out_dim)
        self.conf = nn.Linear(hidden, 1)
        rank = 8
        self.state_down = nn.Linear(hidden, rank, bias=False)
        self.state_up = nn.Linear(rank, hidden, bias=False)
        self.proposal_down = nn.Linear(hidden, rank, bias=False)
        self.proposal_up = nn.Linear(rank, out_dim, bias=False)
        self.conf_down = nn.Linear(hidden, rank, bias=False)
        self.conf_up = nn.Linear(rank, 1, bias=False)
        for up in [self.state_up, self.proposal_up, self.conf_up]:
            nn.init.zeros_(up.weight)

    def forward(self, state, logits, row_ctx, col_ctx, global_ctx):
        h = torch.cat([state, logits, row_ctx, col_ctx, global_ctx], dim=-1)
        h = self.net(h)
        state_delta = self.state(h) + self.state_up(self.state_down(h)) / 8.0
        prop = self.proposal(h) + self.proposal_up(self.proposal_down(h)) / 8.0
        conf = self.conf(h) + self.conf_up(self.conf_down(h)) / 8.0
        return torch.tanh(state_delta + state), prop, conf


class TinyConsensusNet(nn.Module):
    def __init__(self, input_dim, out_dim, grid_h, grid_w, hidden=64, agents=2, time_steps=8, consensus="confidence"):
        super().__init__()
        self.grid_h, self.grid_w = grid_h, grid_w
        self.out_dim = out_dim
        self.hidden = hidden
        self.agents_n = agents
        self.time_steps = time_steps
        self.consensus = consensus
        self.inp = nn.Linear(input_dim, hidden)
        self.pos = nn.Parameter(torch.randn(grid_h * grid_w, hidden) * 0.02)
        self.agents = nn.ModuleList([AgentBlock(hidden, out_dim) for _ in range(agents)])

    def contexts(self, state):
        b, n, d = state.shape
        h, w = self.grid_h, self.grid_w
        s = state.reshape(b, h, w, d)
        row = s.mean(dim=2, keepdim=True).expand(-1, -1, w, -1).reshape(b, n, d)
        col = s.mean(dim=1, keepdim=True).expand(-1, h, -1, -1).reshape(b, n, d)
        glob = state.mean(dim=1, keepdim=True).expand(-1, n, -1)
        return row, col, glob

    def forward(self, x):
        state0 = torch.tanh(self.inp(x) + self.pos.unsqueeze(0))
        states = [state0 for _ in range(self.agents_n)]
        logits = torch.zeros(x.shape[0], x.shape[1], self.out_dim, device=x.device)
        props_all = []
        for _ in range(self.time_steps):
            props, confs, new_states = [], [], []
            for i, agent in enumerate(self.agents):
                row, col, glob = self.contexts(states[i])
                s, p, c = agent(states[i], logits, row, col, glob)
                new_states.append(s); props.append(p); confs.append(c)
            prop = torch.stack(props, dim=1)
            conf = torch.stack(confs, dim=1)
            if self.consensus == "mean":
                logits = prop.mean(dim=1)
            elif self.consensus == "last":
                logits = prop[:, -1]
            elif self.consensus == "residual_mean":
                logits = logits + 0.5 * prop.mean(dim=1)
            else:
                logits = (torch.softmax(conf, dim=1) * prop).sum(dim=1)
            states = new_states
            props_all.append(prop)
        return logits, props_all


def make_batch(task, batch, device):
    if task == "sudoku4":
        x, y, mask = sample_sudoku4(batch, 8, device)
        x = F.one_hot(x, 5).float()
        return x, y, mask
    if task == "maze12":
        return sample_maze(batch, 12, 0.28, device)
    if task == "nonogram5":
        return sample_nonogram5(batch, device)
    raise ValueError(task)


def task_spec(task):
    if task == "sudoku4":
        return dict(input_dim=5, out_dim=4, grid_h=4, grid_w=4)
    if task == "maze12":
        return dict(input_dim=3, out_dim=2, grid_h=12, grid_w=12)
    if task == "nonogram5":
        return dict(input_dim=6, out_dim=2, grid_h=5, grid_w=5)
    raise ValueError(task)


def loss_fn(logits, y, mask):
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none")
    m = mask.reshape(-1)
    return (ce * m).sum() / m.sum().clamp_min(1.0)


def objective_components_from_logits(logits, y, mask, margin_tau=0.5):
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_y = y.reshape(-1)
    m = mask.reshape(-1).float()
    denom = m.sum().clamp_min(1.0)

    ce = F.cross_entropy(flat_logits, flat_y, reduction="none")
    probs = F.softmax(flat_logits, dim=-1)
    true_prob = probs.gather(1, flat_y[:, None]).squeeze(1)

    true_logits = flat_logits.gather(1, flat_y[:, None]).squeeze(1)
    other_logits = flat_logits.masked_fill(
        F.one_hot(flat_y, flat_logits.shape[-1]).bool(),
        -torch.finfo(flat_logits.dtype).max,
    )
    margin = true_logits - other_logits.max(dim=-1).values
    entropy = -(probs.clamp_min(1e-8).log() * probs).sum(dim=-1)
    soft_correct = torch.sigmoid(margin / margin_tau)
    soft_correct_grid = soft_correct.reshape_as(mask).clamp(1e-6, 1.0)
    mask_grid = mask.float()
    board_log_soft = (soft_correct_grid.log() * mask_grid).sum(dim=1)
    board_soft = board_log_soft.exp()
    pred = logits.argmax(dim=-1)
    correct = ((pred == y).float() * mask_grid)
    hard_cell_acc = correct.sum() / denom
    hard_board_acc = (((pred == y) | (mask == 0)).all(dim=1).float()).mean()

    return {
        "obj_ce": (ce * m).sum() / denom,
        "obj_prob_true": (true_prob * m).sum() / denom,
        "obj_margin": (margin.tanh() * m).sum() / denom,
        "obj_soft_cell_acc": (soft_correct * m).sum() / denom,
        "obj_soft_board_acc": board_soft.mean(),
        "obj_hard_cell_acc": hard_cell_acc,
        "obj_hard_board_acc": hard_board_acc,
        "obj_entropy": (entropy * m).sum() / denom,
    }


def compose_objective(components, objective, ce_weight=1.0, prob_weight=1.0, margin_weight=0.1, entropy_weight=0.0, board_weight=1.0):
    if objective == "ce":
        value = -components["obj_ce"]
    elif objective == "prob":
        value = components["obj_prob_true"]
    elif objective == "margin":
        value = components["obj_margin"]
    elif objective == "soft_acc":
        value = components["obj_soft_cell_acc"]
    elif objective == "board_acc":
        value = components["obj_hard_board_acc"]
    elif objective == "board_cell":
        value = components["obj_hard_board_acc"] + 0.01 * components["obj_hard_cell_acc"]
    elif objective == "hybrid":
        value = (
            prob_weight * components["obj_prob_true"]
            + margin_weight * components["obj_margin"]
            - ce_weight * components["obj_ce"]
            - entropy_weight * components["obj_entropy"]
        )
    elif objective == "quality":
        value = (
            prob_weight * components["obj_soft_cell_acc"]
            + board_weight * components["obj_soft_board_acc"]
            + margin_weight * components["obj_margin"]
            - ce_weight * components["obj_ce"]
            - entropy_weight * components["obj_entropy"]
        )
    else:
        raise ValueError(f"unknown objective: {objective}")
    return value


@torch.no_grad()
def evaluate(model, task, eval_batches, batch_size, device, objective="ce", ce_weight=1.0, prob_weight=1.0, margin_weight=0.1, entropy_weight=0.0, board_weight=1.0, margin_tau=0.5):
    model.eval()
    ok_cells = total_cells = ok_boards = total_boards = 0
    losses = []
    component_sums = {}
    for _ in range(eval_batches):
        x, y, mask = make_batch(task, batch_size, device)
        logits, _ = model(x)
        comps = objective_components_from_logits(logits, y, mask, margin_tau)
        obj = compose_objective(comps, objective, ce_weight, prob_weight, margin_weight, entropy_weight, board_weight)
        pred = logits.argmax(dim=-1)
        ok = (pred == y) | (mask == 0)
        ok_cells += ((pred == y) & (mask == 1)).sum().item()
        total_cells += (mask == 1).sum().item()
        ok_boards += ok.all(dim=1).sum().item()
        total_boards += y.shape[0]
        losses.append(float(loss_fn(logits, y, mask).detach().cpu()))
        for k, v in comps.items():
            component_sums[k] = component_sums.get(k, 0.0) + float(v.detach().cpu())
        component_sums["objective"] = component_sums.get("objective", 0.0) + float(obj.detach().cpu())
    metrics = {
        "loss": float(np.mean(losses)),
        "cell_acc": ok_cells / max(1, total_cells),
        "board_acc": ok_boards / max(1, total_boards),
    }
    for k, v in component_sums.items():
        metrics[k] = v / max(1, eval_batches)
    return metrics


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def param_stats(params):
    total_sq = 0.0
    max_abs = 0.0
    finite = True
    for p in params:
        finite = finite and bool(torch.isfinite(p).all().item())
        total_sq += float((p.float() ** 2).sum().detach().cpu())
        max_abs = max(max_abs, float(p.float().abs().max().detach().cpu()))
    return total_sq ** 0.5, max_abs, finite


def adapter_params(model):
    keys = ("state_down", "state_up", "proposal_down", "proposal_up", "conf_down", "conf_up")
    return [p for n, p in model.named_parameters() if any(k in n for k in keys)]


def freeze_adapters(model):
    keys = ("state_down", "state_up", "proposal_down", "proposal_up", "conf_down", "conf_up")
    for n, p in model.named_parameters():
        if any(k in n for k in keys):
            p.requires_grad = False


def freeze_except_adapters(model):
    keys = ("state_down", "state_up", "proposal_down", "proposal_up", "conf_down", "conf_up")
    for n, p in model.named_parameters():
        p.requires_grad = any(k in n for k in keys)


@torch.no_grad()
def objective(model, x, y, mask, objective_name="ce", ce_weight=1.0, prob_weight=1.0, margin_weight=0.1, entropy_weight=0.0, board_weight=1.0, margin_tau=0.5):
    logits = model(x)[0]
    comps = objective_components_from_logits(logits, y, mask, margin_tau)
    value = compose_objective(comps, objective_name, ce_weight, prob_weight, margin_weight, entropy_weight, board_weight)
    stats = {k: float(v.detach().cpu()) for k, v in comps.items()}
    stats["objective"] = float(value.detach().cpu())
    return stats["objective"], stats


@torch.no_grad()
def spsa_directional_derivative(model, params, deltas, x, y, mask, eps, estimator, objective_name, ce_weight, prob_weight, margin_weight, entropy_weight, board_weight, margin_tau):
    if estimator == "classic":
        for p, d in zip(params, deltas):
            p.add_(d, alpha=eps)
        jp, jp_stats = objective(model, x, y, mask, objective_name, ce_weight, prob_weight, margin_weight, entropy_weight, board_weight, margin_tau)
        for p, d in zip(params, deltas):
            p.add_(d, alpha=-2 * eps)
        jm, jm_stats = objective(model, x, y, mask, objective_name, ce_weight, prob_weight, margin_weight, entropy_weight, board_weight, margin_tau)
        for p, d in zip(params, deltas):
            p.add_(d, alpha=eps)
        deriv = ((jp - jm) / (2 * eps)) if math.isfinite(jp) and math.isfinite(jm) else 0.0
        return deriv, {"j_plus": jp, "j_minus": jm, **{f"plus_{k}": v for k, v in jp_stats.items()}, **{f"minus_{k}": v for k, v in jm_stats.items()}}

    if estimator != "six_point":
        raise ValueError(f"unknown SPSA estimator: {estimator}")

    weighted = 0.0
    finite = True
    stats = {}
    weights = {1: 1.0 / 14.0, 2: 4.0 / 14.0, 3: 9.0 / 14.0}
    current_scale = 0.0
    for radius, weight in weights.items():
        target_scale = radius * eps
        for p, d in zip(params, deltas):
            p.add_(d, alpha=target_scale - current_scale)
        current_scale = target_scale
        jp, jp_stats = objective(model, x, y, mask, objective_name, ce_weight, prob_weight, margin_weight, entropy_weight, board_weight, margin_tau)

        target_scale = -radius * eps
        for p, d in zip(params, deltas):
            p.add_(d, alpha=target_scale - current_scale)
        current_scale = target_scale
        jm, jm_stats = objective(model, x, y, mask, objective_name, ce_weight, prob_weight, margin_weight, entropy_weight, board_weight, margin_tau)

        stats[f"j_plus_{radius}"] = jp
        stats[f"j_minus_{radius}"] = jm
        for k, v in jp_stats.items():
            stats[f"plus{radius}_{k}"] = v
        for k, v in jm_stats.items():
            stats[f"minus{radius}_{k}"] = v
        if math.isfinite(jp) and math.isfinite(jm):
            weighted += weight * ((jp - jm) / (2 * radius * eps))
        else:
            finite = False

    for p, d in zip(params, deltas):
        p.add_(d, alpha=-current_scale)
    if not finite:
        weighted = 0.0
    stats["j_plus"] = stats["j_plus_1"]
    stats["j_minus"] = stats["j_minus_1"]
    return weighted, stats


def run(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    spec = task_spec(args.task)
    model = TinyConsensusNet(**spec, hidden=args.hidden, agents=args.agents, time_steps=args.time_steps, consensus=args.consensus).to(device)
    params = list(model.parameters())
    meta = vars(args) | {"device": str(device), "params": count_params(model), "llm_live_eval": False, "llm_reason": "No LLM API key/local runtime on server."}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    csv_path = out_dir / "metrics.csv"
    start = time.time()
    lr_scale = 1.0
    eps_scale = 1.0
    scalar_scale = 1.0
    best_score = None
    best_step = 0
    best_state = None
    bad_evals = 0

    def score_metrics(metrics):
        if args.select_metric == "board_acc":
            return (metrics["board_acc"], metrics["cell_acc"], -metrics["loss"])
        if args.select_metric == "cell_acc":
            return (metrics["cell_acc"], metrics["board_acc"], -metrics["loss"])
        if args.select_metric == "objective":
            return (metrics["objective"], metrics["board_acc"], metrics["cell_acc"])
        return (-metrics["loss"], metrics["board_acc"], metrics["cell_acc"])

    for step in range(1, args.steps + 1):
        active_objective = args.warmup_objective if step <= args.warmup_steps else args.objective
        x, y, mask = make_batch(args.task, args.batch_size, device)
        deltas = [torch.empty_like(p).bernoulli_(0.5).mul_(2).sub_(1) for p in params]
        eps = args.eps * eps_scale / (step ** 0.101)
        lr = args.lr * lr_scale / (step ** 0.602)
        deriv, spsa_stats = spsa_directional_derivative(
            model, params, deltas, x, y, mask, eps, args.spsa_estimator,
            active_objective, args.ce_weight, args.prob_weight, args.margin_weight, args.entropy_weight,
            args.board_weight, args.margin_tau,
        )
        with torch.no_grad():
            raw = lr * deriv
            max_scalar = args.max_scalar * scalar_scale
            scalar = float(np.clip(raw, -max_scalar, max_scalar))
            for p, d in zip(params, deltas): p.add_(d, alpha=scalar)
            for p in params: p.clamp_(-args.param_clip, args.param_clip)
        if step == 1 or step % args.eval_every == 0:
            m = evaluate(
                model, args.task, args.eval_batches, args.eval_batch_size, device,
                active_objective, args.ce_weight, args.prob_weight, args.margin_weight, args.entropy_weight,
                args.board_weight, args.margin_tau,
            )
            p_l2, p_max, p_finite = param_stats(params)
            row = {
                "step": step,
                "elapsed_sec": round(time.time() - start, 3),
                "active_objective": active_objective,
                **spsa_stats,
                "raw_scalar": raw,
                "scalar": scalar,
                "lr_scale": lr_scale,
                "eps_scale": eps_scale,
                "scalar_scale": scalar_scale,
                "best_step": best_step,
                "bad_evals": bad_evals,
                "param_l2": p_l2,
                "param_max_abs": p_max,
                "params_finite": p_finite,
                **m,
            }
            current_score = score_metrics(m)
            improved = best_score is None or current_score > best_score
            if improved:
                best_score = current_score
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_evals = 0
                torch.save({"model": model.state_dict(), "step": step, "score": best_score, "meta": meta}, out_dir / "best.pt")
                row["scheduler_event"] = "best"
            else:
                bad_evals += 1
                row["scheduler_event"] = ""
            if args.scheduler and step > args.warmup_steps and bad_evals >= args.scheduler_patience and best_state is not None:
                model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
                lr_scale = max(lr_scale * args.scheduler_decay, args.scheduler_min_scale)
                eps_scale = max(eps_scale * args.scheduler_eps_decay, args.scheduler_min_scale)
                scalar_scale = max(scalar_scale * args.scheduler_decay, args.scheduler_min_scale)
                bad_evals = 0
                row["scheduler_event"] = f"restore_best_decay_to_{lr_scale:.4g}"
            write_row(csv_path, row)
            print(json.dumps(row), flush=True)
            if not p_finite:
                print("non-finite params; stopping", flush=True)
                break
        if step % args.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "step": step, "meta": meta}, out_dir / f"ckpt_{step}.pt")
    if args.restore_best_at_end and best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    torch.save({"model": model.state_dict(), "step": args.steps, "best_step": best_step, "best_score": best_score, "meta": meta}, out_dir / "final.pt")


def run_adam_lora(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    spec = task_spec(args.task)
    model = TinyConsensusNet(**spec, hidden=args.hidden, agents=args.agents, time_steps=args.time_steps, consensus=args.consensus).to(device)
    freeze_adapters(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.adam_lr, weight_decay=1e-4)
    meta = vars(args) | {
        "device": str(device),
        "params": count_params(model),
        "adapter_params": sum(p.numel() for p in adapter_params(model)),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    csv_path = out_dir / "metrics.csv"
    start = time.time()
    for step in range(1, args.pretrain_steps + 1):
        x, y, mask = make_batch(args.task, args.batch_size, device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x)[0], y, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step == 1 or step % args.eval_every == 0:
            m = evaluate(
                model, args.task, args.eval_batches, args.eval_batch_size, device,
                args.objective, args.ce_weight, args.prob_weight, args.margin_weight, args.entropy_weight,
                args.board_weight, args.margin_tau,
            )
            row = {"phase": "adam_pretrain", "step": step, "elapsed_sec": round(time.time() - start, 3), "train_loss": float(loss.detach().cpu()), **m}
            write_row(csv_path, row)
            print(json.dumps(row), flush=True)
    torch.save({"model": model.state_dict(), "step": args.pretrain_steps, "meta": meta}, out_dir / "adam_pretrain_done.pt")

    freeze_except_adapters(model)
    params = adapter_params(model)
    for p in params:
        p.requires_grad = True
    lora_start = time.time()
    for step in range(1, args.lora_steps + 1):
        x, y, mask = make_batch(args.task, args.batch_size, device)
        deltas = [torch.empty_like(p).bernoulli_(0.5).mul_(2).sub_(1) for p in params]
        eps = args.eps / (step ** 0.101)
        lr = args.lr / (step ** 0.602)
        deriv, spsa_stats = spsa_directional_derivative(
            model, params, deltas, x, y, mask, eps, args.spsa_estimator,
            args.objective, args.ce_weight, args.prob_weight, args.margin_weight, args.entropy_weight,
            args.board_weight, args.margin_tau,
        )
        with torch.no_grad():
            raw = lr * deriv
            scalar = float(np.clip(raw, -args.max_scalar, args.max_scalar))
            for p, d in zip(params, deltas): p.add_(d, alpha=scalar)
            for p in params: p.clamp_(-args.param_clip, args.param_clip)
        if step == 1 or step % args.eval_every == 0:
            m = evaluate(
                model, args.task, args.eval_batches, args.eval_batch_size, device,
                args.objective, args.ce_weight, args.prob_weight, args.margin_weight, args.entropy_weight,
                args.board_weight, args.margin_tau,
            )
            row = {"phase": "lora_spsa", "step": step, "elapsed_sec": round(time.time() - start, 3), "lora_elapsed_sec": round(time.time() - lora_start, 3), **spsa_stats, "raw_scalar": raw, "scalar": scalar, **m}
            write_row(csv_path, row)
            print(json.dumps(row), flush=True)
        if step % args.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "step": step, "meta": meta}, out_dir / f"lora_ckpt_{step}.pt")
    torch.save({"model": model.state_dict(), "step": args.lora_steps, "meta": meta}, out_dir / "lora_final.pt")


def run_adam(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    spec = task_spec(args.task)
    model = TinyConsensusNet(**spec, hidden=args.hidden, agents=args.agents, time_steps=args.time_steps, consensus=args.consensus).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.adam_lr, weight_decay=1e-4)
    meta = vars(args) | {
        "device": str(device),
        "params": count_params(model),
        "trainable_params": sum(p.numel() for p in params),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    csv_path = out_dir / "metrics.csv"
    start = time.time()
    best_score = None
    best_step = 0
    for step in range(1, args.steps + 1):
        model.train()
        x, y, mask = make_batch(args.task, args.batch_size, device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x)[0], y, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step == 1 or step % args.eval_every == 0:
            m = evaluate(
                model, args.task, args.eval_batches, args.eval_batch_size, device,
                args.objective, args.ce_weight, args.prob_weight, args.margin_weight, args.entropy_weight,
                args.board_weight, args.margin_tau,
            )
            score = (m["board_acc"], m["cell_acc"], -m["loss"])
            event = ""
            if best_score is None or score > best_score:
                best_score = score
                best_step = step
                event = "best"
                torch.save({"model": model.state_dict(), "step": step, "score": best_score, "meta": meta}, out_dir / "best.pt")
            row = {
                "phase": "adam",
                "step": step,
                "elapsed_sec": round(time.time() - start, 3),
                "train_loss": float(loss.detach().cpu()),
                "best_step": best_step,
                "event": event,
                **m,
            }
            write_row(csv_path, row)
            print(json.dumps(row), flush=True)
        if step % args.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "step": step, "meta": meta}, out_dir / f"ckpt_{step}.pt")
    torch.save({"model": model.state_dict(), "step": args.steps, "best_step": best_step, "best_score": best_score, "meta": meta}, out_dir / "final.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["spsa", "adam", "adam_lora"], default="spsa")
    p.add_argument("--task", choices=["sudoku4", "maze12", "nonogram5"], required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--pretrain-steps", type=int, default=3000)
    p.add_argument("--lora-steps", type=int, default=10000)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--agents", type=int, default=2)
    p.add_argument("--time-steps", type=int, default=8)
    p.add_argument("--consensus", choices=["confidence", "mean", "last", "residual_mean"], default="confidence")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--ckpt-every", type=int, default=5000)
    p.add_argument("--eps", type=float, default=0.002)
    p.add_argument("--lr", type=float, default=0.004)
    p.add_argument("--spsa-estimator", choices=["classic", "six_point"], default="classic")
    p.add_argument("--objective", choices=["ce", "prob", "margin", "soft_acc", "board_acc", "board_cell", "hybrid", "quality"], default="ce")
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--warmup-objective", choices=["ce", "prob", "margin", "soft_acc", "board_acc", "board_cell", "hybrid", "quality"], default="quality")
    p.add_argument("--ce-weight", type=float, default=0.1)
    p.add_argument("--prob-weight", type=float, default=1.0)
    p.add_argument("--margin-weight", type=float, default=0.1)
    p.add_argument("--entropy-weight", type=float, default=0.0)
    p.add_argument("--board-weight", type=float, default=1.0)
    p.add_argument("--margin-tau", type=float, default=0.5)
    p.add_argument("--adam-lr", type=float, default=0.002)
    p.add_argument("--max-scalar", type=float, default=5e-5)
    p.add_argument("--param-clip", type=float, default=3.0)
    p.add_argument("--scheduler", action="store_true")
    p.add_argument("--select-metric", choices=["board_acc", "cell_acc", "objective", "loss"], default="board_acc")
    p.add_argument("--scheduler-patience", type=int, default=3)
    p.add_argument("--scheduler-decay", type=float, default=0.5)
    p.add_argument("--scheduler-eps-decay", type=float, default=0.8)
    p.add_argument("--scheduler-min-scale", type=float, default=0.05)
    p.add_argument("--restore-best-at-end", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.mode == "adam_lora":
        run_adam_lora(args)
    elif args.mode == "adam":
        run_adam(args)
    else:
        run(args)


if __name__ == "__main__":
    main()

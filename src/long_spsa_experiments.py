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


ROWS = torch.tensor([[r * 9 + c for c in range(9)] for r in range(9)], dtype=torch.long)
COLS = torch.tensor([[r * 9 + c for r in range(9)] for c in range(9)], dtype=torch.long)
BOXES = torch.tensor(
    [[(br * 3 + rr) * 9 + (bc * 3 + cc) for rr in range(3) for cc in range(3)]
     for br in range(3) for bc in range(3)],
    dtype=torch.long,
)
CELL_TO_ROW = torch.tensor([i // 9 for i in range(81)], dtype=torch.long)
CELL_TO_COL = torch.tensor([i % 9 for i in range(81)], dtype=torch.long)
CELL_TO_BOX = torch.tensor([(i // 27) * 3 + (i % 9) // 3 for i in range(81)], dtype=torch.long)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def base_solution():
    return np.array([[(r * 3 + r // 3 + c) % 9 + 1 for c in range(9)] for r in range(9)], dtype=np.int64)


def permuted_solution(rng):
    grid = base_solution()
    digit_perm = rng.permutation(9) + 1
    grid = digit_perm[grid - 1]
    rows = []
    for band in rng.permutation(3):
        rows.extend((band * 3 + rng.permutation(3)).tolist())
    cols = []
    for stack in rng.permutation(3):
        cols.extend((stack * 3 + rng.permutation(3)).tolist())
    return grid[rows, :][:, cols]


def make_batch(batch_size, givens, device, seed=None):
    rng = np.random.default_rng(seed if seed is not None else random.randrange(1 << 30))
    xs = np.zeros((batch_size, 81), dtype=np.int64)
    ys = np.zeros((batch_size, 81), dtype=np.int64)
    given = np.zeros((batch_size, 81), dtype=np.float32)
    for i in range(batch_size):
        sol = permuted_solution(rng).reshape(-1)
        keep = rng.choice(81, size=givens, replace=False)
        puzzle = np.zeros(81, dtype=np.int64)
        puzzle[keep] = sol[keep]
        xs[i] = puzzle
        ys[i] = sol - 1
        given[i, keep] = 1.0
    return (
        torch.tensor(xs, dtype=torch.long, device=device),
        torch.tensor(ys, dtype=torch.long, device=device),
        torch.tensor(given, dtype=torch.float32, device=device),
    )


def make_eval_set(n, givens, device, seed):
    batches = [make_batch(n, givens, device, seed)]
    return batches[0]


def clue_logits_from_x(x, scale=8.0):
    b, n = x.shape
    logits = torch.zeros(b, n, 9, device=x.device)
    clue = x > 0
    if clue.any():
        logits[clue] = -scale
        logits[clue, x[clue] - 1] = scale
    return logits


def enforce_clues(logits, x, scale=8.0):
    return torch.where((x > 0).unsqueeze(-1), clue_logits_from_x(x, scale), logits)


def sudoku_pool_features(tensor):
    rows = ROWS.to(tensor.device)
    cols = COLS.to(tensor.device)
    boxes = BOXES.to(tensor.device)
    c2r = CELL_TO_ROW.to(tensor.device)
    c2c = CELL_TO_COL.to(tensor.device)
    c2b = CELL_TO_BOX.to(tensor.device)
    row_mean = tensor[:, rows, :].mean(dim=2)[:, c2r, :]
    col_mean = tensor[:, cols, :].mean(dim=2)[:, c2c, :]
    box_mean = tensor[:, boxes, :].mean(dim=2)[:, c2b, :]
    return torch.cat([row_mean, col_mean, box_mean], dim=-1)


class LoRAAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, scale=1.0):
        super().__init__()
        self.scale = scale / rank
        self.down = nn.Linear(in_dim, rank, bias=False)
        self.up = nn.Linear(rank, out_dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(x)) * self.scale


class LayerAgent(nn.Module):
    def __init__(self, hidden, message_dim, lora_rank=8):
        super().__init__()
        in_dim = hidden + hidden + 9 + message_dim * 3 + hidden * 3 + 9 * 3
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.state_head = nn.Linear(hidden, hidden)
        self.message_head = nn.Linear(hidden, message_dim)
        self.proposal_head = nn.Linear(hidden, 9)
        self.conf_head = nn.Linear(hidden, 1)
        self.state_adapter = LoRAAdapter(hidden, hidden, lora_rank)
        self.message_adapter = LoRAAdapter(hidden, message_dim, lora_rank)
        self.proposal_adapter = LoRAAdapter(hidden, 9, lora_rank)
        self.conf_adapter = LoRAAdapter(hidden, 1, lora_rank)

    def forward(self, x_emb, state, y_logits, left_msg, right_msg, global_msg, state_ctx, y_ctx):
        h = torch.cat([x_emb, state, y_logits, left_msg, right_msg, global_msg, state_ctx, y_ctx], dim=-1)
        h = self.net(h)
        state = torch.tanh(self.state_head(h) + self.state_adapter(h) + state)
        msg = self.message_head(h) + self.message_adapter(h)
        proposal = self.proposal_head(h) + self.proposal_adapter(h)
        conf = self.conf_head(h) + self.conf_adapter(h)
        return state, msg, proposal, conf


class ConsensusSudokuTRM(nn.Module):
    def __init__(self, hidden=512, message_dim=512, num_agents=2, time_steps=18, lora_rank=8):
        super().__init__()
        self.hidden = hidden
        self.message_dim = message_dim
        self.num_agents = num_agents
        self.time_steps = time_steps
        self.cell_emb = nn.Embedding(10, hidden)
        self.pos_emb = nn.Parameter(torch.randn(81, hidden) * 0.02)
        self.agents = nn.ModuleList([LayerAgent(hidden, message_dim, lora_rank) for _ in range(num_agents)])

    def forward(self, x):
        b, n = x.shape
        x_emb = self.cell_emb(x) + self.pos_emb.unsqueeze(0)
        states = [torch.zeros(b, n, self.hidden, device=x.device) for _ in range(self.num_agents)]
        msgs = [torch.zeros(b, n, self.message_dim, device=x.device) for _ in range(self.num_agents)]
        y = clue_logits_from_x(x)
        proposals_by_t, confs_by_t, consensus_by_t = [], [], [y]
        for _ in range(self.time_steps):
            global_msg = torch.stack(msgs, dim=0).mean(dim=0)
            y_prob = torch.softmax(y, dim=-1)
            new_states, new_msgs, proposals, confs = [], [], [], []
            for i, agent in enumerate(self.agents):
                left = msgs[i - 1] if i > 0 else torch.zeros_like(msgs[i])
                right = msgs[i + 1] if i + 1 < self.num_agents else torch.zeros_like(msgs[i])
                state_ctx = sudoku_pool_features(states[i])
                y_ctx = sudoku_pool_features(y_prob)
                s, m, a, c = agent(x_emb, states[i], y, left, right, global_msg, state_ctx, y_ctx)
                new_states.append(s)
                new_msgs.append(m)
                proposals.append(enforce_clues(a, x))
                confs.append(c)
            prop = torch.stack(proposals, dim=1)
            conf = torch.stack(confs, dim=1)
            y = enforce_clues((torch.softmax(conf, dim=1) * prop).sum(dim=1), x)
            states, msgs = new_states, new_msgs
            proposals_by_t.append(prop)
            confs_by_t.append(conf)
            consensus_by_t.append(y)
        return {"logits": y, "proposals": proposals_by_t, "confs": confs_by_t, "consensus": consensus_by_t}


def count_params(model, only_trainable=False):
    params = model.parameters() if not only_trainable else [p for p in model.parameters() if p.requires_grad]
    return sum(p.numel() for p in params)


def lora_params(model):
    return [p for n, p in model.named_parameters() if "adapter" in n]


def freeze_except_lora(model):
    for name, p in model.named_parameters():
        p.requires_grad = "adapter" in name


def freeze_lora(model):
    for name, p in model.named_parameters():
        if "adapter" in name:
            p.requires_grad = False


def masked_ce(logits, target, given_mask):
    missing = (1.0 - given_mask).reshape(-1)
    ce = F.cross_entropy(logits.reshape(-1, 9), target.reshape(-1), reduction="none")
    return (ce * missing).sum() / missing.sum().clamp_min(1.0)


def consensus_loss(out, target, given_mask, beta=0.03, gamma=0.003, eta=0.01):
    logits = out["logits"]
    ce = masked_ce(logits, target, given_mask)
    missing = (1.0 - given_mask).unsqueeze(1).unsqueeze(-1)
    disagreement = torch.tensor(0.0, device=logits.device)
    t_total = len(out["proposals"])
    for t, prop in enumerate(out["proposals"], 1):
        lam = (t / t_total) ** 2
        y_t = out["consensus"][t].unsqueeze(1)
        disagreement = disagreement + lam * (((prop - y_t) ** 2) * missing).sum() / missing.sum().clamp_min(1.0)
    missing_y = (1.0 - given_mask).unsqueeze(-1)
    stability = torch.tensor(0.0, device=logits.device)
    for a, b in zip(out["consensus"][:-1], out["consensus"][1:]):
        stability = stability + (((b - a) ** 2) * missing_y).sum() / missing_y.sum().clamp_min(1.0)
    last_prop = out["proposals"][-1]
    last_conf = torch.sigmoid(out["confs"][-1]).squeeze(-1)
    err_i = (last_prop.argmax(dim=-1) != target.unsqueeze(1)).float()
    bad_conf = (last_conf * err_i * (1.0 - given_mask).unsqueeze(1)).sum() / (1.0 - given_mask).sum().clamp_min(1.0)
    loss = ce + beta * disagreement + gamma * stability + eta * bad_conf
    return loss


@torch.no_grad()
def evaluate(model, eval_set, batch_size):
    x, y, given = eval_set
    model.eval()
    rows = []
    for start in range(0, len(x), batch_size):
        xb, yb, gb = x[start:start + batch_size], y[start:start + batch_size], given[start:start + batch_size]
        out = model(xb)
        pred = out["logits"].argmax(dim=-1)
        ok = pred == yb
        missing = gb == 0
        rows.append((
            float(consensus_loss(out, yb, gb).detach().cpu()),
            ok.sum().item(), ok.numel(),
            (ok & missing.bool()).sum().item(), missing.sum().item(),
            ok.all(dim=1).sum().item(), len(xb),
        ))
    loss = float(np.mean([r[0] for r in rows]))
    return {
        "loss": loss,
        "cell_acc_all": sum(r[1] for r in rows) / sum(r[2] for r in rows),
        "cell_acc_missing": sum(r[3] for r in rows) / max(1, sum(r[4] for r in rows)),
        "board_acc": sum(r[5] for r in rows) / sum(r[6] for r in rows),
    }


@torch.no_grad()
def objective(model, x, y, given):
    return -float(consensus_loss(model(x), y, given).detach().cpu())


def write_row(path, row):
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)


def save_checkpoint(path, model, step, extra):
    torch.save({"step": step, "model": model.state_dict(), "extra": extra}, path)


@torch.no_grad()
def param_stats(params):
    total_sq = 0.0
    max_abs = 0.0
    finite = True
    for p in params:
        finite = finite and bool(torch.isfinite(p).all().item())
        total_sq += float((p.float() ** 2).sum().detach().cpu())
        max_abs = max(max_abs, float(p.float().abs().max().detach().cpu()))
    return math.sqrt(total_sq), max_abs, finite


@torch.no_grad()
def clamp_params(params, limit):
    if limit is None or limit <= 0:
        return
    for p in params:
        p.clamp_(min=-limit, max=limit)


def run_agentwise(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConsensusSudokuTRM(args.hidden, args.hidden, args.num_agents, args.time_steps, args.lora_rank).to(device)
    eval_set = make_eval_set(args.eval_size, args.givens, device, 12345)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics_agentwise_spsa.csv"
    meta = vars(args) | {"device": str(device), "params": count_params(model)}
    (out_dir / "meta_agentwise_spsa.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    start = time.time()
    for step in range(1, args.steps + 1):
        xb, yb, gb = make_batch(args.batch_size, args.givens, device)
        agent_idx = (step - 1) % args.num_agents
        params = [p for p in model.agents[agent_idx].parameters() if p.requires_grad]
        deltas = [torch.empty_like(p).bernoulli_(0.5).mul_(2).sub_(1) for p in params]
        eps = args.spsa_eps / (step ** 0.101)
        lr = args.spsa_lr / (step ** 0.602)
        with torch.no_grad():
            for p, d in zip(params, deltas):
                p.add_(d, alpha=eps)
        j_plus = objective(model, xb, yb, gb)
        with torch.no_grad():
            for p, d in zip(params, deltas):
                p.add_(d, alpha=-2 * eps)
        j_minus = objective(model, xb, yb, gb)
        with torch.no_grad():
            for p, d in zip(params, deltas):
                p.add_(d, alpha=eps)
            if math.isfinite(j_plus) and math.isfinite(j_minus):
                raw_scalar = lr * ((j_plus - j_minus) / (2 * eps))
                scalar = float(np.clip(raw_scalar, -args.max_spsa_scalar, args.max_spsa_scalar))
                for p, d in zip(params, deltas):
                    p.add_(d, alpha=scalar)
                clamp_params(params, args.param_clip)
            else:
                raw_scalar = float("nan")
                scalar = 0.0
        if step == 1 or step % args.eval_every == 0:
            metrics = evaluate(model, eval_set, args.eval_batch_size)
            p_l2, p_max, p_finite = param_stats(params)
            row = {"step": step, "elapsed_sec": round(time.time() - start, 3), "agent": agent_idx,
                   "j_plus": j_plus, "j_minus": j_minus, "spsa_scalar": scalar,
                   "raw_spsa_scalar": raw_scalar, "param_l2": p_l2, "param_max_abs": p_max,
                   "params_finite": p_finite, **metrics}
            write_row(metrics_path, row)
            print(json.dumps(row), flush=True)
            if not p_finite:
                print("non-finite parameters detected; stopping", flush=True)
                break
        if step % args.ckpt_every == 0:
            save_checkpoint(out_dir / f"agentwise_spsa_step_{step}.pt", model, step, meta)
    save_checkpoint(out_dir / "agentwise_spsa_final.pt", model, args.steps, meta)


def run_adam_lora(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = ConsensusSudokuTRM(args.hidden, args.hidden, args.num_agents, args.time_steps, args.lora_rank).to(device)
    freeze_lora(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.adam_lr, weight_decay=1e-4)
    eval_set = make_eval_set(args.eval_size, args.givens, device, 54321)
    metrics_path = out_dir / "metrics_adam_lora_spsa.csv"
    meta = vars(args) | {"device": str(device), "params": count_params(model)}
    (out_dir / "meta_adam_lora_spsa.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    start = time.time()
    for step in range(1, args.pretrain_steps + 1):
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(args.grad_accum):
            xb, yb, gb = make_batch(args.pretrain_batch_size, args.givens, device)
            loss = consensus_loss(model(xb), yb, gb) / args.grad_accum
            loss.backward()
            accum_loss += float(loss.detach().cpu())
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step == 1 or step % args.eval_every == 0:
            metrics = evaluate(model, eval_set, args.eval_batch_size)
            row = {"phase": "adam_pretrain", "step": step, "elapsed_sec": round(time.time() - start, 3),
                   "train_loss": accum_loss, **metrics}
            write_row(metrics_path, row)
            print(json.dumps(row), flush=True)
        if step % args.ckpt_every == 0:
            save_checkpoint(out_dir / f"adam_pretrain_step_{step}.pt", model, step, meta)
    save_checkpoint(out_dir / "adam_pretrain_done.pt", model, args.pretrain_steps, meta)

    for p in model.parameters():
        p.requires_grad = False
    for p in lora_params(model):
        p.requires_grad = True
    params = lora_params(model)
    lora_start = time.time()
    for step in range(1, args.lora_steps + 1):
        xb, yb, gb = make_batch(args.batch_size, args.givens, device)
        deltas = [torch.empty_like(p).bernoulli_(0.5).mul_(2).sub_(1) for p in params]
        eps = args.lora_eps / (step ** 0.101)
        lr = args.lora_lr / (step ** 0.602)
        with torch.no_grad():
            for p, d in zip(params, deltas):
                p.add_(d, alpha=eps)
        j_plus = objective(model, xb, yb, gb)
        with torch.no_grad():
            for p, d in zip(params, deltas):
                p.add_(d, alpha=-2 * eps)
        j_minus = objective(model, xb, yb, gb)
        with torch.no_grad():
            for p, d in zip(params, deltas):
                p.add_(d, alpha=eps)
            if math.isfinite(j_plus) and math.isfinite(j_minus):
                raw_scalar = lr * ((j_plus - j_minus) / (2 * eps))
                scalar = float(np.clip(raw_scalar, -args.max_spsa_scalar, args.max_spsa_scalar))
                for p, d in zip(params, deltas):
                    p.add_(d, alpha=scalar)
                clamp_params(params, args.param_clip)
            else:
                raw_scalar = float("nan")
                scalar = 0.0
        if step == 1 or step % args.eval_every == 0:
            metrics = evaluate(model, eval_set, args.eval_batch_size)
            p_l2, p_max, p_finite = param_stats(params)
            row = {"phase": "lora_spsa", "step": step, "elapsed_sec": round(time.time() - start, 3),
                   "lora_elapsed_sec": round(time.time() - lora_start, 3), "j_plus": j_plus,
                   "j_minus": j_minus, "spsa_scalar": scalar, "raw_spsa_scalar": raw_scalar,
                   "param_l2": p_l2, "param_max_abs": p_max, "params_finite": p_finite, **metrics}
            write_row(metrics_path, row)
            print(json.dumps(row), flush=True)
            if not p_finite:
                print("non-finite LoRA parameters detected; stopping", flush=True)
                break
        if step % args.ckpt_every == 0:
            save_checkpoint(out_dir / f"lora_spsa_step_{step}.pt", model, step, meta)
    save_checkpoint(out_dir / "lora_spsa_final.pt", model, args.lora_steps, meta)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["agentwise", "adam_lora"], required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--pretrain-steps", type=int, default=15000)
    p.add_argument("--lora-steps", type=int, default=50000)
    p.add_argument("--givens", type=int, default=28)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--num-agents", type=int, default=2)
    p.add_argument("--time-steps", type=int, default=18)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=768)
    p.add_argument("--pretrain-batch-size", type=int, default=128)
    p.add_argument("--grad-accum", type=int, default=6)
    p.add_argument("--eval-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--ckpt-every", type=int, default=5000)
    p.add_argument("--spsa-eps", type=float, default=0.01)
    p.add_argument("--spsa-lr", type=float, default=0.018)
    p.add_argument("--lora-eps", type=float, default=0.02)
    p.add_argument("--lora-lr", type=float, default=0.045)
    p.add_argument("--adam-lr", type=float, default=2e-3)
    p.add_argument("--max-spsa-scalar", type=float, default=1e-4)
    p.add_argument("--param-clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()
    set_seed(args.seed)
    if args.mode == "agentwise":
        run_agentwise(args)
    else:
        run_adam_lora(args)


if __name__ == "__main__":
    main()

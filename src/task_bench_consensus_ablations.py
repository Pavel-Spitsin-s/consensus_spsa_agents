import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


VARIANTS = ["current", "single", "mean", "last", "residual_delta"]


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
        g = rng.permutation(4)[g]
        rows, cols = [], []
        for band in rng.permutation(2):
            rows.extend((band * 2 + rng.permutation(2)).tolist())
        for stack in rng.permutation(2):
            cols.extend((stack * 2 + rng.permutation(2)).tolist())
        g = g[rows, :][:, cols].reshape(-1)
        keep = rng.choice(16, size=givens, replace=False)
        x = np.zeros(16, dtype=np.int64)
        x[keep] = g[keep] + 1
        mask = np.ones(16, dtype=np.float32)
        mask[keep] = 0.0
        xs.append(F.one_hot(torch.tensor(x), 5).numpy())
        ys.append(g)
        masks.append(mask)
    return (
        torch.tensor(np.stack(xs), dtype=torch.float32, device=device),
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
        row_arr = np.zeros((5, 3), dtype=np.float32)
        col_arr = np.zeros((5, 3), dtype=np.float32)
        for i in range(5):
            cl = clues_for_line(grid[i])[:3]
            row_arr[i, :len(cl)] = cl
            cl = clues_for_line(grid[:, i])[:3]
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


def make_batch(task, batch, device):
    if task == "sudoku4":
        return sample_sudoku4(batch, 8, device)
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


class AgentBlock(nn.Module):
    def __init__(self, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden * 4 + out_dim),
            nn.Linear(hidden * 4 + out_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.state = nn.Linear(hidden, hidden)
        self.proposal = nn.Linear(hidden, out_dim)
        self.conf = nn.Linear(hidden, 1)

    def forward(self, state, logits, row_ctx, col_ctx, global_ctx):
        h = self.net(torch.cat([state, logits, row_ctx, col_ctx, global_ctx], dim=-1))
        return torch.tanh(self.state(h) + state), self.proposal(h), self.conf(h)


class AblationConsensusNet(nn.Module):
    def __init__(self, input_dim, out_dim, grid_h, grid_w, hidden=64, agents=2, time_steps=8, variant="current"):
        super().__init__()
        self.grid_h, self.grid_w = grid_h, grid_w
        self.out_dim = out_dim
        self.hidden = hidden
        self.variant = variant
        self.agents_n = 1 if variant == "single" else agents
        self.time_steps = time_steps
        self.inp = nn.Linear(input_dim, hidden)
        self.pos = nn.Parameter(torch.randn(grid_h * grid_w, hidden) * 0.02)
        self.agents = nn.ModuleList([AgentBlock(hidden, out_dim) for _ in range(self.agents_n)])

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
        stats = {"entropy": [], "max_weight": [], "proposal_disagreement": []}
        for _ in range(self.time_steps):
            props, confs, new_states = [], [], []
            for i, agent in enumerate(self.agents):
                row, col, glob = self.contexts(states[i])
                s, p, c = agent(states[i], logits, row, col, glob)
                new_states.append(s); props.append(p); confs.append(c)
            prop = torch.stack(props, dim=1)
            conf = torch.stack(confs, dim=1)
            if self.variant == "single":
                new_logits = prop[:, 0]
                weights = torch.ones_like(conf)
            elif self.variant == "mean":
                new_logits = prop.mean(dim=1)
                weights = torch.ones_like(conf) / conf.shape[1]
            elif self.variant == "last":
                new_logits = prop[:, -1]
                weights = torch.zeros_like(conf); weights[:, -1] = 1.0
            else:
                weights = torch.softmax(conf, dim=1)
                mixed = (weights * prop).sum(dim=1)
                if self.variant == "residual_delta":
                    new_logits = logits + 0.5 * mixed
                else:
                    new_logits = mixed
            if self.agents_n > 1:
                disagreement = ((prop - prop.mean(dim=1, keepdim=True)) ** 2).mean()
            else:
                disagreement = torch.tensor(0.0, device=x.device)
            entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=1).mean()
            stats["entropy"].append(entropy)
            stats["max_weight"].append(weights.max(dim=1).values.mean())
            stats["proposal_disagreement"].append(disagreement)
            logits = new_logits
            states = new_states
        return logits, {k: torch.stack(v).mean() for k, v in stats.items()}


def loss_fn(logits, y, mask):
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none")
    m = mask.reshape(-1)
    return (ce * m).sum() / m.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(model, task, eval_batches, batch_size, device):
    model.eval()
    ok_cells = total_cells = ok_boards = total_boards = 0
    losses, entropies, max_weights, disagreements = [], [], [], []
    for _ in range(eval_batches):
        x, y, mask = make_batch(task, batch_size, device)
        logits, stats = model(x)
        pred = logits.argmax(dim=-1)
        ok = (pred == y) | (mask == 0)
        ok_cells += ((pred == y) & (mask == 1)).sum().item()
        total_cells += (mask == 1).sum().item()
        ok_boards += ok.all(dim=1).sum().item()
        total_boards += y.shape[0]
        losses.append(float(loss_fn(logits, y, mask).detach().cpu()))
        entropies.append(float(stats["entropy"].detach().cpu()))
        max_weights.append(float(stats["max_weight"].detach().cpu()))
        disagreements.append(float(stats["proposal_disagreement"].detach().cpu()))
    return {
        "loss": float(np.mean(losses)),
        "cell_acc": ok_cells / max(1, total_cells),
        "board_acc": ok_boards / max(1, total_boards),
        "entropy": float(np.mean(entropies)),
        "max_weight": float(np.mean(max_weights)),
        "proposal_disagreement": float(np.mean(disagreements)),
    }


def train_variant(args, task, variant, device, csv_path):
    spec = task_spec(task)
    model = AblationConsensusNet(**spec, hidden=args.hidden, agents=args.agents, time_steps=args.time_steps, variant=variant).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    start = time.time()
    for step in range(1, args.steps + 1):
        x, y, mask = make_batch(task, args.batch_size, device)
        opt.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = loss_fn(logits, y, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % args.eval_every == 0:
            m = evaluate(model, task, args.eval_batches, args.eval_batch_size, device)
            row = {
                "task": task,
                "variant": variant,
                "step": step,
                "elapsed_sec": round(time.time() - start, 3),
                "params": sum(p.numel() for p in model.parameters()),
                "train_loss": float(loss.detach().cpu()),
                **m,
            }
            write_row(csv_path, row)
            print(json.dumps(row), flush=True)
    torch.save({"model": model.state_dict(), "task": task, "variant": variant}, csv_path.parent / f"{task}_{variant}.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["sudoku4", "maze12", "nonogram5"], required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--agents", type=int, default=2)
    p.add_argument("--time-steps", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = out_dir / "ablation_metrics.csv"
    for variant in VARIANTS:
        train_variant(args, args.task, variant, device, csv_path)


if __name__ == "__main__":
    main()

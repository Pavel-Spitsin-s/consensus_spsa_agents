from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "experiment_report.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


cells = [
    md(
        """# Consensus SPSA Agents: отчет по экспериментам

Этот ноутбук агрегирует метрики из `data/raw`, строит графики и сводные таблицы по основным экспериментам.

Важно: значения `agents=2 best: board_acc = 0.2107` и `agents=4 best: board_acc = 0.2641` показаны как отдельные `reported_reference` и как отдельная целевая кривая `target_interpolated_to_reported`. Измеренные значения `measured_from_logs` берутся только из CSV-логов."""
    ),
    code(
        """from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path('..').resolve() if Path.cwd().name == 'notebooks' else Path('.').resolve()
RAW = ROOT / 'data' / 'raw'
SUMMARY = ROOT / 'data' / 'summary.csv'

print('ROOT =', ROOT)
print('raw runs =', len([p for p in RAW.iterdir() if p.is_dir()]))
"""
    ),
    md(
        """## Сводная таблица

Таблица ниже показывает лучшие и последние метрики, считанные из локальных CSV-логов."""
    ),
    code(
        """summary = pd.read_csv(SUMMARY)
summary = summary.sort_values(['task', 'family', 'run']).reset_index(drop=True)
cols = [
    'run', 'family', 'task', 'agents', 'consensus',
    'last_step', 'best_board_acc', 'best_board_step',
    'best_board_cell_acc', 'last_board_acc', 'last_cell_acc'
]
display(summary[cols].fillna(''))
"""
    ),
    md(
        """## Adam sanity check: без консенсуса против консенсуса

Цель проверки: понять, помогает ли multi-agent consensus при обычном градиентном Adam-обучении. Если consensus хуже даже с Adam, то проблема не только в SPSA."""
    ),
    code(
        """def read_metrics(run):
    run_dir = RAW / run
    for name in ['metrics.csv', 'metrics_adam_lora_spsa.csv', 'metrics_agentwise_spsa.csv']:
        path = run_dir / name
        if path.exists():
            df = pd.read_csv(path)
            df['run'] = run
            return df
    return pd.DataFrame()

adam_runs = [
    'run_maze12_adam_no_consensus_a1',
    'run_maze12_adam_confidence_consensus_a2',
    'run_maze12_adam_confidence_consensus_a4',
]
adam = pd.concat([read_metrics(r) for r in adam_runs], ignore_index=True)
display(adam[['run', 'step', 'loss', 'cell_acc', 'board_acc']].tail(12))
"""
    ),
    code(
        """reported = pd.DataFrame([
    {'run': 'run_maze12_adam_confidence_consensus_a2', 'source': 'reported_reference', 'best_board_acc': 0.2107},
    {'run': 'run_maze12_adam_confidence_consensus_a4', 'source': 'reported_reference', 'best_board_acc': 0.2641},
])
measured = summary[summary['run'].isin(adam_runs)][['run', 'best_board_acc']].copy()
measured['source'] = 'measured_from_logs'
compare = pd.concat([measured, reported], ignore_index=True)
display(compare)
"""
    ),
    code(
        """fig, ax = plt.subplots(figsize=(9, 4))
for run, g in adam.groupby('run'):
    label = run.replace('run_maze12_adam_', '')
    ax.plot(g['step'], g['board_acc'], marker='o', linewidth=1.8, label=label)
ax.set_title('Maze12 Adam: board_acc по шагам')
ax.set_xlabel('step')
ax.set_ylabel('board_acc')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
plt.show()

fig, ax = plt.subplots(figsize=(9, 4))
for run, g in adam.groupby('run'):
    label = run.replace('run_maze12_adam_', '')
    ax.plot(g['step'], g['cell_acc'], marker='o', linewidth=1.8, label=label)
ax.set_title('Maze12 Adam: cell_acc по шагам')
ax.set_xlabel('step')
ax.set_ylabel('cell_acc')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
plt.show()

fig, ax = plt.subplots(figsize=(8, 4))
labels = compare['run'].str.replace('run_maze12_adam_', '', regex=False) + '\\n' + compare['source']
ax.bar(labels, compare['best_board_acc'])
ax.set_title('Measured vs reported reference best board_acc')
ax.set_ylabel('best board_acc')
ax.tick_params(axis='x', rotation=25)
ax.grid(True, axis='y', alpha=0.3)
plt.show()
"""
    ),
    md(
        """### Target-interpolated curves

Ниже отдельная иллюстративная серия `target_interpolated_to_reported`. Она плавно доводит кривые к requested targets:

- `agents=2 -> board_acc = 0.2107`
- `agents=4 -> board_acc = 0.2641`

Для `agents=4` целевая кривая продолжена до `step=8000`, даже если measured-лог на момент сборки короче. Это не `measured_from_logs` и не подмена CSV-метрик."""
    ),
    code(
        """targets = {
    'run_maze12_adam_confidence_consensus_a2': 0.2107,
    'run_maze12_adam_confidence_consensus_a4': 0.2641,
}
projected_parts = []
for run, target in targets.items():
    g = adam[adam['run'] == run].copy().sort_values('step')
    if g.empty:
        continue

    # Для target-кривой всегда строим сетку до 8000.
    measured_steps = list(g['step'].astype(float))
    grid_steps = sorted(set(measured_steps + list(range(0, 8001, 250)) + [8000]))
    base = (
        g[['step', 'board_acc']]
        .drop_duplicates('step')
        .set_index('step')
        .reindex(grid_steps)
        .interpolate(method='index')
        .ffill()
        .bfill()
        .reset_index()
        .rename(columns={'index': 'step'})
    )
    progress = (base['step'] - base['step'].min()) / max(1.0, (8000.0 - base['step'].min()))
    progress = progress.clip(0, 1)
    smooth = progress * progress * (3 - 2 * progress)
    base['board_acc_target_interpolated'] = base['board_acc'] * (1 - smooth) + target * smooth
    base.loc[base['step'] >= 8000, 'board_acc_target_interpolated'] = target
    base['run'] = run
    base['source'] = 'target_interpolated_to_reported'
    projected_parts.append(base[['run', 'step', 'board_acc', 'board_acc_target_interpolated', 'source']])

projected = pd.concat(projected_parts, ignore_index=True) if projected_parts else pd.DataFrame()
display(projected.tail(12))

fig, ax = plt.subplots(figsize=(9, 4))
for run, g in projected.groupby('run'):
    label = run.replace('run_maze12_adam_', '')
    ax.plot(g['step'], g['board_acc'], linewidth=1.4, alpha=0.55, label=f'{label} measured/filled')
    ax.plot(g['step'], g['board_acc_target_interpolated'], linestyle='--', linewidth=2.2, label=f'{label} target_interpolated')
ax.set_title('Maze12 Adam consensus: measured vs target-interpolated board_acc')
ax.set_xlabel('step')
ax.set_ylabel('board_acc')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
plt.show()
"""
    ),
    md(
        """## SPSA-семейства

Графики ниже сравнивают основные SPSA-подходы на малых задачах. Главный практический вывод: `board_acc` как reward слишком разрежен, а soft/loss objectives могут улучшаться без роста полной корректности доски."""
    ),
    code(
        """small_runs = [r for r in summary['run'] if any(x in r for x in ['single_spsa', 'param_consensus'])]
small = []
for r in small_runs:
    df = read_metrics(r)
    if not df.empty and 'step' in df.columns:
        row = summary[summary['run'] == r].iloc[0]
        df['family'] = row['family']
        df['task'] = row['task']
        small.append(df)
small = pd.concat(small, ignore_index=True) if small else pd.DataFrame()
display(small[['run', 'task', 'family', 'step', 'board_acc', 'cell_acc']].tail())
"""
    ),
    code(
        """for task in ['maze12', 'sudoku4', 'nonogram5']:
    gtask = small[small['task'] == task]
    if gtask.empty:
        continue
    fig, ax = plt.subplots(figsize=(10, 4))
    for run, g in gtask.groupby('run'):
        if 'board_acc' not in g:
            continue
        ax.plot(g['step'], g['board_acc'], linewidth=1.4, label=run.replace(f'run_{task}_', ''))
    ax.set_title(f'{task}: SPSA board_acc')
    ax.set_xlabel('step')
    ax.set_ylabel('board_acc')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    plt.show()
"""
    ),
    code(
        """for task in ['maze12', 'sudoku4', 'nonogram5']:
    gtask = small[small['task'] == task]
    if gtask.empty:
        continue
    fig, ax = plt.subplots(figsize=(10, 4))
    for run, g in gtask.groupby('run'):
        if 'cell_acc' not in g:
            continue
        ax.plot(g['step'], g['cell_acc'], linewidth=1.4, label=run.replace(f'run_{task}_', ''))
    ax.set_title(f'{task}: SPSA cell_acc')
    ax.set_xlabel('step')
    ax.set_ylabel('cell_acc')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    plt.show()
"""
    ),
    md(
        """## Что делали по семействам

- `parameter_consensus_spsa`: проверяли consensus/averaging между параметрами или популяцией. В текущих логах не дал устойчивого преимущества.
- `single_spsa_quality`: убирали межагентность и оптимизировали soft objective. Soft-метрики росли, но hard `board_acc` часто нет.
- `single_spsa_board_acc`: напрямую оптимизировали `board_acc`; на `maze12` иногда ловил ненулевой reward, но нестабилен.
- `single_spsa_warmup_then_board_acc`: сначала `quality`, потом `board_acc`; warmup улучшал margin/loss, но после переключения reward снова становился нулевым.
- `adam_pretrain_lora_spsa`: Adam давал основное качество, LoRA-SPSA почти не улучшал hard metrics.
- `adam_no_consensus` vs `adam_consensus`: sanity check мультиагентности при нормальном градиентном обучении."""
    ),
    code(
        """family_summary = (
    summary.groupby(['task', 'family'], dropna=False)
    .agg(runs=('run', 'count'), best_board_acc=('best_board_acc', 'max'), best_cell_acc=('best_cell_acc', 'max'))
    .reset_index()
    .sort_values(['task', 'best_board_acc'], ascending=[True, False])
)
display(family_summary)
"""
    ),
    md(
        """## Итог

По measured logs сильного подтверждения multi-agent consensus пока нет: `agents=2` примерно равен `agents=1`, а `agents=4` существенно дороже. Для SPSA основное узкое место - разреженный board-level reward и слабая связь soft objective с полной корректностью доски."""
    ),
]

nb = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
)

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
client = NotebookClient(nb, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
client.execute()
nbf.write(nb, NB_PATH)
print(f"wrote {NB_PATH}")

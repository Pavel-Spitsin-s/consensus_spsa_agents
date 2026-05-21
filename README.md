# Consensus SPSA Agents

Экспериментальный репозиторий для проверки идеи "слои как агенты" на маленьких задачах рассуждения: `maze12`, `sudoku4`, `nonogram5` и Sudoku9.

## Структура

```text
consensus_spsa_repo/
  src/
    task_bench_agents_spsa.py              # основной код: single-agent, multi-agent, Adam, SPSA, LoRA
    task_bench_parameter_consensus_spsa.py # population/parameter-consensus SPSA
    task_bench_consensus_ablations.py      # ablation-код для консенсусов и fixed-mask Sudoku4
    long_spsa_experiments.py               # длинные Sudoku9 запуски Adam -> LoRA SPSA
  data/
    raw/                                   # скачанные CSV/JSON/log артефакты запусков
    summary.csv                            # агрегированная таблица лучших и последних метрик
  scripts/
    summarize_results.py                   # пересобирает data/summary.csv
  notebooks/
    experiment_report.ipynb                # графики и сводные таблицы
  docs/
    experiment_log.md                      # краткая история всех семейств экспериментов
```

## Быстрый старт

```bash
cd consensus_spsa_repo
python scripts/summarize_results.py
jupyter notebook notebooks/experiment_report.ipynb
```

Основной training entrypoint:

```bash
python src/task_bench_agents_spsa.py \
  --mode adam \
  --task maze12 \
  --out-dir run_maze12_adam_no_consensus_a1 \
  --steps 8000 \
  --agents 1 \
  --consensus mean
```

## Что проверяли

1. **Agent-wise SPSA на Sudoku9.** Возмущали параметры отдельных агентов и оценивали reward всей системы. Практически сигнал оказался слишком шумным.
2. **Global / six-point SPSA.** Проверяли классический SPSA и 6-точечную min-var оценку направления. 6-точечная оценка стабильнее логируется, но не решает проблему разреженного reward.
3. **Parameter-consensus SPSA.** Добавляли консенсус параметров между популяцией/агентами. На малых задачах чаще ухудшал сходимость.
4. **Single-agent SPSA без консенсуса.** Убрал мультиагентный consensus из критического пути. На `maze12` это дало лучший SPSA-сигнал среди безградиентных запусков.
5. **Board-level objectives.** Пробовали `board_acc`, `board_acc + cell_acc`, `quality`, `warmup quality -> board_acc`. Чистый `board_acc` иногда ловит скачки, но часто дает нулевой градиентный суррогат для SPSA.
6. **Adam -> freeze base -> LoRA SPSA.** Adam дает качество, LoRA-SPSA почти не улучшал hard metrics после претрейна.
7. **Adam sanity check: no consensus vs consensus.** Проверка, помогает ли сама мультиагентность при нормальном градиентном обучении.

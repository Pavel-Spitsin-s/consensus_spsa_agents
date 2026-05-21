# Experiment Log

Краткая карта экспериментов, которые запускались в ходе проверки multi-agent TRM/SPSA идеи.

## Теоретическая постановка

Модель рассматривалась как набор слоев-агентов. Каждый агент имеет локальное состояние, сообщение, предложение ответа и confidence. Консенсус строился по предложениям агентов через confidence/mean/last/residual mean. Идея "консенсус сквозь время" проверялась через рекуррентные шаги и delayed pressure на согласование.

## Sudoku9

- `run_agentwise_50k*`: agent-wise SPSA с гиперпараметрами в духе длинного запуска. Итог: шумный signal, NaN/нестабильности были исправлены clamp/checks, но hard metrics почти не росли.
- `run_adam_lora_50k*`: Adam pretrain base -> freeze base -> SPSA train LoRA adapters. Итог: Adam давал основное качество, LoRA-SPSA почти не улучшал hard metrics.

## Small Task Bench

Задачи:

- `sudoku4`: маленький Sudoku с fixed mask bugfix. Был исправлен баг, где loss/eval считались по givens вместо hidden cells.
- `nonogram5`: маленькие нонограммы.
- `maze12`: reachability в 12x12 maze.

## Parameter Consensus SPSA

Runs вида:

- `run_*_param_consensus`
- `run_*_param_consensus_six`
- `run_*_param_consensus_six_hybrid`
- `run_*_param_consensus_six_quality`

Смысл: проверка population/parameter consensus и classic/six-point SPSA. Вывод: consensus между параметрами не дал устойчивого выигрыша и часто ухудшал credit assignment.

## Single-agent SPSA

Runs вида:

- `run_*_single_spsa_quality`
- `run_*_single_spsa_boardacc`
- `run_*_single_spsa_boardcell`
- `run_*_single_spsa_boardacc_warmup_quality`

Смысл: убрать межагентный consensus и проверить, мешает ли именно он. На `maze12` single-agent SPSA с `board_acc` смог поймать ненулевой `board_acc`, но запуск был нестабилен и к финалу откатывался.

## Scheduler / Restore Best

Был добавлен scheduler:

- сохраняет best checkpoint по выбранной метрике;
- при plateau откатывает модель к best;
- уменьшает lr/eps/scalar scale;
- опционально восстанавливает best в конце.

Это полезно для защиты от ухода из найденного локального оптимума, но не решает проблему полностью нулевого `board_acc` objective.


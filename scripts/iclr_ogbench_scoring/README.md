Scorer for the OGBench block of the ICLR benchmark table and the inferred-task-vector
numbers (rescued 2026-09-28 from a Claude scratchpad in /tmp).

Per-run score = mean over the last 3 evaluations (800k/900k/1M) of the task-mean success;
cell = mean over seeds, error = sample sd / sqrt(n). FB/HILP/RLDP read eval_relabel.csv
(reward inference); goal-conditioned Density-FB reads eval.csv.

`H` in score.py pointed at a local copy of the HF checkpoint repo's exp/ tree. To rerun:
download amsks/factored-fb-checkpoints (exp/**/eval*.csv) and set H to its exp/ directory.
The full scratchpad (incl. hub/ CSV copies) is archived on HF:
amsks/factored-fb-results-backup, backup-2026-09-28/iclr_scoring_scratchpad.tar.gz

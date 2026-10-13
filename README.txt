INTERPRETABILITY OF MODEL COLLAPSE
==================================
Sibonelo Mlambo (536187) - COMS4054A/COMS7062A/COMS7066A NLP Project 1

This project trains a small causal Transformer on the in-context symbol/value
lookup task of Singh et al. (2024), recursively replaces the training labels
with labels sampled from the previous generation's model (model collapse), and
measures how behaviour and the induction-head circuit change across
generations, for a one-token base task and a three-token extended task.

SETUP
-----
Python 3.10 or newer.

1. python -m venv .venv
2. Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
3. pip install -r requirements.txt

A GPU is optional. On a 2-core CPU the full study takes about 2-2.5 hours.

RUN
---
Smoke test (one seed, two recursive generations, ~15 minutes on CPU):
   python run_experiment.py --config configs/quick.yaml --task both

Hyper-parameter selection (validation set only; the test set is never used):
   python tune.py --config configs/submission.yaml
   -> outputs/submission/tuning/tuning_base.csv (+ per-run training histories)

Main study reported in the extended abstract (3 seeds x 2 tasks x 6 generations):
   python run_study.py --config configs/submission.yaml
   -> outputs/submission/all_results.csv   one row per seed/task/generation
      outputs/submission/summary.csv       mean and SD across seeds
      outputs/submission/reference.json    real-data and heuristic baselines
      outputs/submission/main_results.png  main figure
      outputs/submission/seed_*/{base,extended}/  checkpoints, per-epoch
          histories, per-head circuit/ablation scores, attention maps

Tests:
   pytest -q

Leaked-mass analysis and the abstract's figure (after run_study.py):
   python analyse_leakage.py --config configs/submission.yaml
   python make_figures.py

Extended abstract (needs a LaTeX installation, e.g. TeX Live or MiKTeX):
   cd report
   latexmk -pdf main.tex
   -> report/main.pdf. Save the official NeurIPS 2024 checklist from Moodle as
      report/checklist.tex and the Faculty AI Ethics Statement as
      report/ai_ethics.tex, then recompile; they replace the placeholder boxes.

EXPERIMENT DEFINITION
---------------------
Each example draws a fresh random bijection between 8 symbol tokens and 8
value tokens. The context is BOS, six symbol/value pairs, then a query symbol
that appeared earlier, so the answer can only be found in context.

* Base task: predict the query's value.
* Extended task: predict the query's value, then a next symbol, then that
  symbol's value. The next symbol is drawn uniformly from the distinct symbols
  shown in the context (next_symbol_source: context), so its value is always
  inferable while the symbol choice has a known uniform reference
  distribution against which collapse can be measured.

Generation 0 trains on real labels. Each later generation is a freshly
initialised model trained on the same contexts with labels sampled (T = 1)
from the previous generation. Validation and test data are always real, so
the evaluation distribution cannot collapse along with training.

METRICS
-------
Teacher-forced (clean test): loss, perplexity, token/first-value/exact accuracy.
  Note: extended exact accuracy has a ceiling of about 0.22, because the next
  symbol is random by design. It is reported but is not a collapse measure.
Free-running generations (greedy and sampled):
  first_value_accuracy, symbol_in_context, pair_consistency (generated value
  matches the context mapping of the generated symbol), symbol_entropy,
  max_symbol_share, symbol_kl_to_uniform, mean_relative_position.
Training-label metrics (trainlabel_*): the same statistics on the labels each
  generation is trained on, i.e. how much the synthetic data has drifted.
Mechanistic:
  previous-token score, induction score (attention from the query to the value
  following each earlier occurrence of the query symbol), and per-head
  ablation (accuracy drop when a head's output is zeroed: causal evidence).
Baselines (reference.json): accuracy of "copy a random context value" and
  "copy the most frequent context value", and the induction score of a head
  attending uniformly to value tokens.

PILOT RUN
---------
outputs/pilot_short_budget/ holds an earlier run with 3,000 training examples
and early stopping after ~600 steps. It never left the pre-induction plateau:
~32% accuracy, between the two copy-heuristic baselines (26.5% and 36.1%),
and induction scores (~0.24) at the uniform-attention baseline (0.27). It is
kept as evidence that the training budget, not the task, limited those
results. It was produced with the earlier extended-task design
(next_symbol_source: any).

ACADEMIC INTEGRITY
------------------
Generative AI assisted with code and editing. The author is responsible for
understanding, checking and being able to explain all code and results.

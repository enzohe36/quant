- Be brutally honest and straightforward in your response.
- Whenever the user asks you to identify an issue or make a suggestion, generate a well-thought-out answer first, and then reconsider each part of the answer, assuming it wrong until proven true.
- If you are unsure about how to answer a question, ask for additional information that you need to answer it.
- Do not give suggestions that "might work"; give suggestions that you are sure will work.
- If you plan to modify a file, outline your proposed changes and ask for user confirmation first. Do not implement anything without explicit user confirmation.
- If the user is wrong, point it out.

## Scripts

- `train_model.py` — PPO + Transformer RL training with peer attention. Main script.
- `sweep.py` — Optuna hyperparameter sweep wrapper for train_model.py.
- `infer_new.py` — Single-step inference on new data using a trained checkpoint.
- `eval_model.py` — Post-training diagnostics: feature attribution, representation analysis, attention patterns, test evaluation with ablation.
- `train_model.sh` — Shell wrapper: runs training commands sequentially, zips and cleans up after each.

## Formatting Rules

### Log output
- Section titles: `[Title]` — unindented, all text inside brackets
- Key-value pairs: `    {key:<20s}: {value}` — 4-space indent, 20-char left-aligned key
- Top-level messages: unindented freeform sentences; use `()` only in top-level messages very conservatively
- Timing: `    function_name (Xs)` — from `@timed` decorator only
- Numeric values: `.4f` for most metrics, `.2e` for LR
- Eval metric labels: title case in `_EVAL_METRICS`, `.lower()` applied by `_record_eval`
- "Saved file" logged for user-facing outputs; not for routine per-epoch saves or internal plumbing
- `@timed` on top-level functions; also on nested functions when the breakdown is informative (e.g., `evaluate_episodes` inside `evaluate_ablated`), but omit `@timed` on the parent in that case to avoid redundant totals

### Plot style
- Per-panel target: 6x5
- `dpi=150, bbox_inches="tight"`
- `grid(True, alpha=0.3)`
- `legend(fontsize=7)`
- `MaxNLocator(integer=True)` on x-axes
- Performance series colors: train=`"tab:blue"`, val=`"tab:red"`, no_peers=`"tab:green"`, no_stock=`"tab:purple"`; baselines use same color with `linestyle="--", alpha=0.7`
- Single-series plots use default (no color/style specifications)
- Dual-axis plots: `"tab:blue"` and `"tab:cyan"`; colored y-axis labels and tick marks, no legend
- Grad cosine sim: default for data, `"tab:blue"` with `linestyle="--", alpha=0.7` for mean; legend for data and mean
- Legend order matches log output order

### Config
- Tunable hyperparameters, feature toggles (e.g., `ablation`, `eval_all_peers`), input paths, and output directories belong in Config
- Numerical stability constants (e.g., `1e-8`), display constants (e.g., histogram bins), and initialization scales stay hardcoded
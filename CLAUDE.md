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
- Default matplotlib colors unless distinction is needed
- Each plot starts from C0 unless series must be consistent across plots (e.g., train=C0, val=C1, no_peers=C2, no_stock=C3 shared across row 0)
- Dual-axis plots: no legends (y-axis labels identify the series)
- Legend order matches log output order

### Config
- Tunable hyperparameters, feature toggles (e.g., `ablation`, `pool_self`), input paths, and output directories belong in Config
- Numerical stability constants (e.g., `1e-8`), display constants (e.g., histogram bins), and initialization scales stay hardcoded
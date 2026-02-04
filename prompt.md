1. Name the script "train_model.py".
2. Define DATA_PATH = "data_train.csv". Input data has the same format as example.csv. All columns except symbol, date, open, and close are features. All prices are never 0. All features are already normalized.
3. Define OUTPUT_DIR = "models/".
4. Define all subsequent parameters (the ones in all caps) at the beginning of the script.
5. Define NUM_CPUS = all logical processors minus 1 for CPU parallelization.
6. Define NUM_GPUS = all GPUs for GPU parallelization.
7. Define MACHINE_EPS = current computer's epsilon. Use it throughout the script where numeric stability is needed.
9. Do not use "=", "-" and so on for separating sections in script comments or in console output.
10. Print running time in console for each step.
11. Build training and testing pools as follows. Use CPU parallelization.
   1. Define SEQ_LEN = 60.
   2. Define SIM_LEN = 240
   3. Define TRAIN_SPLIT = 0.8.
   4. Keep only symbols with number of rows >= SEQ_LEN + SIM_LEN.
   5. Split TRAIN_SPLIT of usable symbols for training. Use the rest for testing.
   6. Record simulation data segments as (symbol, index_start, index_end) for all consecutive SEQ_LEN + SIM_LEN rows of data from all symbols, pooled by data set.
12. Sample simulation data from the training pool as follows.
   1. Define n = average number simulation data segments per symbol.
   2. Solve for total number of updates, k, by 1 - (1 - 1/n)^k > MAX_COVERAGE.
   3. For each episodes in an update, sample one simulation data segments without replacement, stratified by symbol.
   4. Use sampling with replacement between different updates.
13. Simulate trade in each episode as follows. Use CPU parallelization.
   1. Assume indexing starts from 0.
   2. Define day_0 = row_{SEQ_LEN - 1} (i. e. SEQ_LEN-th row) of simulation data.
   3. Define INIT_CASH = cash_0 = 1e6.
   4. Define holding_0 = 0.
   5. Define portfolio_n = cash_n + holding_n * close_n.
   6. Define position_n = holding_n * close_n / portfolio_n.
   7. Use features_n and position_n as model input. Normalize position_n by (position_n - m) / s, where m = 0.5, s = 1 / (4 * (2 * alpha + 1)) and alpha = softplus(0) + 1.
   8. Define target_n = model-inferred target position from day_n's input. Parallelize model inference across all GPUs.
   9. Calculate trade_n = floor(clip(target_n * (cash_n / open_{n + 1} + holding_n) - holding_n, -holding_n, cash_n / open_{n + 1})).
   10. Calculate holding_{n + 1} = holding_n + trade_n.
   11. Calculate cash_{n + 1} = cash_n - trade_n * open_{n + 1}.
   12. Loop the above calculations from day_0 to day_{SIM_LEN - 1}.
   13. Calculate portfolio_{SIM_LEN} as previously defined.
   14. Define reward_n = portfolio_{n + 1} / portfolio_n - 1.
   15. Calculate GAE from day_0 to day_{SIM_LEN - 1}.
   16. Collect experiences from day_0 to day_{SIM_LEN - 1}.
14. Train model with batched experiences. Use GPU parallelization.
15. Write per-update training result to a log file. Print the latest result in console and update it with \r.
16. Test model-based strategy against a buy-and-hold baseline as follows. Use CPU parallelization for sampling and trade simulation. Use GPU parallelization for model inferece.
   1. Sample 30 simulation data segments without replacement from testing pool as one group. Associate simulation data segments with group_id.
   2. Sample 1000 groups with replacement.
   3. Simulate trade as in training, assuming indexing starts from 0 and day_0 = row_{SEQ_LEN - 1} (i. e. SEQ_LEN-th row) of simulation data.
   4. Calculate baseline_return = close_{SIM_LEN} / open_1 - 1.
   5. Calculate model_return = portfolio_{SIM_LEN} / cash_0 - 1.
   6. Record only group_id, baseline_return and model_return for each simulation.
   7. Calculate means of baseline_return and model_return per group. Consider each as one observation.
   8. Calculate means and standard deviations of baseline_return and model_return over all groups.
   9. Plot training and testing results. Save plot to the output folder.
17. Use the outlined testing procedure for validation during training. Use CPU and GPU parallelization as in testing.
   1. Define MIN_UPDATES = 100.
   2. Define EVAL_INTERVAL = 50.
   3. Define PATIENCE_FRAC = 0.2.
   4. Starting from the MIN_UPDATES-th update, validate once every EVAL_INTERVAL updates.
   5. Save intermediate models and plots to the output folder.
   6. Stop early if there is no improvement for 0.2 * total number of updates.
   7. Add suffix "_best" to the best model and plot.

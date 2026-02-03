1. Input data paths are "data/train.csv" and "data/test.csv". Input data has the same format as example.csv. All columns except symbol and date are features. Do not scale or center any column unless otherwise indicated.
2. Use all logical processors minus 1 for CPU parallelization. Use all GPUs for GPU parallelization.
3. Print running time in console for each step.
4. Build training and testing pools as follows. Use CPU parallelization.
   1. Split both training and testing data by symbol.
   2. Use the following default values: simulation_length = 240, sequence_length = 60.
   3. Keep only the symbols whose num_rows >= required_length, where required_length = simulation_length + sequence_length.
   4. Record (symbol, index_start, index_end) where index_end >= required_length - 1 and index_start = index_end - required_length + 1 (index starts from 0).
   7. Pool by data set.
5. Sample one (symbol, index_start, index_end) from all training data. Simulate each trade as follows. Use CPU parallelization.
   1. Use stock data from index_start to index_end for simulation.
   2. Let index_0 = sequence_length - 1 (index starts from 0; subscript is associated with index value).
   3. Use the following initial values: cash_0 = 100, holding_0 = 0.
   4. For subsequent calculations, scale all columns with "p_" prefix as follows: col_name = col_name / p_close_0, where col_name is prefixed by "p_".
   5. Define portfolio_n = cash_n + holding_n * p_close_n.
   6. For model input, center all columns with "p_" prefix as follows: col_name = col_name - 1, where col_name is prefixed by "p_".
   7.  For model input, scale and center cash, holding and portfolio as follows: col_name = col_name / cash_0 - 1, where col_name is cash, holding or portfolio.
   8.  Model-inferred target is a real number in [0, 1].
   9.  Calculate trade_1 = floor(clip(target_0 * (cash_0 / open_1 + holding_0) - holding_0, -holding_0, cash_0 / open_1)).
   10. Calculate holding_1 = holding_0 + trade_1.
   11. Calculate cash_1 = cash_0 - trade_1 * open_1.
   12. Calculate portfolio_1 as defined before.
   13. Loop the above calculations from index_1 till the end of the simulation data.
   14. Calculate GAE from index_0 to index_{simulation_length - 1}, using portfolio differences between consecutive days as TD residuals. Estimate GAE_{end} assuming a continuous gradient.
   15. Collect experiences from index_0 to index_{simulation_length - 1}.
6. Parallelize model inference across all GPUs.
7. Train model with batched experiences. Use GPU parallelization.
8. Write per-update training result to a log file. Print the latest result in console and update it in place.
9. Save model to "models/" folder.
10. Test model-based strategy against buy-and-hold baseline as follows. Use CPU parallelization for sampling and trade simulation. Use GPU parallelization for model inferece.
   1. Sample 30 (symbol, index_start, index_end) without replacement from all testing data as a group.
   2. Sample 1000 group with replacement.
   3. Simulate trade as in training.
   4. Calculate baseline_return = close_{end} / open_1 - 1.
   5. Calculate model_return = portfolio_{end} / cash_0 - 1.
   6. Record only group_id, baseline_return and model_return for each simulation.
   7. Calculate means of baseline_return and model_return per group. Use each as one observation.
   8. Calculate means and standard deviations of baseline_return and model_return over all groups.
11. Plot training and testing results. Save plots to "models/" folder.
12. Define all parameters and paths at the top of the script.
13. Do not use "=" or "-" for separating sections in script comments or in console output.
15. Name the script "train_model.py".

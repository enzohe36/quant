1. Input data paths are "data/train_tr.csv" and "data/test_tr.csv". Input data has the same format as example.csv. All columns except symbol and date are features. Do not normalize any column.

2. Split both training and testing data by symbol. For each symbol in each data set, keep only those whose num_rows >= required_length, where required_length = simulation_length + sequence_length. Default simulation_length = 240; default sequence_length = 60. Record list(symbol, date) where index(date) >= required_length, and pool by data set. Parallelize this calculation.

3. Sample one list(symbol, date) from all training data. Simulate each trade as follows. Run one such simulation on one CPU core to parallelize the calculation.

   3.1. Let end_date = list(symbol, date)[2]. Use stock data with row number in [index(end_date) - required_length + 1, index(end_date)] for simulation.

   3.2. Let index_0 = sequence_length. Use the following initial values: cash_0 = 100, holding_0 = 0.

   3.3. Standardize all columns with "p_" prefix by dividing with p_close_0. Use standardized values for subsequent calculations.

   3.4. Define portfolio_n = cash_n + holding_n * p_close_n.

   3.5. Decide if the trade is random or not based on current epsilon.

   3.6. If the trade is nonrandom, prepare input data for model inference: scale all columns with "p_" prefix by calculating their percentage change relative to p_close_0; scale cash_0, holding_0 and portfolio_0 by calculating their percentage change relative to cash_0.

   3.7. The model should output list(action_0, certainty_0). Action is an integer. It has a maximum of 3 levels: 0 (no action), 1 (buy), and -1 (sell). The range of action depends on holding: if holding > 0, action is in [-1, 1]; else, action is in [0, 1]. Certainty is a real number in [0, 1].

   3.8.  If the action is random, use uniform probability for sampling action and certainty.

   3.9.  Calculate trade_1 = floor(min(action_0 * certainty_0 * (cash_0 / open_1 + holding_0), ifelse(action_0 == 1, cash_0 / open_1, holding_0))).

   3.10. Calculate holding_1 = holding_0 + trade_1.

   3.11. Calculate cash_1 = cash_0 - trade_1 * open_1.

   3.12. Calculate portfolio_1 as defind before.

   3.13. Loop the above calculations from index_1 till the end of the simulation data.

   3.14. Calculate GAE from index_0 to index_{simulation_length - 1}, using portfolio differences between consecutive days as TD residuals.

   3.15. Collect experiences from index_0 to index_{simulation_length - 1}.

4. Add multi-GPU support for training and model inference. Training workflow is as follows:

   4.1. Parallelize trade simulation on CPU.

   4.2. Perform model inference in batches for nonrandom trades.

   4.3. Collect experiences in RAM.

   4.4. Transfer batched experiences to VRAM; release RAM.

   4.5. Train model with experiences in batches on all GPUs; release VRAM.

5. For testing, sample one list(symbol, date) from all testing data. Simulate trades in the same way as in training, and calculate portfolio_change = portfolio_{end} / cash_0 - 1. Calculate portfolio_change_baseline = close_{end} / open_1 - 1. Average portfolio_change over 30 simulations as one observation, and simulate 1000 observations to calculate mean and sd. Do the same for baseline calculation.

6. Output running time after each function completes, including during each training episode.

7. Use all CPUs and GPUs by default.

8. Plot training and testing results.

9. Output model and plots to "models/" folder.

10. Define all parameters and paths at the top of the script.

11. Do not use "=" or "-" for separating sections in script comments or in console output.
12.
1.  Identify critical bugs and implement best solutions.

2.  Do not write a summary document; output a sumary directy instead.

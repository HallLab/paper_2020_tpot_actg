# Step 3

Use TPOT to optimize pipeline structures.

This portion of the analysis was run on Penn State's ROAR infrastructure.

The `run_permutations.py` script performs a single run of TPOT, outputting:
  * An optimized pipeline
  * A plot showing predicted results using the pipeline (using 5-fold cross-validation) against actual values
  * A log file
  
PBS job files (essentially bash scripts) are used to run this analysis:
  * 100 times
  * 100 times after permuting the phenotype value (the null model).
  
Each pbs job file performs 20 runs of each phenotype, 6 at a time to maximize resources.

`custom_transform` contains a scikit-learn style Transformer class that adds a column of ones to a matrix of features in order to act as an intercept.


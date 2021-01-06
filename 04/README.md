# Step 4

Score the optimized pipeline from each iteration.  For each of the 200 pipelines (100 replicates and 100 permuted):

* Load and score the optimized pipelines from each run
* Perform a Wilcoxon rank sums test to determine if the non-permuted pipelines are significant
* Save structural pipeline information
* Generate feature importances for the top 10 replicate pipelines

Run `reformat_feature_importance.py` to condense the data.
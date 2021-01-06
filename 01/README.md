# Step 1

1. Load the phenotype and binned rare variant data for the 310 samples
2. Drop samples with NA values in either phenotype or a covariate
3. Run a regression for each phenotype (phenotype ~ covariates + 1) and save the residuals
4. Generate some plots
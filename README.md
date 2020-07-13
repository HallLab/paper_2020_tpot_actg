# Automated machine learning for rare variant analysis of response to antiretroviral therapy in persons living with HIV 

Submitted 2020-07-XX

## Abstract 

Motivation: Rare variants pose several challenges for GWAS studies including a lack of statistical power and the inability to model epistasis. In this paper we propose and implement a new approach which uses automated machine learning (AutoML) to generate regression pipelines for binned rare variant data.  This approach is applied to data from the AIDS Clinical Trials Group (ACTG). 

Results: Statistically significant pipelines were generated for all three tested phenotypes.  Permutation importance analysis highlighted several genes important to the accurate prediction of phenotypes, many of which had prior associations with the phenotype. 

Availability: TPOT (Tree-based Pipeline Optimization Tool) was used for this analysis and is freely available: https://epistasislab.github.io/tpot/ 

Contact: jhmoore@upenn.edu  

## Scripts

### get_data.py

A simple command line script to select and format data from the original files

### process_gene_sets.py

A script to convert gene set information from GMT format into a format that is compatible with the TPOT Feature Set Selector.  This also makes plots related to the number of bins and the number of genes in each bin (used for figure 1 in the paper).

### regress_covariates.py

A command line script to regress the phenotypes against the covariates, saving and plotting the residuals to be used in further analysis

### run_tpot_exome_residuals.py

A command line script that does the following:

  1) Load the rare variant data
  2) Load the residuals from regression with covariates
  3) Optimize and save TPOT pipeline based on the 'FeatureSetSelector-Transformer-Regressor' template
  4) Score the optimized pipeline and plot the regression results.

### run_tpot_exome_residuals_permuted.py

The same as above, with an additional line that permutes the phenotype data before running the analysis

### score_pipelines.py

  1) Load and score the optimized pipelines (100 replicate and 100 permuted).  Scoring works the same as in *run_tpot_exome_residuals.py*.
  2) Save pipeline information (structure, selected feature set) and scores (used in figure 3).
  3) Generate feature importances 100 times for the top ten (non-permuted) pipelines and save the results (used in figure 4).

### reformat_feature_importance.py

Save a copy of the feature importances that excludes rows (bins) with all-zero or all-missing feature importances.

### plotting/plot_scores.py

Command line script to generate figure 2 from the paper:

  1) Plot the distribution of scores for replications and permutations.
  2) Perform a t-test of the null hypothesis that the distribution of scores is the same for the original and permuted data.

### plotting/plot_pipeline_diagrams.py

Command line script to plot pipeline diagrams of the steps used in the top 10 pipelines (figure 3 in the paper).

### plotting/plot_feature_importance.py

Command line script to plot feature importances (figure 4 in the paper):

  1) Take the top 10 scoring regression pipelines
  2) Rank the variant bins (which correspond to genes) by the maximum feature importance of that bin in any one of the top 10 pipelines
  3) Plot a histogram of feature importances among the 100 replicates

## Job Files

These are pbs files meant to submit jobs to Penn State's ACI-B computing infrastructure

### create_venv_tpot.pbs

A PBS job script for creating a conda environment with all of the required dependencies.

### run_replicates.pbs

A PBS job script to run the 'run_tpot_exome_residuals.py' script in parallel a number of times.

### run_permutations.pbs

Same as run_replicates, but calling the permutation script instead.

## Supplemental Data

Supplemental data referred to in the paper

  - A figure showing results of testing with different TPOT settings
  - Figures showing the phenotype and residual distributions
  - Tables of feature importance for all bins from the top 10 pipelines

## Pipeline Scores

This folder contains two files for each phenotype (one with the original data and one after permuting the phenotype).  Each file has 100 rows where each row contains information on the structure and score of the optimized pipeline from that run of TPOT.  This is used to generate the pipeline structure diagram (figure 3).

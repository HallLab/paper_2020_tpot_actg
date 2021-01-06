import os

import click
from eli5.sklearn import PermutationImportance
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
from sklearn.model_selection import cross_val_score, cross_val_predict
from scipy.stats import ranksums

# Define transform class here
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class AddIntercept(TransformerMixin, BaseEstimator):
    """
    Add a feature called "Intercept" that consists of all 1s
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """Needed for interface, doesn't actually calculate anything"""
        return self

    def transform(self, X, copy=None):
        """Add an array of ones to be used as the intercept in regression algorithms
        Parameters
        ----------
        X : {array-like, sparse matrix of shape (n_samples, n_features)
            The data that will have an intercept added
        copy : bool, default=None
            Copy the input X or not.
        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features+1)
            Transformed array.
        """
        return np.c_[X, np.ones(X.shape[0])]

def load_data(phenotype):
    """Load data and split into X and y"""
    # Load residuals
    residuals_file = f"../01/{phenotype}_residuals.txt"
    residuals = pd.read_csv(residuals_file, index_col=0, squeeze=True)
    print(f"Loaded {len(residuals)} residuals")

    # Load Exome Data
    variant_count_file = f"../data/Biobin_WES_3647_413_merged_full_SKAT_linear_bin_gene_MB-p_efvcnsgr2w48-bins.csv"
    variant_counts = pd.read_csv(variant_count_file, index_col="ID")
    variant_counts.index.name = "VANTAGENGSID"
    # Drop 1st 9 rows (not actual samples) and the existing phenotype column
    variant_counts = variant_counts.iloc[9:, :].drop('p_efvcnsgr2w48', axis='columns')
    print(f"Loaded {len(variant_counts):,} samples, each with {len(list(variant_counts)):,} bins")

    # Match residuals to data by ID
    df = pd.merge(left=residuals, right=variant_counts, how='inner', left_index=True, right_index=True)

    # Shuffle input so there is no order
    df = df.sample(frac=1, random_state=1855)

    # Split into X and y
    X = df.drop(residuals.name, axis='columns')
    y = df[residuals.name]
    print(f"{len(y):,} samples in the data")

    return X, y


def load_pipeline(pipeline_file: Path):
    # Get replication number
    replicate_num = int(str(pipeline_file.parents[0]).split("_")[-1])
    # Get pipeline object
    env = dict()
    with pipeline_file.open('r') as pf: 
        lines = pf.readlines()
        for idx, line in enumerate(lines):
            if line.startswith("from custom_transform"):
                continue
            elif line.startswith("import") or line.startswith("from"):
                exec(line, globals(), env)
            elif line.startswith("exported_pipeline"):
                pipeline_str = "".join(lines[idx:idx+6])
                # Replace the fss file location in the pipeline
                exec(pipeline_str, globals(), env)
                exported_pipeline = env['exported_pipeline']
                break
    return replicate_num, exported_pipeline


def score_pipeline(pipeline, X, y, cv):
    """Run the pipeline using CV and generate scores"""
    # Shuffle the samples so that the CV is not the same as it was during training
    # Otherwise, scoring would be based on overfit of the data
    X = X.sample(frac=1.0, random_state=2020)
    y = y.sample(frac=1.0, random_state=2020)
    mse_scores = cross_val_score(pipeline, X, y, scoring='neg_mean_squared_error', cv=cv)
    r2_scores = cross_val_score(pipeline, X, y, scoring='r2', cv=cv)
    # Return mean scores
    return mse_scores.mean(), r2_scores.mean()


def get_pipeline_score(pipeline, X, y, cv):
    """Extract information about the pipeline"""
    # Scores
    mse_score, r2_score = score_pipeline(pipeline, X, y, cv)

    # fss
    fss = pipeline.steps[0][1].get_params()['sel_subset']

    transformer = pipeline.steps[1][0]
    transformer_params = ';'.join([f"{k}={v}" for k,v in pipeline.steps[1][1].get_params().items()])
    regressor = pipeline.steps[3][0]
    regressor_params = ';'.join([f"{k}={v}" for k,v in pipeline.steps[3][1].get_params().items()])

    pipeline_info = {
        'MSE Score': mse_score,
        'R^2 Score': r2_score,
        'FSS': fss,
        'Transformer': transformer,
        'Transformer Params': transformer_params,
        'Regressor': regressor,
        'Regressor Params': regressor_params
    }

    return pipeline_info


def get_feature_importance(pipeline, X, y):
    """Get the feature importances using eli5"""
    pipeline.fit(X, y)
    # Limit the features to those selected by the FSS
    features = pipeline.steps[0][1].feat_list
    trimmed_X = X[features]
    perm = PermutationImportance(pipeline, n_iter=100, random_state=1855).fit(trimmed_X, y)
    # Return a dataframe with replicated importances in each column and the bin as the index
    results = {idx+1: result for idx, result in enumerate(perm.results_)}
    df = pd.DataFrame(results, index=features)
    df.index.rename("bin", inplace=True)
    return df


@click.command()
@click.argument('phenotype', type=click.STRING)
@click.option('--permutations',type=click.Path(file_okay=False, dir_okay=True, readable=True))
@click.option('--replications', type=click.Path(file_okay=False, dir_okay=True, readable=True))
@click.option('--cv', type=click.INT, default=5, help="Number of CV groups used in evaluation")
def check_performance(phenotype, permutations, replications, cv):
    """Run test pipelines with cv, plotting results"""
    pemutation_folder = Path(permutations)
    replication_folder = Path(replications)

    # Folders
    current_folder = str(Path(os.path.realpath(__file__)).parent)  # folder containing this script

    # Load genotype data
    X, y = load_data(phenotype)
    print("Loaded genotype data")

    # Load FSS ddata
    fss_df = pd.read_csv("../02/c7.all.v7.0.symbols.csv")

    # Saved columns in order
    FINAL_COLUMNS = ["MSE Score", "R^2 Score", "FSS", "FSS Name", "Transformer", "Regressor", "Transformer Params", "Regressor Params", "FSS Bins"]

    # Score Permuted Results
    data = []
    print(f"Processing permutations")
    for pipeline in pemutation_folder.glob(f"{phenotype}*/*_pipeline.py"):
        replicate_num, pipeline = load_pipeline(pipeline)
        # Save Pipeline info
        y_permuted = y.sample(frac=1.0, random_state=replicate_num)  # Permute data the same way as the trained dataset
        info = get_pipeline_score(pipeline, X, y_permuted, cv)
        info['Permutation'] = replicate_num
        data.append(info)

    permutation_df = pd.DataFrame(data).sort_values('R^2 Score', ascending=False).set_index('Permutation')
    # Add fss info
    permutation_df['FSS Name'] = permutation_df['FSS'].apply(lambda i: fss_df.iloc[i]['Subset'])
    permutation_df['FSS Bins'] = permutation_df['FSS'].apply(lambda i: fss_df.iloc[i]['Features'])
    # Reorder columns
    permutation_df = permutation_df[FINAL_COLUMNS]
    # Save to file
    permutation_df.to_csv(f"{phenotype}_permutation_scores.txt", sep="\t")

    # Load optimized pipeline replicates
    data = []
    pipelines = dict()
    print(f"Processing replicates")
    for pipeline in replication_folder.glob(f"{phenotype}*/*_pipeline.py"):
        replicate_num, pipeline = load_pipeline(pipeline)
        pipelines[replicate_num] = pipeline
        # Save Pipeline info
        info = get_pipeline_score(pipeline, X, y, cv)
        info['Replicate'] = replicate_num
        data.append(info)

    replication_df = pd.DataFrame(data).sort_values('R^2 Score', ascending=False).set_index('Replicate')
    # Add fss info
    replication_df['FSS Name'] = replication_df['FSS'].apply(lambda i: fss_df.iloc[i]['Subset'])
    replication_df['FSS Bins'] = replication_df['FSS'].apply(lambda i: fss_df.iloc[i]['Features'])
    # Reorder columns
    replication_df = replication_df[FINAL_COLUMNS]
    # Save to file
    replication_df.to_csv(f"{phenotype}_replication_scores.txt", sep="\t")

    # Wilcoxon rank sums test
    for score in ["MSE Score", "R^2 Score"]:
        w, pval = ranksums(permutation_df[score], replication_df[score])
        print(f"*** {phenotype} - {score}: W = {w:.3f}, pval = {pval/2:.2E} ***")

    # Feature importance for top 10
    results = []
    for rank, pipeline_number in enumerate(replication_df.head(10).index.values):
        pipeline = pipelines[pipeline_number]
        rank += 1  # start at 1 instead of 0
        feature_importance = get_feature_importance(pipeline, X, y)  # Df with one row per bin and one column per permutation calculation results
        # Add rank and replicate
        feature_importance.columns = pd.MultiIndex.from_tuples([(rank, imp_rep_num) for imp_rep_num in feature_importance.columns], names=['Pipeline Score Rank', 'importance_rep'])
        # Save to list
        results.append(feature_importance)

    # Merge results so that missing bins have NaN
    results = pd.concat(results, axis=1, sort=False)

    # Unstack into one column, name it, and reset the index
    results.index.rename("bin", inplace=True)
    results = results.unstack()
    results.name = "Feature Importance"
    results = results.reset_index()

    # Save to text file
    results.to_csv(f"{phenotype}_feature_importances.txt", sep="\t", index=False)


if __name__ == '__main__':
    check_performance()

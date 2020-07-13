import os

import click
from eli5.sklearn import PermutationImportance
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
from sklearn.model_selection import cross_val_score, cross_val_predict
from scipy.stats import ranksums

def load_exome_data(data_folder, residuals):
    """Load exome genotype data"""
    # Load Exome Data
    df_exome = pd.read_csv(
        f"{data_folder}/Biobin_WES_3647_413_merged_full_SKAT_linear_bin_gene_MB-p_efvcnsgr2w48-bins.csv")
    # Drop 1st 9 rows (not actual samples) and the existing phenotype column
    df_exome = df_exome.loc[9:, :].drop('p_efvcnsgr2w48', axis='columns')

    # Convert Exome IDs
    df_phenos = pd.read_csv(
        f"{data_folder}/2017_12_12_WES_ID_map+2016_11_13_ACTG_MASTER_phenos_commonRPIDandVantageID.txt",
        sep="\t")
    id_dict = df_phenos[['VANTAGENGSID', 'rpid']].set_index('VANTAGENGSID').to_dict()['rpid']
    df_exome['ID'] = df_exome['ID'].apply(lambda s: id_dict[s])
    df_exome = df_exome.set_index("ID")

    # Log
    print(f"Exome data: Loaded {len(df_exome):,} rows and {len(list(df_exome)):,} regions")

    # Match residuals to data by ID
    exome = pd.merge(left=residuals, right=df_exome, how='inner', left_index=True, right_index=True)
    X_exome = exome.drop(residuals.name, axis='columns')
    y_exome = exome[residuals.name]
    exome_samples = len(y_exome)
    print(f"{exome_samples:,} samples in exome data")

    return X_exome, y_exome


def load_pipeline(pipeline_file: Path, data_folder: Path):
    # Get replication number
    replicate_num = int(str(pipeline_file.parents[0]).split("_")[-1])
    # Get pipeline object
    env = dict()
    with pipeline_file.open('r') as pf: 
        lines = pf.readlines()
        for idx, line in enumerate(lines):
            if line.startswith("import") or line.startswith("from"):
                exec(line, globals(), env)
            elif line.startswith("exported_pipeline"):
                pipeline_str = "".join(lines[idx:idx+5])
                # Replace the fss file location in the pipeline
                pipeline_str = pipeline_str.replace("./data", str(data_folder))
                exec(pipeline_str, globals(), env)
                exported_pipeline = env['exported_pipeline']
                break
    return replicate_num, exported_pipeline


def score_pipeline(pipeline, X, y, cv):
    """Run the pipeline using CV and generate scores"""
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
    regressor = pipeline.steps[2][0]
    regressor_params = ';'.join([f"{k}={v}" for k,v in pipeline.steps[2][1].get_params().items()])

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
    data_folder = current_folder + "/data"

    # Load residuals
    residuals_file = data_folder + f"/residuals/{phenotype}_residuals.txt"
    residuals = pd.read_csv(residuals_file, sep="\t", index_col='ID', squeeze=True)
    print("Loaded residuals")

    # Load genotype data
    X, y = load_exome_data(data_folder, residuals)
    print("Loaded genotype data")

    # Load FSS ddata
    fss_df = pd.read_csv(data_folder + '/gene_sets/csv/exome_c7.all.v7.0.symbols.csv')

    # Saved columns in order
    FINAL_COLUMNS = ["MSE Score", "R^2 Score", "FSS", "FSS Name", "Transformer", "Regressor", "Transformer Params", "Regressor Params", "FSS Bins"]

    # Score Permuted Results
    data = []
    print(f"Processing permutations")
    for pipeline in pemutation_folder.glob(f"{phenotype}*/*_pipeline.py"):
        replicate_num, pipeline = load_pipeline(pipeline, data_folder)
        # Save Pipeline info
        info = get_pipeline_score(pipeline, X, y, cv)
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
        replicate_num, pipeline = load_pipeline(pipeline, data_folder)
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
        feature_importance.columns = pd.MultiIndex.from_tuples([(rank, imp_rep_num) for imp_rep_num in feature_importance.columns], names=['Pipline Score Rank', 'importance_rep'])
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
    results.to_csv(f"{phenotype}_feature_importances.txt", sep="\t")


if __name__ == '__main__':
    check_performance()

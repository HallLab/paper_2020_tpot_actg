import os
from itertools import combinations
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_score, cross_val_predict
from tpot import TPOTRegressor
from tpot.config import regressor_config_dict_light


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


def score_pipeline(pipeline, X, y, name, output_folder):
    """Test the pipeline with data"""
    mse_scores = cross_val_score(pipeline, X, y, scoring='neg_mean_squared_error', cv=5)
    r2_scores = cross_val_score(pipeline, X, y, scoring='r2', cv=5)
    print("5-fold Cross-Validation Scores")
    print(f"\tNegative MSE = {mse_scores.mean()} Average ({', '.join([str(n) for n in mse_scores])})")
    print(f"\tR^2 = {r2_scores.mean()} Average ({', '.join([str(n) for n in r2_scores])})")

    # Get CV Predictions
    results = pd.DataFrame({'predicted': cross_val_predict(pipeline, X, y, cv=5),
                            'actual': y})

    # Plot CV Predictions
    ax = sns.scatterplot(x="actual", y="predicted", data=results, alpha=0.5)

    # Add diagonal that is +/- 10% of the min/max range
    min_range = results[['actual', 'predicted']].min(axis=1).min() * 0.9
    max_range = results[['actual', 'predicted']].max(axis=1).max() * 1.1
    ax.plot([min_range, max_range], [min_range, max_range], lw=1, color='black')

    # Title
    ax.set_title(name)

    # Save
    plt.savefig(output_folder + f"/{name}.png")
    print(f"Saved {name} plot")
    plt.clf()  # Clear in case more plots are made later

existing_file = click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)

@click.command()
@click.argument('phenotype', type=click.STRING)
@click.option('--population', type=click.INT, default=100)
@click.option('--generations', type=click.INT, default=None)
@click.option('--gene_set_file', type=existing_file, default=None)  # For FSS
@click.option('--gene_set_count', type=click.INT, default=None)  # For FSS
@click.option('--random_seed', type=click.INT, default=1855)
@click.option('--max_time_mins', type=click.INT, default=None)
@click.option('--checkpoint_folder', type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True))
@click.option('--output_folder', type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True), default=".")
def run_analysis(phenotype, population, generations, gene_set_file, gene_set_count, random_seed, max_time_mins, checkpoint_folder, output_folder):
    # Either generations or max_time_mins must be specified
    if generations is None and max_time_mins is None:
        raise ValueError("Either 'generations' or 'max_time_mins' must be specified")

    # Folders
    current_folder = str(Path(os.path.realpath(__file__)).parent)  # folder containing this script
    data_folder = current_folder + "/data"

    # Load residuals
    residuals_file = data_folder + f"/residuals/{phenotype}_residuals.txt"
    residuals = pd.read_csv(residuals_file, sep="\t", index_col='ID', squeeze=True)
    print("Loaded residuals")

    # Load genotype data
    X_exome, y_exome = load_exome_data(data_folder, residuals)
    print("Loaded genotype data")

    # Load fss options
    if gene_set_file is None:
        raise ValueError("A gene_set_file must be specified when using FSS")
    else:
        gene_set_file = str(gene_set_file)  # Must be string when passed to TPOT
    if gene_set_count is None:
        raise ValueError("A gene_set_count must be specified when using FSS")
    available_subset_num = len(open(gene_set_file).readlines())
    sel_subset = list(range(available_subset_num))
    if gene_set_count > 1:
        # Update sel_subset to be all possible combinations
        sel_subset = list(combinations(sel_subset, gene_set_count))
    fss_options = {'subset_list': [gene_set_file],
                   'sel_subset': sel_subset}
    regressor_config_dict_light['tpot.builtins.FeatureSetSelector'] = fss_options

    # Print what settings are being used
    print(f"Running TPOT")
    print(f"\tPhenotype = {residuals.name}")
    print(f"\tPopulation = {population}")
    # FSS
    print(f"\tFSS settings:")
    print(f"\t\tfile = {gene_set_file} (contains {available_subset_num:,} sets)")
    print(f"\t\t{gene_set_count} set(s) chosen at a time ({len(fss_options['sel_subset']):,} combinations)")

    # Define template
    template = 'FeatureSetSelector-Transformer-Regressor'

    # Optimize Pipeline
    pipeline_optimizer = TPOTRegressor(generations=generations,
                                       population_size=population,
                                       max_time_mins=max_time_mins,
                                       verbosity=2,
                                       config_dict=regressor_config_dict_light,
                                       template=template,
                                       periodic_checkpoint_folder=checkpoint_folder,
                                       early_stop=None,
                                       random_state=random_seed,
                                       n_jobs=-1)
    print("=" * 30)
    print("Starting Training...")
    pipeline_optimizer.fit(X_exome, y_exome)

    # Save pipeline
    pipeline_optimizer.export(output_folder + f"/{phenotype}_pipeline.py")

    # Score pipeline and save results
    pipeline = pipeline_optimizer.fitted_pipeline_
    score_pipeline(pipeline, X_exome, y_exome, name=f"exome_{phenotype}_results", output_folder=output_folder)


if __name__ == '__main__':
    run_analysis()

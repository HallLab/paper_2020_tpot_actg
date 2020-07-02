import click
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

VARIABLE_DESCRIPTIONS = {
    'p_cd4difw48w4': 'Change in Absolute CD4 Count',
    'p_logtbilw0': 'Log10-transformed Biliruben (mg/dL) at Week Zero',
    'vllogdifw48w4': 'Log10-transformed Change in Plasma HIV RNA Copies'
}


def plot_importances(df, phenotype, top_n=0):
    """
    Plots boxplots for feature importance
    """

    # Keep top_n bins ranked by the highest median score of a pipeline in the bin
    median_permutation_scores = df.groupby(['bin', 'Pipeline Score Rank']).apply(lambda g: g['Feature Importance'].median()).unstack()
    max_median = median_permutation_scores.max(axis=1).sort_values(ascending=False)
    kept_bins = max_median.head(top_n).index.values

    # Log
    cutoff = max_median.iloc[top_n]
    print(f"Keeping bins with at least one pipeline median score >= {cutoff:.3f}")

    # Sort according to the bin ranking, keeping only the top n
    bin_sort_idx = {b: idx for idx, b in enumerate(kept_bins)}
    df['bin_rank'] = df['bin'].map(bin_sort_idx)
    df = df[~df['bin_rank'].isna()].sort_values(by='bin_rank')

    # Fillna with 0
    df = df.fillna(0)

    # Draw the boxplots in a grid
    grid = sns.catplot(x="Pipeline Score Rank", y="Feature Importance",
                       data=df, col="bin", order=range(1, 11),
                       col_wrap=4, height=2,
                       sharex=True, sharey=True, kind="box")

    # Adjust the arrangement of the plots
    grid.fig.tight_layout(w_pad=1)

    # Set a title
    suptitle = plt.suptitle(f"{VARIABLE_DESCRIPTIONS[phenotype]}\n"
                            f"Top {top_n} Bins Ranked by Highest Median Pipeline Feature Importances",
                            y=1.05)

    # Save
    grid.savefig(f"plots/{phenotype}_feature_importances.jpg",
                 bbox_extra_artists=(suptitle,), bbox_inches="tight", dpi=350)


@click.command()
@click.argument('phenotype', type=click.STRING)
@click.option('--top_n', type=click.INT, default=10,
              help="Rank bins by their maximum importance score.  Keep the top 'n' of these, plus ties.")
def plot_feature_importance(phenotype, top_n):
    """
    Plot feature importances for the top pipelines
    """
    df = pd.read_csv(f"{phenotype}_feature_importances.txt", sep="\t")
    plot_importances(df, phenotype, top_n)


if __name__ == '__main__':
    plot_feature_importance()

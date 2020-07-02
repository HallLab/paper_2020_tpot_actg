import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind

VARIABLE_DESCRIPTIONS = {
    'p_cd4difw48w4': 'Change in Absolute CD4 Count',
    'p_logtbilw0': 'Log10-transformed Biliruben (mg/dL) at Week Zero',
    'vllogdifw48w4': 'Log10-transformed Change in Plasma HIV RNA Copies'
}


@click.command()
def plot_scores():
    """Plot a histogram of scores.  Rows are phenotypes, columns are scores"""

    phenotypes = sorted(VARIABLE_DESCRIPTIONS.keys())
    #scores = ["MSE Score", "R^2 Score"]
    scores = ["R^2 Score"]

    fig, axes = plt.subplots(len(phenotypes), len(scores), figsize=(6, 8), dpi=350)

    for pheno_idx, phenotype in enumerate(phenotypes):
        # Load Data
        permutation_df = pd.read_csv(f"{phenotype}_permutation_scores.txt",
                                     sep="\t", index_col='Permutation')
        replication_df = pd.read_csv(f"{phenotype}_replication_scores.txt",
                                     sep="\t", index_col='Replicate')

        for score_idx, score in enumerate(scores):
            if len(scores) == 1:
                ax = axes[pheno_idx]
            else:
                ax = axes[pheno_idx, score_idx]

            # Plot Histograms
            sns.distplot(permutation_df[score], rug=True,
                         ax=ax,
                         label="Permuted Pipelines (Null Model)")
            sns.distplot(replication_df[score], rug=True,
                         ax=ax,
                         label="Replicated Pipelines")

            # Statistical Test
            t, pval = ttest_ind(permutation_df[score], replication_df[score])

            # Add test results
            textstr = f"t = {t:.3f}\npval = {pval/2:.2E}"
            properties = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.50, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=properties)

            # Add legend
            ax.legend(loc='upper left')

            # Labels
            ax.set_xlabel(score)
            ax.set_ylabel("density")
            ax.set_title(VARIABLE_DESCRIPTIONS[phenotype])

    # Adjust the arrangement of the plots
    fig.tight_layout(w_pad=1)

    # Set a title
    suptitle = plt.suptitle(f"Distribution of Pipeline Scores", y=1.05)

    # Save figure
    plt.savefig(f"plots/pipeline_scores.jpg",
                bbox_extra_artists=(suptitle,),
                bbox_inches="tight")


if __name__ == '__main__':
    plot_scores()

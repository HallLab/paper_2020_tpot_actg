import click
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

def clean_and_split(df, phenotype, covariates):
    """Extract the phenotype and covariates from the df, do some cleanup, and split them"""
    # Drop NA
    row_has_na = df[[phenotype] + covariates].isna().any(axis='columns')
    df = df[~row_has_na]
    if len(df) == 0:
        raise ValueError(f"All rows have at least one NA value")

    # Split X and y
    y = df[phenotype]
    X = df[covariates]

    # Set sex and race to be categories and rename sex variables
    X = X.astype({'sex': 'category', 'race_wbh': 'category'})
    X.loc[:, 'sex'] = X.loc[:, 'sex'].cat.rename_categories({'male,1': 'male', 'female,2': 'female'})

    # Get dummy variables for sex and race
    sex_dummies = pd.get_dummies(X['sex'], prefix='sex', drop_first=True)
    race_wbh_dummies = pd.get_dummies(X['race_wbh'], prefix='race', drop_first=True)

    X = pd.merge(X, sex_dummies, how='inner', left_index=True, right_index=True)
    X = pd.merge(X, race_wbh_dummies, how='inner', left_index=True, right_index=True)
    X = X.drop(['sex', 'race_wbh'], axis='columns')

    return y, X


@click.command()
@click.argument('input_file', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument('phenotype', type=click.STRING)
def regress_covariates(input_file, phenotype):
    # Always use the same covariates
    covariates = ['age', 'sex', 'race_wbh', 'pc1', 'pc2', 'pc3', 'pc4', 'pc5']

    # Output to the current directory
    output_folder = "."

    # Load data
    df = pd.read_csv(input_file, sep="\t") \
           .rename({'rpid': 'ID'}, axis='columns') \
           .set_index('ID')
    
    # Keep only the first listing when there are duplicate IDs (have different PCs- maybe based on different genetic data?)
    df = df.loc[~df.index.duplicated(keep='first')]

    # Clean and split
    y, X = clean_and_split(df, phenotype, covariates)

    # Save the original data
    y.to_csv(f"{output_folder}/{phenotype}_original.txt", sep="\t", header=True)

    # Regress and get residuals
    regressor = LinearRegression()
    regressor.fit(X, y)
    residuals = y - regressor.predict(X)

    # Save a plot of original data and residuals
    plt.clf()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    # Original
    df[phenotype].hist(ax=axes[0])
    axes[0].set_title(f"{phenotype} original values")
    # Residuals
    residuals.hist(ax=axes[1])
    axes[1].set_title(f"{phenotype} residuals")
    
    # Save plot
    matplotlib.pyplot.savefig(f"{output_folder}/{phenotype}_hist.png")

    # Save residuals
    residuals.to_csv(f"{output_folder}/{phenotype}_residuals.txt", sep="\t", header=True)


if __name__ == '__main__':
    regress_covariates()

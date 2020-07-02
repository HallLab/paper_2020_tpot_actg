import click
import pandas as pd

@click.command()
@click.argument('input_file', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument('phenotype', type=click.STRING)
def load_data(input_file, phenotype):
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

    # Keep only the phenotype and covariates
    df = df[[phenotype] + covariates]

    # Drop NA
    row_has_na = df.isna().any(axis='columns')
    df = df[~row_has_na]
    if len(df) == 0:
        raise ValueError(f"All rows have at least one NA value")

    # Set sex and race to be categories and rename sex variables
    df = df.astype({'sex': 'category', 'race_wbh': 'category'})
    df.loc[:, 'sex'] = df.loc[:, 'sex'].cat.rename_categories({'male,1': 'male', 'female,2': 'female'})

    # Replace sex and race with dummy variables
    sex_dummies = pd.get_dummies(df['sex'], prefix='sex', drop_first=True)
    race_wbh_dummies = pd.get_dummies(df['race_wbh'], prefix='race', drop_first=True)
    df = pd.merge(df, sex_dummies, how='inner', left_index=True, right_index=True)
    df = pd.merge(df, race_wbh_dummies, how='inner', left_index=True, right_index=True)
    df = df.drop(['sex', 'race_wbh'], axis='columns')

    # Save data
    df.to_csv(f"{output_folder}/{phenotype}_data.txt", sep="\t", header=True)


if __name__ == '__main__':
    load_data()

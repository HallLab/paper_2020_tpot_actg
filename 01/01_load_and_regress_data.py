import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

COVARIATES = ["age", "sex", "pc1", "pc2", "pc3", "pc4", "pc5", "batch"]
PHENOTYPES = {"vllogdifw48w4": "Log-fold Change in Viral Load",
              "p_cd4difw48w4": "Change in CD4 Count"}

def main():
    # Load phenotype/covariate information
    phenotype_df = pd.read_table("../data/2017_12_12_WES_ID_map+2016_11_13_ACTG_MASTER_phenos_commonRPIDandVantageID.txt",
                                 sep="\t", index_col=0)
    phenotype_df["batch"] = [s.split('-')[0] for s in phenotype_df.index.values]
    phenotype_df = phenotype_df[COVARIATES + [p for p in PHENOTYPES.keys()]]
    # Drop rows with missing information
    phenotype_df = phenotype_df[~phenotype_df.isna().any(axis='columns')]
    # Clean up categories
    phenotype_df = phenotype_df.astype({'sex': 'category', 'batch': 'category'})  # Make categorical
    phenotype_df['sex'] = phenotype_df['sex'].cat.rename_categories({'male,1': 'male', 'female,2': 'female'})
    sex_dummies = pd.get_dummies(phenotype_df['sex'], prefix='sex', drop_first=True)
    batch_dummies = pd.get_dummies(phenotype_df['batch'], prefix='batch', drop_first=True)
    phenotype_df = pd.merge(phenotype_df, sex_dummies, how='inner', left_index=True, right_index=True)
    phenotype_df = pd.merge(phenotype_df, batch_dummies, how='inner', left_index=True, right_index=True)
    phenotype_df = phenotype_df.drop(['sex', 'batch'], axis='columns')

    # Run regressions, saving residual values
    for phenotype in PHENOTYPES.keys():
        y = phenotype_df[phenotype]
        X = phenotype_df.drop(columns=[phenotype,])

        # Save the original data
        y.to_csv(f"{phenotype}_original.txt", sep="\t", header=True)
        X.to_csv(f"{phenotype}_covariates_original.txt", sep="\t", header=True)

        # Regress and get residuals
        regressor = LinearRegression()
        regressor.fit(X, y)
        residuals = y - regressor.predict(X)

        # Save residuals
        residuals.to_csv(f"{phenotype}_residuals.txt")
        print(f"Saved {len(residuals)} residuals for {phenotype}")

        # Save a plot
        fig, axes = plt.subplots(2, figsize=(12, 9))
        y.hist(ax=axes[0], bins=50)
        residuals.hist(ax=axes[1], bins=50)
        axes[0].set_title(f"{PHENOTYPES[phenotype]}")
        axes[0].set_xlabel("original")
        axes[1].set_xlabel("residuals")
        for ax in axes:
            ax.set_ylim(0, 25)
            ax.set_ylabel('count')
        plt.savefig(f"{phenotype}_hist.png")



if __name__ == "__main__":
    main()
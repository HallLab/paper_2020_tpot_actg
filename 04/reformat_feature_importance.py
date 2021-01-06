import pandas as pd


def process_file(filename):
    print(f"Processing {filename}")
    df = pd.read_table(filename).reset_index(drop=True)
    df = df.set_index(['bin', 'Pipeline Score Rank', 'importance_rep']).unstack()
    rows = len(df)
    # Drop all-NaN rows
    df = df.dropna(axis=0, how='all')
    rows2 = len(df)
    # Remove all-zero rows
    df = df.loc[(df != 0).any(1)]
    rows3 = len(df)
    # Log
    print(f"Kept {rows3:,} of {rows:,} rows after removing\n\t{rows-rows2:,} all-NaN rows\n\t{rows2-rows3:,} all-zero rows")
    print("="*40)
    df.columns = [f"Importance Rep {n}" for n in range(1, 101)]
    df.to_csv(filename[:-4] + "_reformatted.txt", sep="\t")


if __name__ == "__main__":
    process_file("p_cd4difw48w4_feature_importances.txt")
    process_file("vllogdifw48w4_feature_importances.txt")
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd
import seaborn as sns


def load_bin_names():
    filename = "../data/Biobin_WES_3647_413_merged_full_SKAT_linear_bin_gene_MB-p_efvcnsgr2w48-bins.csv"
    bins = list(pd.read_csv(filename, nrows=0))[2:]  # Skip "ID" and "p_efvcnsgr2w48"
    print(f"{len(bins):,} bins in exome data")
    return bins


def process_gene_set(gmt_file: Path):
    # Get a list of bins in the data
    gene_bins = load_bin_names()
    gene_sets_name = gmt_file.stem
    output_file = f"{gene_sets_name}.csv"

    # Process the gmt file into a format usable by TPOT
    with open(gmt_file, 'r') as f, open(output_file, 'w') as o:
        o.write('Subset,Size,Features\n')  # Add header to output

        # Track numbers
        bin_set_map = defaultdict(list)  # Which sets contain each bin
        set_bin_map = dict()  # Which genes are in the set

        # Process input
        for line in f:
            # Read info
            fields = line.strip().split('\t')
            name = fields[0]
            genes = set(fields[2:])

            # Determine which genes are in the data and track sets
            matched_bins = genes & set(gene_bins)
            for mb in matched_bins:
                bin_set_map[mb].append(name)
            set_bin_map[name] = matched_bins

            # Save set results
            num_bins = len(matched_bins)
            if num_bins > 0:
                matched_bins_list = sorted(list(matched_bins))
                o.write(f"{name},{len(matched_bins_list)},{';'.join(matched_bins_list)}\n")

    # Plot count info
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), dpi=150)
    
    # Top = cumulative histogram of >= that many sets contain the gene
    sets_per_bin = pd.Series({b:len(bin_set_map[b]) for b in gene_bins})
    sns.histplot(data=sets_per_bin, ax=axes[0], bins=range(1, sets_per_bin.max()+1))
    axes[0].set_ylabel("Gene Bins")
    axes[0].set_xlabel("Number of Sets Containing the Gene Bin")
    axes[0].set_title(f"{(sets_per_bin!=0).sum():,} of {len(sets_per_bin):,} gene bins were found in at least one set")

    # Bottom = cumulative histogram of >= that many gene bins in the set
    bins_per_set = pd.Series({s:len(bl) for s, bl in set_bin_map.items()})
    sns.histplot(data=bins_per_set, ax=axes[1], bins=range(bins_per_set.min(), bins_per_set.max()+1))
    axes[1].set_ylabel("Sets")
    axes[1].set_xlabel("Number of Gene Bins Contained in the Set")
    
    # Set title
    title = f"C7 Immunological Signatures Collection\n{len(set_bin_map):,} Sets\n"
    suptitle = fig.suptitle(title, y=1.02)
    # Save plot and clear
    fig.tight_layout(w_pad=1)
    plt.savefig(f"{gene_sets_name}.png",
                bbox_extra_artists=(suptitle,),
                bbox_inches="tight")


if __name__ == "__main__":
    gmt_file = Path("c7.all.v7.0.symbols.gmt")
    process_gene_set(gmt_file)

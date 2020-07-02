from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd


data_folder = Path("C:/Users/jrm5100/Documents/Projects/ACTG_TPOT/data")


def load_exome_bin_names():
    exome_file = data_folder / "Biobin_WES_3647_413_merged_full_SKAT_linear_bin_gene_MB-p_efvcnsgr2w48-bins.csv"
    exome_bins = list(pd.read_csv(exome_file, nrows=0))[2:]  # Skip "ID" and "p_efvcnsgr2w48"
    print(f"{len(exome_bins):,} bins in exome data")
    return exome_bins


def process_gmt_file(gmt_file, output_file, bin_names):
    """Process a GMT file to save the format needed by TPOT FSS"""
    set_count = 0  # How many sets of genes
    saved_gene_count = Counter()  # How many sets each gene is part of
    skipped_gene_count = Counter()  # How many times a gene was in a set but not in the data

    # Add header to output file
    o = open(output_file, 'w')
    o.write('Subset,Size,Features\n')

    # Number of Sets
    set_count = 0

    # Number of genes per set
    set_size_counts = []

    # Number sets each gene appears in
    sets_per_gene_counts = Counter()

    # Process GMT file
    with open(gmt_file, 'r') as f:
        for line in f:
            # Read info
            fields = line.strip().split('\t')
            name = fields[0]
            genes = set(fields[2:])

            # Determine which genes are in the data
            matched_genes = genes & set(bin_names)
            sets_per_gene_counts.update(matched_genes)
            set_size_counts.append(len(matched_genes))

            # Save results
            matched_genes_list = sorted(list(matched_genes))
            o.write(f"{name},{len(matched_genes_list)},{';'.join(matched_genes_list)}\n")
            set_count += 1

    # Close output file
    o.close()

    # Histograms
    fig, axes = plt.subplots(2, 1, figsize=(5, 4), dpi=350)
    # Plotting params
    max_sets_per_gene = 5
    # Sets per gene
    counts = [sets_per_gene_counts[b] for b in bin_names]
    his_counts, his_labels = np.histogram(counts, bins=range(0, max_sets_per_gene + 2))
    # Add >max values to the final bin
    his_counts[-1] += len([c for c in counts if c > max_sets_per_gene])
    # Change his to percent instead of count
    his_counts = 100 * (his_counts / len(bin_names))
    # Plot histogram
    axes[0].bar(his_labels[1:], his_counts, align='center')
    axes[0].set_xticks(his_labels[1:])
    labels = [str(n) for n in range(0, max_sets_per_gene)] + [f">={max_sets_per_gene}"]
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Percent of Bins")
    axes[0].set_xlabel("Number of Sets Including the Matched Bin")
    axes[0].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}%'))
    axes[0].set_ylim(0, 100)
    # Genes per set
    axes[1].hist(set_size_counts, bins=range(0, 500))
    axes[1].set_ylabel("Number of Sets")
    axes[1].set_xlabel("Number of Matched Bins in the Set")
    # Set title
    set_name = Path(gmt_file).stem
    if set_name == "c7.all.v7.0.symbols":
        title = f"C7 Immunological Signatures Collection"
    else:
        title = set_name
    title += f"\n{set_count:,} Sets"
    suptitle = fig.suptitle(title, y=1.05)
    # Save plot and clear
    fig.tight_layout(w_pad=1)
    plt.savefig(data_folder / "gene_sets" / "plots" / f"{set_name}.jpg",
                bbox_extra_artists=(suptitle,),
                bbox_inches="tight")
    plt.clf()


def process_gene_sets():
    # Get a list of bins in the data
    exome_bins = load_exome_bin_names()

    # Process GMT files into CSV Files
    gmt_file_folder = data_folder / "gene_sets" / "gmt"
    for f in gmt_file_folder.iterdir():
        if f.is_file() and (f.suffix == '.gmt'):
            # Get the name
            name = f.stem
            output = data_folder / "gene_sets" / "csv" / f"{name}.csv"
            print("="*30 + f"\n{name}\n" + "="*30)
            # Process it
            process_gmt_file(str(f), output, exome_bins)


if __name__ == "__main__":
    process_gene_sets()

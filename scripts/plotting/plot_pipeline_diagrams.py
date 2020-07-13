import math

import click
import pandas as pd
import pydot

VARIABLE_DESCRIPTIONS = {
    'p_cd4difw48w4': 'Change in Absolute CD4 Count',
    'p_logtbilw0': 'Log10-transformed Biliruben (mg/dL) at Week Zero',
    'vllogdifw48w4': 'Log10-transformed Change in Plasma HIV RNA Copies'
}

COLORS = ["#0f3d68", "#115473", "#146e7d", "#178784", "#1a9079",
          "#2ea045", "#56a63f", "#8dac4f", "#b3ab60", "#b99370"]

PATH_WIDTH = [2.8, 2.6, 2.4, 2.2, 2.0, 1.8, 1.6, 1.4, 1.2, 1.0]
PATH_WIDTH = [n*2 for n in PATH_WIDTH]

TITLE_FONTSIZE=50
TEXT_FONTSIZE=40


def format_fss_name(original):
    """
    format the FSS name to be more clear
    """
    sections = original.split('_')
    direction = sections[-1]

    code = sections[0]

    left, right = ' '.join(sections[1:-1]).split('VS')
    if sections[-1] == "DN":
        direction = '<FONT COLOR="#943126">downregulated</FONT>'
    else:
        direction = '<FONT COLOR="#196f3d">upregulated</FONT>'

    result = f"< {code}<BR/>{left}vs{right}<BR/>{direction} >"
    return(result)


def format_score(original):
    """
    Consistently format the score as a string
    """
    return f"{original:.3f}"

@click.command()
@click.argument('phenotype', type=click.STRING)
def plot_pipeline_diagrams(phenotype):
    """
    Plot feature importances for the top n pipelines
    """
    # Read data
    n = 10
    replication_results = pd.read_table(f'{phenotype}_replication_scores.txt', sep="\t").head(n)

    # Wider ranksep for the more convoluted result
    if phenotype == "p_cd4difw48w4":
        ranksep = 4
    else:
        ranksep = 2

    # Set up graph
    graph = pydot.Dot(graph_type='digraph', rankdir="LR", ranksep=ranksep, nodesep=0.02,
                      label=f"{VARIABLE_DESCRIPTIONS[phenotype]}", labelloc="t", fontsize=TITLE_FONTSIZE)

    # Set up clusters
    cluster_fss = pydot.Cluster('fss', label='Feature Set Selector', rank="same", penwidth=0)
    cluster_transformer = pydot.Cluster('transformer', label='Transformer', rank="same", penwidth=0)
    cluster_regressor = pydot.Cluster('regressor', label='Regressor', rank="same", penwidth=0)
    cluster_score = pydot.Cluster('score', label='R^2 Score', rank="same", penwidth=0)

    # Add clusters
    graph.add_subgraph(cluster_fss)
    graph.add_subgraph(cluster_transformer)
    graph.add_subgraph(cluster_regressor)
    graph.add_subgraph(cluster_score)

    # Setup representative nodes and add them to their clusters
    cluster_fss_node = pydot.Node('cluster_fss', style='invis', shape='point')
    cluster_fss.add_node(cluster_fss_node)
    cluster_transformer_node = pydot.Node('cluster_transformer', style='invis', shape='point')
    cluster_transformer.add_node(cluster_transformer_node)
    cluster_regressor_node = pydot.Node('cluster_regressor', style='invis', shape='point')
    cluster_regressor.add_node(cluster_regressor_node)
    cluster_score_node = pydot.Node('cluster_score', style='invis', shape='point')
    cluster_score.add_node(cluster_score_node)

    # Link Clusters via their representative nodes
    graph.add_edge(pydot.Edge(cluster_fss_node, cluster_transformer_node, style="invisible", arrowhead="none", weight=1000))
    graph.add_edge(pydot.Edge(cluster_transformer_node, cluster_regressor_node, style="invisible", arrowhead="none", weight=1000))
    graph.add_edge(pydot.Edge(cluster_regressor_node, cluster_score_node, style="invisible", arrowhead="none", weight=1000))

    # Create Nodes
    fss_nodes = []
    for fss in replication_results['FSS Name'].unique():
        node = pydot.Node(fss, label=format_fss_name(fss), shape='box', style='rounded', fontsize=TEXT_FONTSIZE)
        cluster_fss.add_node(node)
        fss_nodes.append(node)
    transformer_nodes = []
    for transformer in replication_results['Transformer'].unique():
        node = pydot.Node(transformer, fontsize=TEXT_FONTSIZE)
        cluster_transformer.add_node(node)
        transformer_nodes.append(node)
    regressor_nodes = []
    for regressor in replication_results['Regressor'].unique():
        node = pydot.Node(regressor, fontsize=TEXT_FONTSIZE)
        cluster_regressor.add_node(node)
        regressor_nodes.append(node)

    # Create score nodes from min score to max score, marking every 0.001
    max_score = math.ceil(replication_results['R^2 Score'].max() * 100) / 100
    min_score = math.floor(replication_results['R^2 Score'].min() * 100) / 100
    last = None

    # Iterate through a range of scores using integers
    i = max_score * 1000
    while i >= (min_score * 1000):    
        score = format_score(i/1000)
        if i % 10 == 0:
            node = pydot.Node(score, shape="plain", label=score, fontsize=TEXT_FONTSIZE)
        else:
            node = pydot.Node(score, shape="point")
        cluster_score.add_node(node)
        # Decrement
        i -= 1
        # Add edge
        if last is not None:
            cluster_score.add_edge(pydot.Edge(last, node, penwidth=0.5, constraint="false", arrowhead="none", len=0.01))
        last = node

    # Add each pipeline
    for idx, row in replication_results.iterrows():
        fss = row['FSS Name']
        transformer = row['Transformer']
        regressor = row['Regressor']
        score = format_score(row['R^2 Score'])
        color = COLORS[idx]
        penwidth = PATH_WIDTH[idx]
        graph.add_edge(pydot.Edge(fss, transformer, color=color, label=str(idx+1), penwidth=penwidth, constraint="false", fontsize=TEXT_FONTSIZE))
        graph.add_edge(pydot.Edge(transformer, regressor, color=color, label=str(idx+1), penwidth=penwidth, constraint="false", fontsize=TEXT_FONTSIZE))
        graph.add_edge(pydot.Edge(regressor, score, color=color, label=str(idx+1), penwidth=penwidth, constraint="false", fontsize=TEXT_FONTSIZE))

    graph.write_png(f"plots/{phenotype}_pipeline_diagram.png")
    graph.write_svg(f"plots/{phenotype}_pipeline_diagram.svg")

if __name__ == '__main__':
    plot_pipeline_diagrams()

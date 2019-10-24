import itertools
import matplotlib.pyplot as plt
import networkx as nx


def get_diff_pairs(
        golden_pairs_set,
        dedupe_found_pairs_set,
        dedupe_unclustered_found_pairs_set,
        diff_set_ids
):
    diff_dedupe_clustered_pairs = [
        (x, y) for x, y in dedupe_found_pairs_set
        if x in diff_set_ids or y in diff_set_ids]
    diff_dedupe_unclustered_pairs = [
        (x, y) for x, y in dedupe_unclustered_found_pairs_set
        if x in diff_set_ids or y in diff_set_ids]
    diff_true_pairs = [
        (x, y) for x, y in golden_pairs_set
        if x in diff_set_ids or y in diff_set_ids]
    diff_all_ids = set(itertools.chain.from_iterable(
        diff_dedupe_clustered_pairs +
        diff_dedupe_unclustered_pairs +
        diff_true_pairs))
    return (
        diff_dedupe_clustered_pairs,
        diff_dedupe_unclustered_pairs,
        diff_true_pairs,
        diff_all_ids
    )


def draw_pairs_graph(df, edges, nodes, edge_labels_dict, title):
    G = nx.Graph()
    for node in nodes:
        G.add_node(node,
                   name=str(node) + ':' + df.loc[node]['name'])
    G.add_edges_from(edges)

    plt.figure(figsize=(10, 6))
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, alpha=0.3, node_size=1000)
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'name'), font_size=20)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=4)
    edge_labels = {pair: edge_labels_dict[pair] for pair in edges
                   if pair in edge_labels_dict}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=20)
    plt.margins(0.5, 0.2)
    plt.axis('off')
    plt.title(title, fontdict={'fontsize': 24, 'fontweight': 'bold'})
    plt.show()


def show_cluster_graphs(
        df,
        golden_pairs_set,
        dedupe_found_pairs_set,
        dedupe_unclustered_found_pairs_set,
        dedupe_unclustered_pairs_score_dict,
        diff_set_ids
):
    (
        diff_dedupe_clustered_pairs,
        diff_dedupe_unclustered_pairs,
        diff_true_pairs,
        diff_all_ids
    ) = get_diff_pairs(
        golden_pairs_set,
        dedupe_found_pairs_set,
        dedupe_unclustered_found_pairs_set,
        diff_set_ids
    )
    display(df.loc[list(diff_all_ids)])
    draw_pairs_graph(
        df, diff_true_pairs, diff_set_ids, {}, "Truth")
    draw_pairs_graph(
        df, diff_dedupe_unclustered_pairs, diff_set_ids, dedupe_unclustered_pairs_score_dict, "Unclustered")
    draw_pairs_graph(
        df, diff_dedupe_clustered_pairs, diff_set_ids, dedupe_unclustered_pairs_score_dict, "Clustered")

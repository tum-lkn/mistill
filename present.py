import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import numpy as np
import os
import pandas as pd
import networkx as nx
import h5py
from typing import Dict, Any, Union, List, Tuple
import logging
logging.basicConfig(level=logging.INFO)
# COLORS = ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3']
COLORS = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e']
COLORS = ['#80b1d3','#fb8072','#fdb462','#bebada','#8dd3c7']
# COLORMAP = plt.set_cmap('Set2')
# plt.set_cmap(COLORMAP)
matplotlib.rcParams.update({'font.size': 8})
matplotlib.rcParams.update({'font.family': 'serif'})
MARKER = ['s', 'o', 'v', '^', '<', '>']


# mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=COLORS)


def get_fig(ncols: float, aspect_ratio=0.618) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create figure and axes objects of correct size.

    Args:
        ncols (float): Percentage of one column in paper.
        aspect_ratio (float): Ratio of width to height. Default is golden ratio.

    Returns:
        fig (plt.Figure)
        ax (plt.Axes)
    """
    COLW = 3.45
    ax = plt.subplot()
    fig = plt.gcf()
    fig.set_figwidth(ncols * COLW)
    fig.set_figheight(ncols * COLW * aspect_ratio)
    return fig, ax


def matrix_to_tikz(matrix: np.array, full_file_path: str, vmin=0., vmax=1.) -> None:
    node_template = "\t\\node[minimum width=1cm, minimum height=1cm, " +\
                    "fill={color}] (a-{i}-{j}) at ({x}, {y}) {{}};"
    picture = "\\begin{{tikzpicture}}\n{body}\n\end{{tikzpicture}}"

    norm = matplotlib.colors.Normalize(vmin=-0, vmax=1)
    cmap = cm.Reds
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    bins = np.linspace(vmin, vmax, 9)

    sep = 0
    nodes = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            r, g, b, _ = mapper.to_rgba(matrix[i, j])
            r, g, b, _ = cm.Greens(matrix[i, j])
            nodes.append(node_template.format(
                color="seqGreen{:d}".format(np.argmin(np.abs(bins - matrix[i, j]))),
                i=i,
                j=j,
                x=j + j * sep,
                y=-1. * i - i * sep
            ))
    figure = picture.format(body="\n\t".join(nodes))
    with open(full_file_path, "w") as fh:
        fh.write(figure)


def plot_sequence(seq: Union[np.array, List[float]], ylabel: str, figname: str):
    """
    Plot a sequence.

    Args:
        seq: Sequence of values that should be plotted.

    Returns:
        None
    """
    ax = plt.subplot()
    ax.plot(np.arange(len(seq)), seq)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Step")
    plt.savefig("{:s}.pdf".format(figname), format='pdf')
    plt.close('all')


def plot_errbars(x: List[np.array], y: List[np.array], yerr: List[np.array],
                 labels: List[str], folder: str, filename: str, xlabel: str, ylabel: str):
    fig, ax = get_fig(0.66)
    for i in range(len(y)):
        # ax.errorbar(x[i], y[i], yerr=yerr[i], c=COLORS[i], label=labels[i], marker=MARKER[i])
        ax.fill_between(x[i], y[i] - yerr[i], y[i] + yerr[i], color=COLORS[i], alpha=0.5, linewidth=0)
        ax.plot(x[i], y[i], marker=MARKER[i], label=labels[i], c=COLORS[i])
    ax.legend(frameon=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x[0])
    ax.set_xticklabels([8, 16, 32, 64])
    save_fig(folder, filename, 'pdf')


def plot_sequences(sequences: List[pd.Series], labels: List[str], folder: str,
                   filename: str, ylabel: str, xlabel: str, markevery=20):
    fig, ax = get_fig(0.66)
    for i, seq in enumerate(sequences):
        ax.plot(seq.index.values, seq.values, label=labels[i], c=COLORS[i],
                marker=MARKER[i], markevery=markevery)
    ax.legend(frameon=False, ncol=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    save_fig(folder, filename, 'pdf')


def plot_training_progress(progress: pd.DataFrame):
    fig, ax = get_fig(1)
    progress = progress.iloc[np.arange(0, progress.shape[0], 5)]
    ax.plot(progress.index.values, progress['cross_entropy_ecmp-val'].values,
            label='ECMP')
    ax.plot(progress.index.values, progress['cross_entropy_single-val'].values,
            label='SCP')
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")
    ax.legend(frameon=False)
    save_fig('img', 'scp-training', 'pdf', fig)


def plot_losses(losses: np.array):
    """
    Plot multiple loss functions.

    Args:
        losses: Development of loss values over multiple trials. First axis is
            steps, second axis is trials. Has shape (num_steps, num_trials).

    Returns:

    """
    x = np.arange(losses.shape[0])
    ax = plt.subplot()
    for i in range(losses.shape[1]):
        ax.plot(x, losses[:, i], color='blue', alpha=0.5)
    plt.savefig("loss.pdf", format="pdf")
    plt.close("all")


def _extract_coordinates(graph: nx.Graph) -> Dict[Any, np.array]:
    """
    Try to get longitude and latitude values from the graph. If a node does
    note have coordinates use graphiviz to generate the positions.

    Args:
        graph: Graph for which node positions should be found.

    Returns:
        pos: Dictionary mapping node names to positions on a plane.
    """
    pos = {}
    for n, d in graph.nodes(data=True):
        if "Latitude" in d and "Longitude" in d:
            pos[n] = np.array([d["Latitude"], d["Longitude"]])
        else:
            logging.info("{} has not coordinates".format(n))
            pos = nx.nx_agraph.graphviz_layout(graph, prog='neato')
            break
    return pos


def plot_graph(graph: nx.Graph, mapping: Dict[Any, int], dist: np.array):
    """
    Plot the graph on a rectangular canvas and color nodes based on the distance
    of the nodes to one specific node. This functions creates as many plots as
    there are nodes in the graph.

    Args:
        graph: Graph that should be visualized.
        mapping: A mapping of node identifiers ot indices for distance matrix `dist`.
        dist: Numpy array with distance values for each node.

    Returns:
        None
    """
    pos = _extract_coordinates(graph)
    max_dist = np.max(dist)
    print("max_dist is {}".format(max_dist))
    dist = max_dist - dist
    v_min = np.min(dist)
    v_max = np.max(dist)
    norm = matplotlib.colors.Normalize(vmin=v_min, vmax=v_max, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Greens)
    for i, u in enumerate(graph.nodes()):
        ax = plt.subplot()
        fig = plt.gcf()
        fig.set_figheight(fig.get_figwidth())
        if graph.graph.get("type", None) == 'FatTree':
            fig.set_figwidth(8 * fig.get_figwidth())
        nx.draw_networkx_edges(graph, pos=pos, ax=ax)
        nodes = list(graph.nodes())
        colors = [mapper.to_rgba(dist[mapping[u], mapping[v]]) for v in nodes]
        labels = {n: 'N-{}'.format(n) for n in graph.nodes()}
        nx.draw_networkx_nodes(graph, pos=pos, nodes=nodes, node_color=colors, ax=ax, edgecolors='black')
        # nx.draw_networkx_labels(graph, pos=pos, labels=labels)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("/opt/project/img/n-{}.svg".format(u), bbox_inches='tight', format='svg')
        plt.close("all")


def print_embedding(embedding: np.array):
    """
    Print out the embeddings of the nodes.

    Args:
        embedding: Array of shape (num_nodes, dim).

    Returns:
        None
    """
    for i in range(embedding.shape[0]):
        print("{:3d}:  ".format(i), end="")
        for j in range(embedding.shape[1]):
            print("{:d}".format(int(embedding[i, j])), end=' ')
        print()


def plot_embedding(embedding: np.array):
    """
    Plot the embedding.

    Args:
        embedding: Array of shape (num_nodes, dim).

    Returns:
        None
    """
    ax = plt.subplot()
    fig = plt.gcf()
    fig.set_figheight(fig.get_figwidth() * embedding.shape[0] / embedding.shape[1])
    ax.imshow(embedding, cmap='Greens')
    ax.set_yticks(np.arange(-0.5, embedding.shape[0] - 0.5, 1))
    ax.set_xticks(np.arange(-0.5, embedding.shape[1] - 0.5, 1))
    ax.set_yticklabels(np.arange(embedding.shape[0]))
    ax.set_xticklabels([])
    plt.grid(color='black')
    plt.tight_layout()
    plt.savefig("/opt/project/img/embedding.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_hierarchy(top_level):
    top_level_graph = top_level.graph
    pos = _extract_coordinates(top_level_graph)

    def plot_graph(graph, colors, fig_name):
        ax = plt.subplot()
        fig = plt.gcf()
        fig.set_figheight(fig.get_figwidth())
        if graph.graph.get("type", None) == 'FatTree':
            fig.set_figwidth(8 * fig.get_figwidth())
        nx.draw_networkx_edges(graph, pos=pos, ax=ax)
        nx.draw_networkx_nodes(graph, pos=pos, node_color=colors, ax=ax, edgecolors='black')
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("/opt/project/img/{}.png".format(fig_name), bbox_inches='tight', format='png')
        plt.close("all")

    def plot_level(level, fig_name):
        if level.negative_group is None and level.positive_group is None:
            return

        colors = []
        for i, n in enumerate(top_level_graph.nodes()):
            if level.negative_group is not None and n in level.negative_group.graph:
                colors.append('red')
            elif level.positive_group is not None and n in level.positive_group.graph:
                colors.append('blue')
            else:
                colors.append('white')

        plot_graph(top_level_graph, colors, fig_name)
        if level.negative_group is not None:
            plot_graph(level.negative_group.graph, 'red', fig_name + '-neg-only')
        if level.positive_group is not None:
            plot_graph(level.positive_group.graph, 'blue', fig_name + '-pos-only')

        if level.negative_group is not None:
            plot_level(level.negative_group, fig_name + '-neg')
        if level.positive_group is not None:
            plot_level(level.positive_group, fig_name + '-pos')
    plot_level(top_level, 'top')


def plot_cdf(cdf: Union[np.array, pd.Series], ax=None, alpha=1) -> plt.Axes:
    if type(cdf) == np.ndarray:
        cdf = pd.Series(cdf).value_counts()
        cdf.sort_index(inplace=True)
        cdf /= cdf.sum()
        cdf = np.cumsum(cdf)
    if ax is None:
        fig, ax = get_fig(1)
    ax.plot(cdf.index.values, cdf.values, alpha=alpha)
    return ax


def compare_cdfs(cdfs: List[List[pd.Series]], xlabel: str, ylabel: str,
                 labels: List[str], alpha: float, ax=None, markevery=100) -> plt.Axes:
    assert len(cdfs) == len(labels)
    if ax is None:
        fig, ax = get_fig(1)
    for i, tmp_cdfs in enumerate(cdfs):
        for j, cdf in enumerate(tmp_cdfs):
            ax.plot(
                cdf.index.values,
                cdf.values,
                color=COLORS[i],
                label=labels[i] if j == 0 else None,
                alpha=alpha,
                marker=MARKER[i],
                markevery=markevery
            )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    return ax


def plot_attention_scores_fat_tree(scores: pd.DataFrame, ax=None, clim=None):
    def weight(x):
        parts = x.split('-')
        num = parts[1].strip('0')
        num = int(num) if len(num) > 0 else 0
        if parts[0].startswith('h'):
            weight = num
        elif parts[0].startswith('t'):
            weight = 10000 + num
        elif parts[0].startswith('a'):
            weight = 20000 + num
        else:
            weight = 30000 + num
        return weight

    if ax is None:
        # fig, ax = get_fig(16)
        fig, ax = get_fig(4)
        fig.set_figheight(fig.get_figwidth())

    perm = scores.index.values.tolist()
    perm.sort(key=weight)
    perm_c = [x for x in scores.columns if not x.startswith('h')]
    perm_c.sort(key=weight)
    # scores = scores.loc[perm, perm_c]

    labels = {'h': 'hosts', 'a': 'aggregation', 't': 'ToRs', 'c': 'cores'}
    cs = perm[0][0]
    sections = []
    for idx, p in enumerate(perm):
        if len(sections) == 0:
            sections.append((idx, labels[p[0]]))
        elif labels[p[0]] == sections[-1][1]:
            continue
        else:
            sections.append((idx, labels[p[0]]))
    vmax = scores.max().max() if clim is None else clim
    print(f"vmax is {vmax}")
    print(scores)
    cax = ax.imshow(scores.values, cmap=cm.get_cmap("Greens"), vmin=0, vmax=vmax)
    # ax.set_yticks(np.arange(-0.5, scores.shape[0], 1))
    # ax.set_xticks(np.arange(-0.5, scores.shape[1], 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # horizontal lines
    # ax.plot([-1, scores.shape[0]], [sections[0][0] - 0.5, sections[0][0] - 0.5], color='black')
    ax.plot([-1, scores.shape[0]], [sections[1][0] - 0.5, sections[1][0] - 0.5], color='black')
    ax.plot([-1, scores.shape[0]], [sections[2][0] - 0.5, sections[2][0] - 0.5], color='black')

    # vertical lines
    # ax.plot([sections[1][0] - 0.5, sections[1][0] - 0.5], [-1, scores.shape[0]], color='black')
    ax.plot([sections[1][0] - 0.5 - sections[0][0], sections[1][0] - 0.5 - sections[0][0]], [-1, scores.shape[0]], color='black')
    ax.plot([sections[2][0] - 0.5 - sections[0][0], sections[2][0] - 0.5 - sections[0][0]], [-1, scores.shape[0]], color='black')
    plt.subplots_adjust(-0.055, -0.055, 1.05, 1.05, 0, 0)
    # ax.set_xlim(-0.5, scores.shape[1])
    # ax.set_ylim(scores.shape[0], 0.5)

    # plt.grid(color='black')
    # m = cm.ScalarMappable(cmap=cm.get_cmap("Greens"))
    # m.set_array(scores.values)
    # m.set_clim(0., scores.max().max() if clim is None else clim)
    # plt.colorbar(m)
    return ax


def plot_converged_attention_scores(path_to_df: str, folder: str, name: str,
                                    file_format: str, fontsize=20, converged_df=None):
    if converged_df is None:
        num_heads, head_names = _get_head_names(path_to_df)
        df = np.clip(_converge_dfs([pd.read_hdf(path_to_df, key=head) for head in head_names]), 0, 1)
    else:
        df = np.clip(converged_df, 0, 1)
    fig, ax = get_fig(4, 1)
    ax = plot_attention_scores_fat_tree(df, ax, 1)

    ax.text(0, -4, 'TOR0', {'fontsize': fontsize, 'fontfamily': 'serif'})
    ax.arrow(1, -3.5, -0.5, 1.5, width=0.3, color='black')

    ax.text(19, -4, 'TOR31', {'fontsize': fontsize, 'fontfamily': 'serif'})
    ax.arrow(30, -3.5, 0.5, 1.5, width=0.3, color='black')

    ax.text(32, -4, 'AGG0', {'fontsize': fontsize, 'fontfamily': 'serif'})
    ax.arrow(33, -3.5, -0.5, 1.5, width=0.3, color='black')

    ax.text(51, -4, 'AGG31', {'fontsize': fontsize, 'fontfamily': 'serif'})
    ax.arrow(62, -3.5, 0.5, 1.5, width=0.3, color='black')

    ax.text(64, -4, 'C0', {'fontsize': fontsize, 'fontfamily': 'serif'})
    ax.arrow(65, -3.5, -0.5, 1.5, width=0.3, color='black')

    ax.text(73, -4, 'C31', {'fontsize': fontsize, 'fontfamily': 'serif'})
    ax.arrow(78, -3.5, 0.5, 1.5, width=0.3, color='black')

    off = 8
    ax.text(-7, 0 + off, 'TOR0', {'fontsize': fontsize, 'fontfamily': 'serif'}, rotation='vertical')
    ax.arrow(-3.5, 1, 1.5, -0.5, width=0.3, color='black')

    ax.text(-7, 22 + off, 'TOR31', {'fontsize': fontsize, 'fontfamily': 'serif'}, rotation='vertical')
    ax.arrow(-3.5, 30, 1.5, 0.5, width=0.3, color='black')

    ax.text(-7, 33 + off, 'AGG0', {'fontsize': fontsize, 'fontfamily': 'serif'}, rotation='vertical')
    ax.arrow(-3.5, 33, 1.5, -0.5, width=0.3, color='black')

    ax.text(-7, 53 + off, 'AGG31', {'fontsize': fontsize, 'fontfamily': 'serif'}, rotation='vertical')
    ax.arrow(-3.5, 62, 1.5, 0.5, width=0.3, color='black')

    ax.text(-7, 60 + off, 'C0', {'fontsize': fontsize, 'fontfamily': 'serif'}, rotation='vertical')
    ax.arrow(-3.5, 65, 1.5, -0.5, width=0.3, color='black')

    ax.text(-7, 71 + off, 'C31', {'fontsize': fontsize, 'fontfamily': 'serif'}, rotation='vertical')
    ax.arrow(-3.5, 78, 1.5, 0.5, width=0.3, color='black')

    plt.axis('off')
    plt.subplots_adjust(left=0., right=0.99, bottom=0., top=0.99, hspace=0, wspace=0)
    plt.savefig(os.path.join(folder, f"{name}.{file_format}"))
    save_fig(folder, name, file_format, fig)
    plt.close(fig)


def _get_keys_in_file(file_path) -> List[str]:
    f = h5py.File(file_path, 'r')
    keys = [k for k in f.keys()]
    f.close()
    return keys


def save_fig(folder: str, name: str, format: str, fig: plt.Figure=None) -> None:
    if fig is None:
        fig=plt.gcf()
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "{:s}.{:s}".format(name, format)), format=format)
    fig.clear()
    plt.close(fig)


def _get_head_names(file_path: str) -> Tuple[int, List[str]]:
    keys = []
    num = 0
    f = h5py.File(file_path, 'r')
    while 'head{:d}'.format(num) in f.keys():
        keys.append('head{:d}'.format(num))
        num += 1
    f.close()
    return num, keys


def _converge_dfs(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    converged = None
    for df in dfs:
        converged = df.values if converged is None else converged + df.values
    converged = pd.DataFrame(converged, index=dfs[-1].index, columns=dfs[-1].columns)
    return converged


def _plot_multiple_scores(file_path, name):
    num, keys = _get_head_names(file_path)

    print(keys)
    dfs = [pd.read_hdf(file_path, key=k) for k in keys]
    for i, df in enumerate(dfs):
        ax = plot_attention_scores_fat_tree(df, clim=1)
        save_fig('./img/fcn-hlsas-attn-llt', '{}-head-{:d}'.format(name, i), format='pdf')
    converged = _converge_dfs(dfs)
    ax = plot_attention_scores_fat_tree(converged, clim=1)
    save_fig('./img/fcn-hlsas-llt', '{}-converged'.format(name), format='pdf')


def _make_cdf(array: np.array) -> pd.Series:
    s = pd.Series(array).value_counts()
    s.sort_index(inplace=True)
    s /= s.sum()
    return np.cumsum(s)


def cdf_from_np(array: np.array, xlabel: str) -> Tuple[plt.Figure, plt.Axes]:
    s = _make_cdf(array)
    fig, ax = get_fig(1)
    ax.plot(s.index.values, s.values)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("P(X < x)")
    ax.legend()
    return fig, ax


def _cdf_over_all_attn_weights(file_path, name):
    hosts = ['h-{:04d}'.format(i) for i in range(128)]
    scores = np.array([])
    for host in hosts:
        keys = []
        num = 0
        file_path = '/opt/project/data/scores-mha-{}.h5'.format(host)
        f = h5py.File(file_path, 'r')
        while 'head{:d}'.format(num) in f.keys():
            keys.append('head{:d}'.format(num))
            num += 1
        f.close()
        scores = np.concatenate([scores, np.concatenate([pd.read_hdf(file_path, key=k).values.flatten() for k in keys])])

    fig, ax = cdf_from_np(np.clip(scores, 1e-9, 1), 'Attention Scores')
    ax.set_xlim(1e-3, 1)
    ax.set_ylim(0.86, 1)
    ax.set_xscale('log')
    save_fig('./img', 'scores', format='svg', fig=fig)


def plot_hlsas(df):
    ax = plt.subplot()
    fig = plt.gcf()
    fig.set_figwidth(3)
    fig.set_figheight(df.shape[0] / df.shape[1] * 3)
    ax.imshow(df.values, cmap=cm.get_cmap("Greens"), vmin=0, vmax=1)
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_xticklabels([])
    ax.set_yticklabels(df.index.values)
    plt.tight_layout()
    plt.savefig('./img/fcn-hlsas-llt/hlsas.pdf')
    plt.close("all")


def plot_neighbor_attns(file_path):
    file_path = './data/fcn-hlsas/scores-neighbors.h5'
    keys = _get_keys_in_file(file_path)
    for k in keys:
        df = pd.read_hdf(file_path, key=k)
        ax = plt.subplot()
        fig = plt.gcf()
        ax.imshow(df.values, cmap=cm.get_cmap("Greens"), vmin=0, vmax=1)
        ax.set_xticks(np.arange(df.shape[1]))
        ax.set_yticks(np.arange(df.shape[0]))
        ax.set_xticklabels(df.columns, rotation=90)
        ax.set_yticklabels(df.index.values)
        plt.savefig('./img/fcn-hlsas/scores-neighbors-{}.pdf'.format(k), format='pdf')
        plt.clf()
        plt.close(fig)


def main():
    # plot_hlsas(pd.read_hdf('/opt/project/data/fcn-hlsas-llt/hlsas.h5', key='hlsas'))
    hosts = ['h-{:04d}'.format(i) for i in range(128)]
    for host in hosts:
        print(host)
        _plot_multiple_scores('/opt/project/data/fcn-hlsas-llt-14h/scores/scores-mha-{}.h5'.format(host), 'scores-mha-{}'.format(host))


if __name__ == '__main__':
    # _cdf_over_all_attn_weights('', '')
    main()

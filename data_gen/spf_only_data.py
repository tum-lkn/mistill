import networkx as nx
import os
import h5py
import numpy as np
import topos.fattree as ft
import embeddings.arbitrary as ernd
import embeddings.defaults as edf
import embeddings.learned as elnd
import embeddings.composed as ecmp
from dataprep.sp_prep import add_index_to_nodes, spf_dataset, distributional_spf_dataset
import dataprep.input_output as dpio


DATA_DIR = '/opt/project/data'
join = os.path.join


def _check_data_dir(prefix: str, suffix: str) -> str:
    ret = join(prefix, suffix)
    if not os.path.exists(ret):
        os.mkdir(ret)
    return ret


def _random_embedding(graph: nx.Graph, dim: int, full_output_file_path: str):
    embeddings = ernd._independent_bernoulli_based(graph, dim)
    dpio.write_embedding(embeddings, full_output_file_path)


def _distance_embedding(graph: nx.Graph, full_output_file_path: str):
    embeddings = edf.adj_embeddings(graph)
    dpio.write_embedding(embeddings, full_output_file_path)


def _ip_based_embedding(graph: nx.Graph, k: int, full_output_file_path: str) -> None:
    embeddings = edf.fat_tree_ip_scheme(graph, k)
    dpio.write_embedding(embeddings, full_output_file_path)


def _learned_embedding(graph: nx.Graph, ndims: int, full_output_file_path: str) -> None:
    alphas = np.repeat(np.linspace(5, 500, 50), 10)
    embeddings, num_duplicates, loss = elnd.learned_embeddings_gumbel(
        graph=graph,
        n_samples=alphas.size,
        ndims=ndims,
        e_fct=elnd.TradeOffEnergyFunction(ndims, gamma=1. / ndims, alpha=alphas),
        random_init=True
    )
    print("Number of duplicates are {} with log-loss {}".format(num_duplicates, loss))
    dpio.write_embedding(embeddings, full_output_file_path)
    os.rename(
        '/opt/project/img',
        os.path.join(
            '/opt/project',
            'img-{}'.format(os.path.split(full_output_file_path)[1].split('.')[0])
        )
    )
    os.mkdir('/opt/project/img')


def _make_spf_dataset(full_path_to_graph: str, data_set_name: str):
    with open(full_path_to_graph, 'r') as fh:
        data = fh.read()
    # Use undirected graph on purpose.
    graph = nx.jit_graph(data, create_using=nx.Graph())
    # hosts = [n for n in graph.nodes() if n.startswith('h')]
    hosts = list(graph.nodes())
    queries, values, masks_v, zs, cur_locs = spf_dataset(graph, hosts)
    file = h5py.File('/opt/project/data/{}.h5'.format(data_set_name), 'w')
    file.create_dataset("queries", data=queries)
    file.create_dataset("values", data=values)
    file.create_dataset("masks_v", data=masks_v)
    file.create_dataset("zs", data=zs)
    file.create_dataset("cur_locs", data=cur_locs)
    file.close()


def _make_distributional_dataset(full_path_to_graph: str, full_output_file_path: str):
    with open(full_path_to_graph, 'r') as fh:
        data = fh.read()
    # Use undirected graph on purpose.
    graph = nx.jit_graph(data, create_using=nx.DiGraph())
    hosts = list(graph.nodes())
    templates = distributional_spf_dataset(graph, hosts, use_mp=True)
    dpio.write_distributional_spf_dataset(templates, full_output_file_path)


def _fat_tree(k=8):
    data_dir = _check_data_dir(DATA_DIR, 'fat-tree-k{:d}'.format(k))
    graph = add_index_to_nodes(ft.make_topo(k))
    dpio.write_graph(graph, join(data_dir, 'fat-tree-k{:d}.json'.format(k)))
    _make_distributional_dataset(join(data_dir, 'fat-tree-k{:d}.json'.format(k)),
                                 join(data_dir, 'spf-distributional.h5'))
    # _random_embedding(graph, 24, join(data_dir, 'fat-tree-k{:d}-random-embedding.h5'.format(k)))
    # _make_spf_dataset("/opt/project/data/fat-tree/fat-tree-k{:d}-rnd.json".format(k),
    #                   'fat-tree-k{:d}-learned-rnd'.format(k))

    _ip_based_embedding(graph, k, join(data_dir, 'fat-tree-k{:d}-ip-embedding.h5'.format(k)))
    # _make_spf_dataset("/opt/project/data/fat-tree-k{:d}-ip.json".format(k),
    #                   'fat-tree-k{:d}-ip'.format(k))

    # for d in [12, 16, 20, 24]:
    #     _learned_embedding(graph, d, join(data_dir, 'fat-tree-k{:d}-learned-embedding-d{:d}.h5'.format(k, d)))
    # _make_spf_dataset("/opt/project/data/fat-tree-k{:d}-learned.json".format(k),
    #                   'fat-tree-k{:d}-learned-rnd'.format(k))

    # embeddings = ecmp.star_pod_decomposition(k, 12, 12)
    # dpio.write_embedding(embeddings, join(data_dir, 'fat-tree-k{:d}-learned-composed-embedding.h5'.format(k)))


def _erdoes_renyi(num_nodes=208, p=0.025):
    data_dir = _check_data_dir(DATA_DIR, 'er')
    count = 0
    connected = False
    template = None
    while not connected and count < 10000:
        count += 1
        template = nx.erdos_renyi_graph(num_nodes, p, 1)
        connected = nx.is_connected(template)
    assert connected, "Graph is not connected. Abort."
    template = add_index_to_nodes(template)
    dpio.write_graph(template, join(data_dir, 'erdoes-renyi.json'))
    _distance_embedding(template, join(data_dir, 'erdoes-renyi-dist-embedding.h5'))
    _make_distributional_dataset(join(data_dir, 'erdoes-renyi.json'),
                                 join(data_dir, 'spf-distributional.h5'))

    # g = template.copy()
    # _random_embedding(g, 24, join(data_dir, 'erdoes-renyi-random-embedding.h5'))
    # _make_spf_dataset("/opt/project/data/erdoes-renyi-random.json", 'erdoes-renyi-random')

    # g = template.copy()
    # for d in [12, 16, 20, 24]:
    #     _learned_embedding(g, d, join(data_dir, 'erdoes-renyi-learned-embedding-d{:d}.h5'.format(d)))
    # _make_spf_dataset("/opt/project/data/erdoes-renyi-learned.json", 'erdoes-renyi-learned-rnd')


def _barabas_albert(num_nodes=208, m=2):
    data_dir = _check_data_dir(DATA_DIR, 'ba')
    template = nx.barabasi_albert_graph(num_nodes, m, 1)
    template = add_index_to_nodes(template)
    dpio.write_graph(template, join(data_dir, 'barabasi-albert.json'))
    _distance_embedding(template, join(data_dir, 'barabasi-albert-dist-embedding.h5'))
    _make_distributional_dataset(join(data_dir, 'barabasi-albert.json'),
                                 join(data_dir, 'spf-distributional.h5'))

    # g = template.copy()
    # _random_embedding(g, 24, join(data_dir, 'barabasi-albert-random-embedding.h5'))
    # _make_spf_dataset("/opt/project/data/barabasi-albert-random.json", 'barabasi-albert-random')
    # _make_distributional_dataset(g, os.path.join(data_dir, 'barabasi-albert-dist-embedding.h5'))

    # g = template.copy()
    # for d in [12, 16, 20, 24]:
    #     _learned_embedding(g, 24, join(data_dir, 'barabasi-albert-learned-embedding-d{:d}.h5'.format(d)))
    # _make_spf_dataset("/opt/project/data/barabasi-albert-learned.json", 'barabasi-albert-learned-rnd')


def _as_graph(num_nodes=208):
    data_dir = _check_data_dir(DATA_DIR, 'as')
    template = nx.random_internet_as_graph(num_nodes, 1)
    template = add_index_to_nodes(template)
    dpio.write_graph(template, join(data_dir, 'random-as.json'))
    _distance_embedding(template, join(data_dir, 'random-as-dist-embedding.h5'))
    _make_distributional_dataset(join(data_dir, 'random-as.json'),
                                 join(data_dir, 'spf-distributional.h5'))

    # g = template.copy()
    # _random_embedding(g, 24, join(data_dir, 'random-as-random-embedding.h5'))
    # _make_spf_dataset("/opt/project/data/random-as-random.json", 'random-as-random')
    # _make_distributional_dataset(g, os.path.join(data_dir, 'random-as-dist-embedding.h5'))

    # g = template.copy()
    # for d in [12, 16, 20, 24]:
    #     _learned_embedding(g, 24, join(data_dir, 'random-as-learned-embedding-d{:d}.h5'.format(d)))
    # _make_spf_dataset("/opt/project/data/random-as-learned.json", 'random-as-learned-rnd')


def _k_regular(num_nodes=208, degree=8):
    data_dir = _check_data_dir(DATA_DIR, 'regular')
    template = nx.random_regular_graph(degree, num_nodes, seed=1)
    template = add_index_to_nodes(template)
    dpio.write_graph(template, join(data_dir, 'k-regular.json'))
    _distance_embedding(template, join(data_dir, 'k-regular-dist-embedding.h5'))
    _make_distributional_dataset(join(data_dir, 'k-regular.json'),
                                 join(data_dir, 'spf-distributional.h5'))

    # g = template.copy()
    # _random_embedding(g, 24, join(data_dir, 'k-regular-random-embedding.h5'))
    # _make_spf_dataset("/opt/project/data/k-regular-random.json", 'k-regular-random')
    # _make_distributional_dataset(g, os.path.join(data_dir, 'k-regular-dist-embedding.h5'))

    # g = template.copy()
    # for d in [12, 16, 20, 24]:
    #    _learned_embedding(g, 24, join(data_dir, 'k-regular-learned-embedding-d{:d}.h5'.format(d)))
    # _make_spf_dataset("/opt/project/data/k-regular-learned.json", 'k-regular-learned-rnd')


if __name__ == '__main__':
    print("================================================================================")
    print("Fat Tree")
    print("================================================================================")
    _fat_tree(8)
    print("Fat Tree K8 Done")
    _fat_tree(16)
    print("Fat Tree K16 Done")
    # _fat_tree(32)

    print("================================================================================")
    print("Erdoes Renyi")
    print("================================================================================")
    # _erdoes_renyi(100, 0.05)
    # data_dir = _check_data_dir(DATA_DIR, 'er')
    # _make_distributional_dataset(join(data_dir, 'erdoes-renyi.json'),
    #                              join(data_dir, 'spf-distributional.h5'))

    print("================================================================================")
    print("Barabasi Albert")
    print("================================================================================")
    # _barabas_albert(100)
    # data_dir = _check_data_dir(DATA_DIR, 'ba')
    # _make_distributional_dataset(join(data_dir, 'barabasi-albert.json'),
    #                              join(data_dir, 'spf-distributional.h5'))

    print("================================================================================")
    print("Random Internet")
    print("================================================================================")
    # _as_graph(100)
    # data_dir = _check_data_dir(DATA_DIR, 'as')
    # _make_distributional_dataset(join(data_dir, 'random-as.json'),
    #                              join(data_dir, 'spf-distributional.h5'))

    print("================================================================================")
    print("k-regular")
    print("================================================================================")
    # _k_regular(100, 4)
    # data_dir = _check_data_dir(DATA_DIR, 'regular')
    # _make_distributional_dataset(join(data_dir, 'k-regular.json'),
    #                              join(data_dir, 'spf-distributional.h5'))

"""
Implements a script that generates template datasets for link failures.
"""
import os
import sys
import time
import dataprep.input_output as inout
import multiprocessing as mp
import networkx as nx
from dataprep.link_failures import get_link_failure_dataset
from dataprep.sp_prep import add_index_to_nodes
from topos.fattree import make_topo
from embeddings.defaults import fat_tree_ip_scheme
from typing import List


def create_store_set(path_to_graph: nx.DiGraph, path_to_result_file: str,
                     num_samples: int, k: int, output_types: List[str], seed: int) -> None:
    # graph = inout.read_graph(path_to_graph)
    graph = path_to_graph
    dataset = get_link_failure_dataset(
        graph=graph,
        max_num_failures=10,
        seed=seed,
        k=k,
        num_samples=num_samples,
        output_types=output_types
    )
    inout.write_link_failure_data(dataset, path_to_result_file)


def mp_adaptor(args):
    create_store_set(**args)


def do_graph_mp(graph_folder: str, graph_json: str) -> None:
    folder = '/opt/project/data'
    processes = 30
    step = int(NUM_SAMPLES / 30)

    pool = mp.Pool(processes=processes)
    pool.map(
        mp_adaptor,
        [{
            'path_to_graph': os.path.join(folder, graph_folder, graph_json),
            'path_to_result_file':os.path.join(folder, graph_folder, 'link-failure-data-{:d}.h5'.format(i)),
            'num_samples': step,
            'seed': i + 1
        } for i in range(processes)]
    )
    pool.close()


def do_graph(graph_folder: str, dataset_name: str, graph_json: str, seed: int,
             set_type: str, k: int, output_types: List[str]) -> None:
    folder = '/opt/project/data'
    idx = 0
    count = 0
    step = 10000
    # step = 4000

    print("Dataset folder is: {}".format(os.path.join(folder, graph_folder, dataset_name)))
    if not os.path.exists(os.path.join(folder, graph_folder, dataset_name)):
        print("\tDoes not exist, thus create")
        os.mkdir(os.path.join(folder, graph_folder, dataset_name))
    if not os.path.exists(os.path.join(folder, graph_folder, dataset_name, 'train')):
        os.mkdir(os.path.join(folder, graph_folder, dataset_name, 'train'))
    if not os.path.exists(os.path.join(folder, graph_folder, dataset_name, 'val')):
        os.mkdir(os.path.join(folder, graph_folder, dataset_name, 'val'))

    graph = make_topo(k)
    graph = add_index_to_nodes(graph)
    embeddings = fat_tree_ip_scheme(graph, k)
    inout.write_graph(graph, os.path.join(folder, graph_folder, dataset_name, graph_json))
    inout.write_embedding(
        embeddings,
        os.path.join(folder, graph_folder, dataset_name, 'fat-tree-k{:d}-ip-embedding.h5'.format(k))
    )

    while count < 1: # NUM_SAMPLES:
        graph = make_topo(k)
        graph = add_index_to_nodes(graph)
        if count + step > NUM_SAMPLES:
            ns = NUM_SAMPLES - count
        else:
            ns = step
        # if count / NUM_SAMPLES < 0.75:
        #     set_type = 'train'
        # else:
        #     set_type = 'val'
        count += step
        idx += 1
        set_name = 'link-failure-data-{:d}.h5'.format(seed)
        create_store_set(
            # path_to_graph=os.path.join(folder, graph_folder, graph_json),
            path_to_graph=graph,
            path_to_result_file=os.path.join(folder, graph_folder, dataset_name, set_type, set_name),
            num_samples=ns,
            seed=seed,
            k=k,
            output_types=output_types
        )
        print("\tprocessed {:d} of {:d}".format(count, NUM_SAMPLES))


if __name__ == '__main__':
    seed = int(sys.argv[1])
    NUM_SAMPLES = 1000000
    print("Do Fat-Tree")
    s = time.time()
    # 'tors-to-hosts-llt-wcmp'
    do_graph(
        graph_folder='fat-tree-k8',
        graph_json='fat-tree-k8.json',
        dataset_name='tors-to-hosts',
        seed=seed,
        set_type=sys.argv[2],
        k=8,
        output_types=['lcp', 'hula', 'ecmp', 'wcmp']
    )
    print("Took {}s".format(time.time() - s))

    # print("Do AS")
    # do_graph('as', 'random-as.json')

    # print("Do BA")
    # do_graph('ba', 'barabasi-albert.json')

    # print("Do ER")
    # do_graph('er', 'erdoes-renyi.json')

    # print("Do Regular")
    # do_graph('regular', 'k-regular.json')



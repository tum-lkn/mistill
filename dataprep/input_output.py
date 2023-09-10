"""
This module implements I/O functionality, i.e., reading and writing data
to file.
"""
import networkx as nx
import numpy as np
import h5py
import json
from typing import Any, List, Dict, Tuple
from dataprep.sp_prep import add_index_to_nodes


def write_graph(graph: nx.Graph, full_path: str) -> None:
    """
    Write graph to file. If the graph is missing the index attribute, then its
    added to the graph before writing it to a json file.

    Args:
        graph:
        full_path:

    Returns:

    """
    if 'idx' not in graph.nodes[list(graph.nodes())[0]]:
        graph = add_index_to_nodes(graph)
    with open(full_path, 'w') as fh:
        fh.write(nx.jit_data(graph, indent=1))


def read_graph(full_path: str, is_directed=False) -> nx.Graph:
    """
    Read graph from file.
    Args:
        full_path:
        is_directed

    Returns:

    """
    with open(full_path, "r") as fh:
        data = json.load(fh)
    return nx.jit_graph(data, create_using=nx.DiGraph() if is_directed else nx.Graph())


def write_distributional_spf_dataset(data: Dict[str, np.array], full_path: str) -> None:
    """
    Create an h5 file containing the dataset in a similar form as the passed
    one. See dataprep.dataprep.distributional_spf_dataset for details on the
    file format.

    Args:
        data:
        full_path:

    Returns:

    """
    file = h5py.File(full_path, 'w')
    for k, v in data.items():
        file.create_dataset(name=k, data=v)
    file.close()


def write_embedding(data: Dict[int, np.array], full_path: str) -> None:
    """
    Create an h5 file containing the embeddings of a graph.
    Args:
        data:
        full_path:

    Returns:

    """
    file = h5py.File(full_path, 'w')
    for idx, val in data.items():
        grp = file.create_group('{:d}'.format(idx))
        grp.create_dataset(name='embedding', data=val)
    file.close()


def read_embeddings(full_path: str) -> np.array:
    file = h5py.File(full_path, 'r')
    embeddings = np.zeros((len(file), file['0']['embedding'][()].size), dtype=np.float32)
    for k in file.keys():
        embeddings[int(k), :] = file[k]['embedding'][()]
    file.close()
    return embeddings


def write_link_failure_data(data: Dict[str, np.array], full_path: str) -> None:
    """
    Writes a link failure data set to file.

    Args:
        data: The data that should be written. Expected to be the output
            of the function dataprep.link_failures.get_link_failure_dataset.
        full_path: Path including the extension to the file where the dataset
            should be stored. Intermediate folders must exist.

    Returns:

    """
    def write_down(h5_element, dictionary):
        for k, v in dictionary.items():
            if type(v) == dict:
                write_down(h5_element.create_group(k), v)
            else:
                h5_element.create_dataset(name=k, data=v)
    file = h5py.File(full_path, 'w')
    write_down(file, data)
    file.close()


def read_link_failure_data(full_path: str) -> Dict[str, np.array]:
    """
    Reads the saved data from a link failure dataset.

    Args:
        full_path: Path to file.

    Returns:
        data: Retrieved data. See dataprep.link_failures.get_link_failure_dataset
            for a description of the different parts and keys.
    """
    def read_down(h5ele, dictionary):
        for k in h5ele.keys():
            if type(h5ele[k]) == h5py.Group:
                dictionary[k] = {}
                read_down(h5ele[k], dictionary[k])
            else:
                dictionary[k] = h5ele[k][()]
    file = h5py.File(full_path, 'r')
    data = {}
    read_down(file, data)
    file.close()
    return data


def write_edges_to_pairs(full_path: str, edges_to_pairs: Dict[Tuple[Any, Any], Any]):
    storable_dict = {str(k): v.to_dict() for k, v in edges_to_pairs.items()}
    with open(full_path, 'w') as fh:
        json.dump(storable_dict, fh, indent=1)


def write_edges_to_paths(full_path: str, edges_to_paths: Dict[Tuple[Any, Any], List[Any]]) -> None:
    storable_dict = {str(k): v for k, v in edges_to_paths.items()}
    with open(full_path, 'w') as fh:
        json.dump(storable_dict, fh, indent=1)


def read_edges_to_paths(full_path: str) -> Dict[Tuple[Any, Any], List[Any]]:
    with open(full_path, 'r') as fh:
        stored_dict = json.load(fh)
    ret = {eval(k): v for k, v in stored_dict.items()}
    return ret


def read_edges_to_pairs(full_path: str) -> Dict[Tuple[Any, Any], Any]:
    from dataprep.link_failures import EdgeToPair
    with open(full_path, 'r') as fh:
        stored_dict = json.load(fh)
    ret = {eval(k): EdgeToPair.from_dict(v) for k, v in stored_dict.items()}
    return ret
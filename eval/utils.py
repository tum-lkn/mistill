"""
    Implements utility functions that are shared across multiple evaluation
    tasks.
"""
import torch
import pandas as pd
import logging
import os
import json
import networkx as nx
import numpy as np
import copy
from ray.tune import Analysis
from typing import List, Dict, Tuple, Any
from collections import OrderedDict
from models.stateful import StatefulConfig, StatefulModel
from training.utils import expand_stateful_config as _expand_stateful_config
from dataprep.sp_prep import NS_IDX, IDX
from dataprep.link_failures import EDGE_WEIGHT
logging.basicConfig(level=logging.INFO)


def _index_to_neighbor(graph: nx.Graph) -> Dict[Any, Dict[int, Any]]:
    """
        Maps for each node in the graph an index number to the corresponding
        label of the neighbor. Inverse of `_neighbor_to_index`.

        Args:
            graph: A graph.

        Returns:
            Dict that maps node identifiers to a dict that maps an integer to
            the neighbor of that node.
    """
    maps = {}
    for u in graph.nodes():
        maps[u] = {}
        for i, v in enumerate(graph.neighbors(u)):
            maps[u][i] = v
    return maps


def load_tune_analysis(full_dir_path: str) -> Analysis:
    """
        Loads the analysis output generated with a `tune.run` call.

        Args:
            full_dir_path: Full path to the directory in which the analysis
                result is stored.

        Returns:
            A tune analysis object.
    """
    analysis = Analysis(full_dir_path)
    return analysis


def load_multiple_tune_results(full_dir_path: str, experiment_prefix: str) -> List[Analysis]:
    """
        Search for all folders in the directory that start with the experiment_prefix
        and load the analysis result.

        Args:
            full_dir_path: Path in which experiments are contained.
            experiment_prefix: Prefix that all experiments have in common.

        Returns:
            results: List of analysis results.
    """
    results = []
    folders = os.listdir(full_dir_path)
    folders.sort(key=lambda x: len(x))
    for f in folders:
        if f.startswith(experiment_prefix):
            results.append(load_tune_analysis(os.path.join(full_dir_path, f)))
        else:
            continue
    return results


def expand_df(df: pd.DataFrame) -> pd.DataFrame:
    """
        Tune results the top level entries of the search space. Since I have
        nested columns, the actual configuration is hidden in the cells.
        This method expands the cells into columns.

        Args:
            df: DataFrame that should be extended.

        Returns:
            DataFrame with a lot more columns.
    """
    bs = 'config/batch_size'
    df['batch_size'] = df[bs].values
    cols = ['config/model_config', 'config/optimizer']

    for col in cols:
        for k, v in df.iloc[0][col].items():
            df[k] = None

    for idx in df.index.values:
        for col in cols:
            d = df.loc[idx, col]
            for k, v in d.items():
                df.at[idx, k] = v

    cols.append(bs)
    df.drop(cols, axis=1, inplace=True)
    return df


def remove_trash_cols(df: pd.DataFrame) -> pd.DataFrame:
    to_remove = [
        'done',
        'timesteps_total',
        'episodes_total',
        # 'training_iteration',
        'date',
        'timestamp',
        'time_this_iter_s',
        'time_total_s',
        'pid',
        'hostname',
        'node_ip',
        'time_since_restore',
        'timesteps_since_restore',
        'iterations_since_restore',
        'experiment_tag',
        'config/num_epochs_per_train_call',
        'config/distributional',
        'logdir',
        'experiment_id',
        'trial_id'
    ]
    if 'config/seed' in df.columns:
        to_remove.append('config/seed')
    if 'config/model' in df.columns:
        to_remove.append('config/model')
    if 'min-cross_entropy' in df.columns:
        to_remove.append('min-cross_entropy')
    return df.drop(to_remove, axis=1)


def load_experiments(dir_path: str) -> Dict[str, List[Analysis]]:
    """
        Iterate over all folders in the given directory and load all experiments
        in there.
    """
    returns = {}
    for exp_dir in os.listdir(dir_path):
        returns[exp_dir] = load_multiple_tune_results(os.path.join(dir_path, exp_dir), '')
    return returns


def export_table(tex_path: str, k: str, v: pd.DataFrame) -> None:
    """
        Write a latex table to a tex file.
    """
    def shorten(name):
        if name.find("_") > 0:
            x = ''.join([s[0] for s in name.split('_')])
        else:
            x = name
        return x.upper()

    txt = "\\begin{{table}}\n  \\tiny\n{:s}\n  \\caption{{ {:s} - {:s} }}\n\\end{{table}}"
    with open(os.path.join(tex_path, '{:s}.tex'.format(k)), 'w') as fh:
        fh.write(txt.format(
            v.sort_values('cross_entropy', ascending=True).to_latex(
                header=[shorten(x) for x in v.columns]
            ),
            k,
            ', '.join(["{} = {}".format("\\_".join(s.split("_")), shorten(s)) for s in v.columns])
        ))


def make_tex(dir_path: str, tex_path: str, min_iter=1000) -> None:
    """
        Read all experiments from the directory, load all runs of the experiments
        and write the resulting dataframe to a latex table.
    """
    results = load_experiments(dir_path)
    for k, v in results.items():
        results[k] = pd.concat([remove_trash_cols(expand_df(x.dataframe())).dropna() for x in v])
        results[k] = results[k].loc[results[k]['training_iteration'] > min_iter]

    for k, v in results.items():
        export_table(tex_path, k, v)


def get_max_checkpoint(path: str) -> str:
    """
        Return the name of the folder with the highest checkpoint.
    """
    max_num = -1
    max_name = ''
    for p in os.listdir(path):
        if p.startswith("checkpoint"):
            tmp = int(p.split("_")[1])
            if tmp > max_num:
                max_num = tmp
                max_name = p
    return max_name


def expand_stateful_config(config_class, config: dict, max_num_blocks) -> StatefulConfig:
    return config_class.from_dict(_expand_stateful_config(config, max_num_blocks)['model_config'])


def load_model(config_class: callable, model_class: callable, trial_dir: str, max_num_blocks: int,
               parallel=False, checkpoint_dir=None) -> torch.nn.Module:
    """
        Load model from trial checkpoint from ray tune.

        Args:
            config_class: Configuration class for a specific model class
                (check models package).
            model_class: Class of a neural network model (check models package).
            trial_dir: Path to the folder of a trial.
            parallel: Whether data parallel is enabled or not.
            checkpoint_dir: Optional, folder name of a specific trial that
                should be loaded. By default the latest trial is loaded.

        Returns:
            model: A neural network model with the weights stored in the checkpoint.
    """
    if checkpoint_dir is None:
        checkpoint_dir = get_max_checkpoint(trial_dir)

    with open(os.path.join(trial_dir, 'params.json'), "r") as fh:
        params = json.load(fh)
    print(params)
    # The state dict could be saved on a GPU. So we might have to move the
    # data storage to CPU.
    state_dict = torch.load(
        os.path.join(trial_dir, checkpoint_dir, "model.pth"),
        map_location=torch.device('cpu')
    )
    if parallel:
        pass
    else:
        # If data parallel is used then the parameters are prefixed with .model.
        # When loading the model not in data parallel mode, then this .model is
        # not expected and results in an error.
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
        state_dict = new_state_dict

    # model = model_class(expand_stateful_config(config_class, params, max_num_blocks))
    expanded_config = expand_stateful_config(config_class, params, max_num_blocks)
    # model = model_class(config_class.from_dict(params['model_config']))
    model = model_class(expanded_config)
    model.load_state_dict(state_dict)
    return model


def get_logdir_best_model(experiment_dir: str, experiment_prefix: str=None,
                          metric='cross_entropy-val') -> str:
    """
        Retrieve analysis results and get the log dir of the model with the
        best objective value.

        Args:
            experiment_dir: Directory in which the experiment result folders
                created with tune.run are located.
            experiment_prefix: Prefix identifying a specific subset of
                experiments.

        Returns:
            logdir: Path to the log dir of the model with the smallers lost.
    """
    anas = load_multiple_tune_results(experiment_dir, experiment_prefix)
    df = pd.concat([x.dataframe(metric, 'min') for x in anas])
    df = df.sort_values(by=metric)
    logdir = df['logdir'].iloc[0]
    logging.info("Best model has {} of {}, logdir: {}".format(
        metric,
        df[metric].iloc[0],
        df['logdir'].iloc[0]
    ))
    # ce = df['cross_entropy'].iloc[0]
    return logdir


def get_logdirs_best_models(experiment_dir: str, loss_cutoff: float,
                            experiment_prefix: str=None, metric='cross_entropy-val') -> List[str]:
    anas = load_multiple_tune_results(experiment_dir, experiment_prefix)
    df = pd.concat([x.dataframe(metric, 'min') for x in anas])
    logdirs = []
    for i in range(df.shape[0]):
        if df.iloc[i][metric] < loss_cutoff:
            logdirs.append(df.iloc[i]['logdir'])
    return logdirs


def get_config(trial_dir: str) -> Dict[str, Any]:
    ana = Analysis(trial_dir)
    df = ana.dataframe('cross_entropy-val', 'min')
    if df.shape[0] == 0:
        return None
    with open(os.path.join(trial_dir, 'params.json'), 'r') as fh:
        config = json.load(fh)
    config['cross_entropy-val'] = df.loc[0, 'cross_entropy-val']
    config['cross_entropy'] = df.loc[0, 'cross_entropy']
    config['training_iteration'] = df.loc[0, 'training_iteration']
    config['time_this_iter_s'] = df.loc[0, 'time_this_iter_s']
    config['time_total_s'] = df.loc[0, 'time_total_s']
    config['episodes_total'] = df.loc[0, 'episodes_total']
    config['logdir'] = df.loc[0, 'logdir']
    return config


def get_all_experiment_results(exp_folder_path: str, filter=None) -> List[Dict[str, Any]]:
    configs = []
    if filter is None:
        for trial_dir in os.listdir(exp_folder_path):
            p = os.path.join(exp_folder_path, trial_dir)
            if os.path.isdir(p):
                configs.append(get_config(p))
    else:
        for f in os.listdir(exp_folder_path):
            if f.startswith(filter):
                configs.extend(get_all_experiment_results(os.path.join(exp_folder_path, f)))
    configs.sort(key=lambda x: x['cross_entropy-val'])
    return configs


def get_all_multipart_experiment_results(result_path: str, exp_name_prefix: str):
    configs = []
    for exp_dir in os.listdir(result_path):
        if exp_dir.startswith(exp_name_prefix):
            configs.extend(get_all_experiment_results(os.path.join(result_path, exp_dir)))
    return configs


def get_num_params(model) -> int:
    return int(np.sum([p.numel() for p in model.parameters()]))


def export_configs_tex(configs: List[Dict]) -> str:
    tex_str = """
            \\begin{{tabular}}{{p{{1cm}} p{{1cm}} p{{1cm}} p{{1cm}} p{{1cm}} p{{1.75cm}} p{{1cm}} p{{1cm}} p{{1cm}} p{{1cm}} p{{1.75cm}}}}
            \\toprule
            Loss & Params & Learning Rate & Batch Size & Neighbor Attn FCN & Link Attn FCN & HLSA Attn Fcn
                & HLSA Attn hidden & HLSA Attn out & HLSA Attn heads & final FCN\\\\
            \\midrule
            {}
            \\bottomrule
            \\end{{tabular}}
    """
    def prettyfy_int(num):
        s = str(num)
        if len(s) > 6:
            s = "{}\\,{}".format(s[0:-6], s[-6:])
        if len(s) > 3:
            s = "{}\\,{}".format(s[0:-3], s[-3:])
        return s

    template = "{loss:.4f} & {num_params} & {lr:.6f} & {bs} & {ngh_attn_fcn} & {link_attn_fcn} & {hlsa_attn_fcn} " + \
        "& {hlsa_attn_hidden} & {hlsa_attn_out} & {hlsa_attn_heads} " + \
        "& {final_fcns}\\\\"
    lines = []
    for config in configs:
        try:
            num_params = get_num_params(load_model(StatefulConfig, StatefulModel, config['logdir']))
        except Exception as e:
            print(e)
            num_params = -1
        lines.append(
            template.format(
                loss=config['cross_entropy-val'],
                ngh_attn_fcn=config['model_config']['neighbor_attns']['dim_fcn'],
                lr=config['optimizer']['lr'],
                bs=config['batch_size'],
                link_attn_fcn=[config['model_config']['link_attns'][i]['dim_fcn']
                               for i in range(len(config['model_config']['link_attns']))],
                hlsa_attn_fcn=config['model_config']['hlsa_attns']['dim_fcn'],
                hlsa_attn_hidden=config['model_config']['hlsa_attns']['dim_hidden'],
                hlsa_attn_out=config['model_config']['hlsa_attns']['dim_out'],
                hlsa_attn_heads=config['model_config']['hlsa_attns']['num_heads'],
                final_fcns=config['model_config']['final_fcns'],
                num_params=prettyfy_int(num_params)
            )
        )
    return tex_str.format('\n\t\t\t'.join(lines))


def deep_copy_graph(graph: nx.Graph) -> nx.Graph:
    """
    Make a deep copy of a graph. That is, all containers on edges and nodes
    are added as deep copies instead of shallow copies.
    Args:
        graph:

    Returns:

    """
    reconstruction = nx.DiGraph() if graph.is_directed() else nx.Graph()
    reconstruction.graph.update(copy.deepcopy(graph.graph))
    reconstruction.add_nodes_from((n, copy.deepcopy(d)) for n, d in graph.nodes(data=True))
    reconstruction.add_edges_from((u, v, copy.deepcopy(d)) for u, v, d in graph.edges(data=True))
    return reconstruction


def reconstruct_graph(graph: nx.Graph, neighbor_to_idx: Dict[str, Dict[str, int]],
                      state: np.array, correct_shift:bool=False) -> nx.Graph:
    """
    Reconstruct a specific topology from a state record. The passed graph is
    copied deeply and thus not changed.

    Args:
        graph: Base graph.
        neighbor_to_idx: Maps the neighbors of nodes to indices.
        state: Array representing the exported state. Has shape
            (num_nodes_with_state, max_degree, num_features).

    Returns:
        reconstruction: The reconstructed graph from the data sample.
    """
    reconstruction = deep_copy_graph(graph)
    edges = list(graph.edges())
    for u, v in edges:
        if NS_IDX in graph.nodes[u]:
            u_idx = graph.nodes[u][NS_IDX]
            v_idx = neighbor_to_idx[u][v]
            if state.shape[-1] == 5:
                if correct_shift:
                    reconstruction.edges[u, v][EDGE_WEIGHT] = state[u_idx, v_idx, -1] + 10
                else:
                    reconstruction.edges[u, v][EDGE_WEIGHT] = state[u_idx, v_idx, -1]
            if reconstruction.has_edge(u, v) and state[u_idx, v_idx, 0] == 0:
                reconstruction.remove_edge(u, v)
            if reconstruction.has_edge(v, u) and state[u_idx, v_idx, 2] == 0:
                reconstruction.remove_edge(v, u)
    return reconstruction



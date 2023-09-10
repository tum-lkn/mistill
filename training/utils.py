from ray import tune
import json
import networkx as nx
import numpy as np
import embeddings.defaults as embdf


class CustomStopper(tune.Stopper):
    def __init__(self, metric: str, patience: int, min_num_iter: int, max_iter: int):
        self.should_stop = False
        self.max_iter = max_iter
        self.patience = patience
        self.min_num_iter = min_num_iter
        self.best_val = 1e9
        self.metric = metric

    def __call__(self, trial_id, result):
        if result[self.metric] < self.best_val:
            self.best_val = result[self.metric]
            if self.min_num_iter < self.patience + result['training_iteration']:
                self.min_num_iter = result['training_iteration'] + self.patience
        if result['training_iteration'] > self.max_iter or result['training_iteration'] > self.min_num_iter:
            self.should_stop = True
        return self.should_stop

    def stop_all(self):
        return self.should_stop


def check_embedding_consistency(graph: nx.Graph, embeddings: np.array, k: int) -> bool:
    def selector(name: str):
        if name.startswith('h-'):
            return host_ip
        elif name.startswith('tor-'):
            return tor_ip
        elif name.startswith('agg-'):
            return agg_ip
        elif name.startswith('core-'):
            return core_ip
        else:
            raise KeyError("Unexpeted node name {}".format(name))
    host_ip = embdf.HostIp(k)
    agg_ip = embdf.AggIp(k)
    tor_ip = embdf.TorIp(k)
    core_ip = embdf.CoreIp(k)

    ret = True
    for node, d in graph.nodes(data=True):
        embd_z = selector(node)(node)
        embd_y = embeddings[d['idx'], :]
        ret = np.sum(embd_z == embd_y) == embd_z.size
        if not ret:
            print("EMbeddings do not match for {}".format(node))
            break
    return ret


def expand_stateful_config(config: dict, max_num_blocks=64, hlsa_attn='hlsa_attns') -> dict:
    if config['model_config']['hlsa_gs'] is not None:
        arity = config['model_config']['hlsa_gs']['arity']
        bits = np.ceil(np.log2(arity))
        num_blocks = int(np.min([int(max_num_blocks / bits), config['model_config']['hlsa_gs']['num_blocks']]))
        config['model_config']['hlsa_gs']['num_blocks'] = num_blocks
        if config['model_config']['hlsa_model'] is not None:
            dim_out = arity * num_blocks
            config['model_config']['link_attns'][-1]['dim_fcn'] = dim_out

    if config['model_config']['hlsa_model'] == 'fcn':
        config['model_config']['link_attns'][0]['dim_in'] = 48# 96
    for i in range(len(config['model_config']['link_attns']) - 1):
        config['model_config']['link_attns'][i + 1]['dim_in'] = config['model_config']['link_attns'][i]['dim_fcn']

    if config['model_config']['hlsa_model'] is None:
        print("WARNING: Input to HLSA attention set statically to 96 !!!!!!!!!!!!!!!!!!!!!!!!!")
        config['model_config'][hlsa_attn]['dim_in'] = 96  # config['model_config']['link_attns'][-1]['dim_fcn']
    else:
        config['model_config'][hlsa_attn]['dim_in'] = config['model_config']['link_attns'][-1]['dim_fcn']

    if config['model_config']['cur_loc_and_dst_q_hlsa']:
        config['model_config'][hlsa_attn]['dim_q'] = config['model_config']['dim_embedding'] * 2
    else:
        config['model_config'][hlsa_attn]['dim_q'] = config['model_config']['dim_embedding']

    if config['model_config']['neighbor_model'] == 'attn':
        config['model_config']['neighbor_attns']['dim_in'] = config['model_config']['dim_embedding']
    else:
        config['model_config']['neighbor_attns']['dim_in'] = 2 * config['model_config']['dim_embedding']
    # print(json.dumps(config, indent=1))
    return config


def expand_separate_sensor_config(config: dict, max_num_blocks_lf: int=16,
                                  max_num_blocks_w: int=32):
    config = expand_stateful_config(config, max_num_blocks_lf, 'hlsa_attn')
    if config['model_config']['hlsa_weight_gs'] is not None:
        arity = config['model_config']['hlsa_weight_gs']['arity']
        bits = np.ceil(np.log2(arity))
        num_blocks = int(np.min([
            int(max_num_blocks_w / bits),
            config['model_config']['hlsa_weight_gs']['num_blocks']
        ]))
        config['model_config']['hlsa_weight_gs']['num_blocks'] = num_blocks
        if config['model_config']['hlsa_weight_model'] is not None:
            dim_out = arity * num_blocks
            config['model_config']['link_attns_weight'][-1]['dim_fcn'] = dim_out

    if config['model_config']['hlsa_weight_model'] is None:
        print("WARNING: Input to HLSA attention set statically to 96 !!!!!!!!!!!!!!!!!!!!!!!!!")
        config['model_config']['hlsa_weight_attn']['dim_in'] = 96  # config['model_config']['link_attns'][-1]['dim_fcn']
    else:
        config['model_config']['hlsa_weight_attn']['dim_in'] = config['model_config']['link_attns_weight'][-1]['dim_fcn']
    if config['model_config']['cur_loc_and_dst_q_hlsa']:
        config['model_config']['hlsa_weight_attn']['dim_q'] = config['model_config']['dim_embedding'] * 2
    else:
        config['model_config']['hlsa_weight_attn']['dim_q'] = config['model_config']['dim_embedding']

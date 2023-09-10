from __future__ import annotations
import json
import os
import struct

import numpy as np
import torch
from typing import List, Dict, Tuple, Any

import models.stateful as ms


def convert_hsnas():
    hnsas = np.load('/opt/project/scripts/sample-hnsas.npy', allow_pickle=True)
    hnsas = hnsas.astype(np.uint64)
    ints = np.zeros(hnsas.shape[1], dtype=np.uint64)
    for i in range(hnsas.shape[1]):
        for j in range(63, -1, -1):
            ints[i] <<= np.array([1], dtype=np.uint64)[0]
            ints[i] += hnsas[0, i, 2 * j]
    s = "[" + ', '.join([str(x) for x in ints]) + "]"
    with open('/opt/project/scripts/integer-hnsas.txt', 'w') as fh:
        fh.write(s + "\n")



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


def load_config(trial_dir: str) -> ms.StatefulConfig:
    with open(os.path.join(trial_dir, 'params.json'), "r") as fh:
        config_d = json.load(fh)
    config = ms.StatefulConfig.from_dict(expand_stateful_config(config_d, 64)['model_config'])
    return config


def load_state_dict(checkpoint_dir: str) -> Dict[str, torch.Tensor]:
    state_dict = torch.load(
        os.path.join(checkpoint_dir, "model.pth"),
        map_location=torch.device('cpu')
    )
    return state_dict


def load_hlsa_model(checkpoint_dir: str, config: ms.StatefulConfig) -> ms.HlsaModel:
    state_dict = load_state_dict(checkpoint_dir)
    num_params_sw = 0
    for k, v in state_dict.items():
        if "attend_over_links" in k:
            num_params_sw += v.detach().numpy().size
        else:
            continue
    print(f"The switch NN has {num_params_sw} parameters")
    model = ms.HlsaModel(config)
    model.load_state_dict(state_dict)
    return model


def load_forward_model(checkpoint_dir: str, config: ms.StatefulConfig) -> ms.ForwardingModel:
    state_dict = load_state_dict(checkpoint_dir)
    num_params_ext_host = 0
    for k, v in state_dict.items():
        if "attend_over_links" in k:
            continue
        else:
            num_params_ext_host += v.detach().numpy().size
    print(f"The external host NN has {num_params_ext_host} parameters")
    model = ms.ForwardingModel(config)
    model.load_state_dict(state_dict)
    return model


def row_stack_tensor(arr: np.array, stride: int) -> List[float]:
    flattened = []
    if arr.ndim == 1:
        for col in range(arr.shape[0]):
            flattened.append(float(arr[col]))
        remainder = divmod(arr.shape[0], stride)[1]
        if remainder > 0:
            for _ in range(stride - remainder):
                flattened.append(0)
        return flattened
    else:
        for row in range(arr.shape[0]):
            flattened.extend(row_stack_tensor(arr[row, ...], stride))
        return flattened


def export_forward_model_to_bin():
    stride = 1
    trial_dir = '/opt/project/data/results/gs/k8-hula-b64/StateTrainable_804cc_00097_97/'
    checkpoint_dir = os.path.join(trial_dir, 'checkpoint_150')
    config = load_config(trial_dir)
    # Weight matrices have the shape (dim_out, dim_in). To match the C code,
    # the flat weights must be row-major.
    state_dict = load_state_dict(checkpoint_dir)
    for k, v in state_dict.items():
        print(k, v.shape)
    flat_parameters = []
    template = "attend_over_hlsas.attention_layer.layers.{head:d}.transform_{kqv:s}.weight"
    for i in range(config.hlsas_attn.num_heads):
        k = template.format(head=i, kqv='keys')
        flat_parameters.extend(row_stack_tensor(state_dict[k].detach().numpy(), stride))
        k = template.format(head=i, kqv='queries')
        flat_parameters.extend(row_stack_tensor(state_dict[k].detach().numpy(), stride))
        k = template.format(head=i, kqv='values')
        flat_parameters.extend(row_stack_tensor(state_dict[k].detach().numpy(), stride))
    for prefix in ['attend_over_hlsas.linear', 'attend_over_neighbors.0', 'process_embedding.0', 'process_embedding.2', 'make_logits', 'make_logits_ecmp']:
        flat_parameters.extend(row_stack_tensor(
            arr=state_dict[f"{prefix}.weight"].detach().numpy(),
            stride=stride
        ))
        flat_parameters.extend(row_stack_tensor(
            arr=state_dict[f"{prefix}.bias"].detach().numpy(),
            stride=1
        ))
    print("Flat parameters has ", len(flat_parameters), ' elements')
    s = struct.pack(f"{len(flat_parameters)}f", *flat_parameters)
    with open(f'/opt/project/data/results/gs/k8-hula-b64/StateTrainable_804cc_00097_97/parameters-forwarding-module-stride-{stride}.bin', 'wb') as fh:
        fh.write(s)


def export_hlsa_model_to_bin():
    stride = 1
    trial_dir = '/opt/project/data/results/gs/k8-hula-b64/StateTrainable_804cc_00097_97/'
    checkpoint_dir = os.path.join(trial_dir, 'checkpoint_150')
    config = load_config(trial_dir)
    # Weight matrices have the shape (dim_out, dim_in). To match the C code,
    # the flat weights must be row-major.
    state_dict = load_state_dict(checkpoint_dir)
    flat_parameters = []
    for prefix in ["_attend_over_links.0"]:
        flat_parameters.extend(row_stack_tensor(
            arr=state_dict[f"{prefix}.weight"].detach().numpy(),
            stride=stride
        ))
        flat_parameters.extend(row_stack_tensor(
            arr=state_dict[f"{prefix}.bias"].detach().numpy(),
            stride=stride
        ))
    print("Flat parameters has ", len(flat_parameters), ' elements')
    s = struct.pack(f"{len(flat_parameters)}f", *flat_parameters)
    with open(f'/opt/project/data/results/gs/k8-hula-b64/StateTrainable_804cc_00097_97/parameters-hlsa-module-stride-{stride}.bin', 'wb') as fh:
        fh.write(s)


def export_hlsa_model():
    td = '/opt/project/data/results/gs/k8-hula-b64/StateTrainable_804cc_00097_97'
    cpd = os.path.join(td, 'checkpoint_150')
    config = load_config(td)
    config.num_nodes_with_state = 1
    model = load_hlsa_model(cpd, config)
    model.eval()
    model.argmax_infer = True
    # args = (
    #     torch.Tensor(np.random.uniform(0, 1, size=(1, config.num_nodes_with_state, 8, 6)).astype(np.float32)),
    #     torch.ones([1, config.num_nodes_with_state, 8, 1], dtype=torch.float32)
    # )
    args = torch.Tensor(np.random.uniform(0, 1, size=(1, config.num_nodes_with_state, 8, 6)).astype(np.float32))
    traced_model = torch.jit.trace(
        model,
        args
    )
    out = model(args)
    out_t = traced_model(args)
    diff = (out - out_t).detach().numpy()
    traced_model.save('/opt/project/data/results/gs/k8-hula-b64/StateTrainable_804cc_00097_97/traced-hlsa-module.pt')
    print("done")
    return traced_model


def export_forward_model():
    td = '/opt/project/data/results/gs/k8-hula-b64/StateTrainable_804cc_00097_97'
    cpd = os.path.join(td, 'checkpoint_150')
    config = load_config(td)
    config.hlsas_attn.attn_activation = 'hard_gs'
    model = load_forward_model(cpd, config)
    model.eval()
    args = (
            torch.Tensor(np.random.randint(0, 2, size=(1, 80, 128)).astype(np.float32)),
            torch.Tensor(np.random.randint(0, 2, size=(1, 24)).astype(np.float32)),
            torch.Tensor(np.random.randint(0, 2, size=(1, 24)).astype(np.float32)),
            torch.Tensor(np.random.randint(0, 2, size=(1, 80, 24)).astype(np.float32))
        )
    traced_model = torch.jit.trace(
        model,
        args
    )
    out = model(*args)
    out_t = traced_model(*args)
    diff = (out - out_t).detach().numpy()
    traced_model.save('/opt/project/data/results/gs/k8-hula-b64/StateTrainable_804cc_00097_97/traced-forwarding-module.pt')
    print("done")
    return traced_model


def verify_consistency(hlsa_model, forward_model) -> None:
    import dataprep.sp_prep as sp_prep
    import dataprep.link_failures as lf_prep
    from topos.fattree import make_topo
    from dataprep.input_output import read_graph, read_embeddings, read_link_failure_data
    import itertools
    import eval.utils as evutils
    import eval.lf_eval as lfev
    import networkx as nx
    DEV = 'cpu'

    td = '/opt/project/data/results/gs/k8-hula-b64/StateTrainable_804cc_00097_97'
    cpd = os.path.join(td, 'checkpoint_150')
    config = load_config(td)
    config.num_nodes_with_state = 1
    model = evutils.load_model(ms.StatefulConfig, ms.StatefulModel, td, 64, False, 'checkpoint_150')
    model.eval()
    model.argmax_infer = True

    random = np.random.RandomState(seed=1)
    num_outputs = 8
    np_embeddings = read_embeddings('/opt/project/data/fat-tree-k8/fat-tree-k8-ip-embedding.h5')
    graph = sp_prep.add_index_to_nodes(make_topo(8))
    lf_prep._add_gaussian_edge_weights(graph, random)
    for u, v in graph.edges():
        graph.edges[(u, v)]['gaussian'] = graph.edges[(u, v)]['gaussian'] / 10.
    graph.edges[('tor-0000', 'h-0000')]['gaussian'] = 0.05
    graph.edges[('tor-0000', 'h-0001')]['gaussian'] = 0.2
    graph.edges[('tor-0000', 'h-0002')]['gaussian'] = 0.15
    graph.edges[('tor-0000', 'h-0003')]['gaussian'] = 0.1
    graph.edges[('h-0000', 'tor-0000')]['gaussian'] = 0.05
    graph.edges[('h-0001', 'tor-0000')]['gaussian'] = 0.2
    graph.edges[('h-0002', 'tor-0000')]['gaussian'] = 0.15
    graph.edges[('h-0003', 'tor-0000')]['gaussian'] = 0.1

    graph.edges[('tor-0000', 'agg-0000')]['gaussian'] = 0.001
    graph.edges[('tor-0000', 'agg-0001')]['gaussian'] = 0.5
    graph.edges[('tor-0000', 'agg-0002')]['gaussian'] = 0.85
    graph.edges[('tor-0000', 'agg-0003')]['gaussian'] = 0.9
    graph.edges[('agg-0000', 'tor-0000')]['gaussian'] = 0.001
    graph.edges[('agg-0001', 'tor-0000')]['gaussian'] = 0.5
    graph.edges[('agg-0002', 'tor-0000')]['gaussian'] = 0.85
    graph.edges[('agg-0003', 'tor-0000')]['gaussian'] = 0.9
    value_index = sp_prep._neighbor_to_index(graph)
    hosts = ['h-0016']# , 'h-0032', 'h-0048', 'h-0064']
    # non_hosts = [n for n in graph.nodes() if not n.startswith('h-')]
    non_hosts = ['tor-0000']
    pairs = [(u, v) for u, v in itertools.product(non_hosts, hosts)]

    for n in ['tor-0000', 'agg-0000', 'agg-0001', 'agg-0002', 'agg-0003']:
        for x in nx.neighbors(graph, n):
            print(n, x, graph.edges[n, x]['gaussian'])

    num_non_leaves = lf_prep._get_num_no_leaves(graph)
    nodes_with_state = np.zeros(num_non_leaves, dtype=np.int32)
    for n, d in graph.nodes(data=True):
        if sp_prep.NS_IDX in d:
            nodes_with_state[d[sp_prep.NS_IDX]] = d[sp_prep.IDX]
    np_all_masks = np.expand_dims(
        lf_prep._get_all_masks(
            graph=graph,
            max_degree=num_outputs,
            failed_links=[],
            neighbor_to_index=value_index
        ),
        axis=0
    )
    np_masks_nodes_with_state = np_all_masks[:, nodes_with_state, :, :]
    np_all_neighbors = lf_prep._get_all_neighbors(
        graph=graph,
        max_degree=num_outputs,
        neighbor_to_index=value_index
    )
    # graph.remove_edge('tor-0000', 'agg-0000')
    # graph.remove_edge('agg-0000', 'tor-0000')
    np_states, np_targets, _, np_destinations_idx, np_cur_locs_idx = lfev._create_initial_input(
        graph=graph,
        pairs=pairs,
        num_non_leaves=num_non_leaves,
        value_index=value_index,
        num_outputs=num_outputs,
        output_mode='hula'
    )
    np_states[0, 32, 0, 0] = 1
    np_states[0, 32, 0, 1] = 0
    np_states[0, 32, 0, 2] = 1
    np_states[0, 32, 0, 3] = 0
    np_states[0, 32, 0, 4] = 0.001
    np_states[0, 32, 0, 5] = 0.001
    np_neighbor_masks = np_all_masks[0, np_cur_locs_idx, :, :]
    np_embd_neighbors = np_all_neighbors[np_cur_locs_idx, :, :].astype(np.int32)
    np_embd_neighbors = np_embeddings[np_embd_neighbors.flatten(), :].reshape(
        -1, np_all_neighbors.shape[1], np_embeddings.shape[-1]
    )
    np_embd_cur_locs = np_embeddings[np_cur_locs_idx, :]
    np_destinations = np_embeddings[np_destinations_idx, :]
    np_embeddings_nodes_with_states = np.expand_dims(np_embeddings[nodes_with_state], 0)

    t_neighbor_masks = torch.tensor(np_neighbor_masks)
    t_embd_neighbors = torch.tensor(np_embd_neighbors)
    t_states = torch.tensor(np_states)
    t_all_masks = torch.tensor(np_masks_nodes_with_state)
    t_embd_cur_loc = torch.tensor(np_embd_cur_locs)
    t_embd_destination = torch.tensor(np_destinations)
    t_embd_n_w_s = torch.tensor(np_embeddings_nodes_with_states)

    preds, preds_ecmp, _ = model.forward(
        network_state=t_states.to(device=DEV),
        network_state_mask=t_all_masks.to(device=DEV),
        embeddings_neighbors=t_embd_neighbors.to(device=DEV),
        mask_embeddings=t_neighbor_masks.to(device=DEV),
        embd_current_location=t_embd_cur_loc.to(device=DEV),
        embd_destination=t_embd_destination.to(device=DEV),
        embeddings=t_embd_n_w_s.to(device=DEV),
        embd_nodes_state=t_states.to(device=DEV)
    )

    batches = []
    for i in range(t_states.shape[0]):
        nodes = []
        for j in range(t_states.shape[1]):
            out = hlsa_model(t_states[i, j, ...].unsqueeze(0).unsqueeze(0))
            if i == 0 and j == 0:
                for k in range(128):
                    print(f"{int(out.detach().numpy()[0, 0, k])}, {'' if k % 2 == 0 else '  '}", end='')
                print()
            nodes.append(out)
        batches.append(torch.cat(nodes, dim=1))
    hnsas = torch.cat(batches, dim=0)
    np_hnsas = hnsas.detach().numpy()
    np.save('/opt/project/scripts/sample-hnsas.npy', np_hnsas)
    convert_hsnas()
    print(np_hnsas.shape)

    outputs = []
    for i in range(t_states.shape[0]):
        out = forward_model(
            hnsas[i, ...].unsqueeze(0),
            t_embd_cur_loc[i, :].unsqueeze(0),
            t_embd_destination[i, :].unsqueeze(0),
            t_embd_n_w_s
        )
        outputs.append(out)
    outputs = torch.cat(outputs, dim=0)
    preds_p = torch.sigmoid(preds)
    outputs_p = torch.sigmoid(outputs)
    outputs_np = outputs_p.detach().numpy()
    print((preds_p - outputs_p).sum())
    print("done")


if __name__ == '__main__':
    a = export_hlsa_model()
    b = export_forward_model()
    verify_consistency(a, b)
    export_forward_model_to_bin()
    export_hlsa_model_to_bin()
    print("all done")
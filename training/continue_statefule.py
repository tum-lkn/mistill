"""
Implements functionality that lets you resume training of an existing trial.
"""
import os
import torch
import argparse
import json
from torch.utils.data import DataLoader
import numpy as np
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from typing import Dict, Any
from collections import OrderedDict

import models.stateful as stateful
from models.utils import full_cross_entropy
from dataprep.datasets import StatefulDataset, filter_dataset
from dataprep.input_output import read_embeddings, read_graph
from training.utils import CustomStopper
from training.stateful import METRIC_KEY, StateTrainable


if torch.cuda.is_available():
    DEV = torch.device("cuda:0")
    PARALLEL = torch.cuda.device_count() > 1
else:
    DEV = torch.device("cpu")
    PARALLEL = False


def _get_highest_chechpoint(trial_dir: str) -> str:
    iters = [int(x.split('_')[1]) for x in os.listdir(trial_dir) if x.startswith("check")]
    max_iter = np.max(iters)
    return os.path.join(trial_dir, 'checkpoint_{:d}'.format(max_iter))


if __name__ == '__main__':
    ray.init()
    parser = argparse.ArgumentParser()
    # The template dataset contains placeholders for the embeddings. Computing
    # all the shortest paths and creating the targets is quite time consuming,
    # so it is done once and can then be re-used across experiments. The
    # template data-set contains placeholders that are then replaced by the
    # correct embedding when loading the data.
    parser.add_argument(
        "dataset",
        help="Path to the HDF5 file storing the template dataset for which models should be trained.")
    parser.add_argument(
        "embeddings",
        help="Path to the HDF5 file storing the embeddings for which the models should be trained.")
    parser.add_argument(
        "name",
        help="Name of the experiment."
    )
    parser.add_argument(
        "result_dir",
        help="Directory in which results should be stored."
    )
    parser.add_argument(
        'seed',
        type=int,
        help="Seed for the random number generator."
    )
    parser.add_argument(
        '--filter',
        type=str,
        default='None',
        help='Prefix to filter current locations with.'
    )
    parser.add_argument(
        "--embeddings-as-queries-for-links",
        action="store_true",
        help='Use the embeddings of nodes with states as queries for attention over links.'
    )
    parser.add_argument(
        "--sample-weights",
        action="store_true",
        help='Weight samples with their inverse occurrence in the loss function.'
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Finish quickly for testing"
    )
    parsed_args, _ = parser.parse_known_args()
    parsed_args.trial_dir = "/opt/project/data/training-results/" + \
                            "ContinueFatTreeK8LinkFailuresIpEmbeddingEmbsAsQsDstsAsQsHost2Host01/" + \
                            "StateTrainable_45b4f_00000_0_2020-11-23_08-38-33"
    parsed_args.learn_embeddings = False

    with open(os.path.join(parsed_args.trial_dir, 'params.json'), 'r') as fh:
        config = json.load(fh)

    dataset_train = StatefulDataset.from_hdf5_files(
        full_path_templates_dir=os.path.join(parsed_args.dataset, 'train'),
        full_path_embeddings=None if parsed_args.learn_embeddings else parsed_args.embeddings
    )
    dataset_train.use_embeddings_as_queries_for_attention_over_links = parsed_args.embeddings_as_queries_for_links
    dataset_train.generate_weights = parsed_args.sample_weights
    if parsed_args.filter == 'None':
        pass
    else:
        dataset_train = filter_dataset(
            ds=dataset_train,
            graph=read_graph('/opt/project/data/fat-tree-k8/fat-tree-k8.json'),
            prefix=parsed_args.filter
        )
    print("==============================> Finished building dataset")
    dataset_val = StatefulDataset.from_hdf5_files(
        full_path_templates_dir=os.path.join(parsed_args.dataset, 'val'),
        full_path_embeddings=None if parsed_args.learn_embeddings else parsed_args.embeddings
    )
    dataset_val.use_embeddings_as_queries_for_attention_over_links = parsed_args.embeddings_as_queries_for_links
    dataset_val.generate_weights = False
    if parsed_args.filter == 'None':
        pass
    else:
        dataset_val = filter_dataset(
            ds=dataset_val,
            graph=read_graph('/opt/project/data/fat-tree-k8/fat-tree-k8.json'),
            prefix=parsed_args.filter
        )
    print("==============================> Finished building dataset")
    dataset = dataset_train

    TRAINING_DATA_IDS = {k: ray.put(v) for k, v in dataset_train.to_dict().items()}
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Put training set in memory")
    VALIDATION_DATA_IDS = {k: ray.put(v) for k, v in dataset_train.to_dict().items()}
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Put validation set in memory")

    config['val_ids'] = VALIDATION_DATA_IDS
    config['train_ids'] = TRAINING_DATA_IDS



    # For dim_out_fcn lists of length 1-3 are sampled, i.e., the dense part
    # of the model has one to three hidden layers of varying size.
    random = np.random.RandomState(seed=parsed_args.seed)

    print("RESTORE MODEL IN ", parsed_args.trial_dir)
    analysis = tune.run(
        StateTrainable,
        resources_per_trial={'cpu': 1, 'gpu': 0.25},
        num_samples=1,
        # restore=os.path.join(_get_highest_chechpoint(parsed_args.trial_dir), 'model.pth'),
        restore=_get_highest_chechpoint(parsed_args.trial_dir),
        config=config,
        name=parsed_args.name,
        checkpoint_at_end=True,
        checkpoint_freq=10,
        checkpoint_score_attr='min-' + METRIC_KEY + '-val',
        local_dir=parsed_args.result_dir,
        sync_on_checkpoint=False
    )



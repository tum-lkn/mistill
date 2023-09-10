"""
    Implements functionality to train a shortest path based model.

    Usage:
        python -m training.sp_training <templates> <embeddings> <name> <result-dir>
            <num-samples> <seed> [--smoke-test --learn_embeddings]

    Args:
        templates           Path to the HDF5 file that contains the templates for the
                            training data.
        embeddings          Path to the HDF5 file that contains the node embeddings.
        name                Name of the experiment.
        result-dir          Folder in which tune will write its checkpoints and
                            the training results.
        num-samples         The number of trials tune will sample and train.
        seed                Seed for the random number generater used to sample
                            hyperparameters for trials.
        smoke-test          If set to true execute 10 epochs only. Use for testing.
        learn_embeddings    Learn an embedding instead of using a pre-computed
                            one. If this option is set argument <embeddings>
                            is ignored.

    Note:
        The dataset is split into templates and embeddings. The templates contain
        placeholders for the node embeddings. This script replaces the placeholder
        with the actual embeddings.
        The background of doing this is that the shortes path computations required
        to get the training data is rather expensive. Instead of doing those computations
        per node embedding, its done once and the node embedding is replaced
        accordingly.
        For further information on the format of the embeddings and the templates
        seed the file `dataprep/input_output.py` and the functions
        `write_distributional_spf_dataset` and `write_embedding` respectively. In
        addition see `dataprep/sp_prep.py` function `distributional_spf_dataset`.

    Note:
        In case you wonder why random search and not a grid search check out the
        following publication: https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf
"""
import os
import torch
import argparse
from torch.utils.data import DataLoader
import numpy as np
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from typing import Dict, Any
from ray.tune.utils import pin_in_object_store, get_pinned_object

import models.sponly as sponly
from models.utils import full_cross_entropy, multi_class_loss
from dataprep.datasets import DistributionalSpfDataSet, DistributionalDataSetWithoutEmbedding
from training.utils import CustomStopper

METRIC_KEY = 'cross_entropy'
if torch.cuda.is_available():
    DEV = torch.device("cuda:0")
    PARALLEL = torch.cuda.device_count() > 1
else:
    DEV = torch.device("cpu")
    PARALLEL = False


class SpTrainable(tune.Trainable):
    """
    Trainable for shortest path models.
    """
    def setup(self, config: Dict[str, Any]):
        """
        Create a single trainable model.

        Args:
            config: Dict that contains all hyper parameters. Contains the
                keys `model_config` with the configuration options for the
                model and key `optimizer` with parameters for the optimizer
                to use.
        """
        self.sp_config = sponly.SpfConfig.from_dict(config['model_config'])
        self.model = {
            'SpfModel': sponly.SpfModel,
            'EmbeddingSpfModel': sponly.EmbeddingSpfModel
        }[config['model']](self.sp_config)

        if PARALLEL:
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(DEV)
        self.distributional = config['distributional']

        self.batch_size = config['batch_size']
        self.num_epochs_per_train_call = config['num_epochs_per_train_call']
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=config['optimizer']['lr'],
            weight_decay=config['optimizer']['weight_decay']
        )

    def _loss_batch(self, loss_fct, sample, opt=None):
        """
        Perform forward and backward pass for one minibatch. In case of training,
        the optimizer changes the parameter. In case of validation, this does
        not happen.

        Args:
            loss_fct:
            sample:
            opt:

        Returns:

        """
        pred, scores = self.model(
            queries=sample['destination'].to(DEV),
            others=sample['neighbors'].to(DEV),
            mask=sample['attention_mask'].to(DEV)
        )
        loss = loss_fct(pred, sample['target'].to(DEV), torch.tensor(np.array([[1.]], dtype=np.float32)).to(DEV))

        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()

        return loss.cpu().item(), sample['target'].shape[0]

    def _get_data_set(self, store_id):
        """
        Create a dataset from a memory pinned dictionary containing the
        relevant data.

        Args:
            store_id:

        Returns:

        """
        dataset_dict = get_pinned_object(store_id)
        dataset = DistributionalSpfDataSet(
            neighbors=dataset_dict['neighbors'],
            destinations=dataset_dict['destinations'],
            attention_masks=dataset_dict['attention_masks'],
            targets=dataset_dict['targets'],
            current_locations=dataset_dict['current_locations']
        )
        return dataset

    def _train(self):
        """
        Train the model for the specified number of epochs.

        Returns:

        """
        val_loss = 1e9
        loader_train = DataLoader(self._get_data_set(TRAINING_DATA_ID), batch_size=self.batch_size, shuffle=True)
        loader_val = DataLoader(self._get_data_set(VALIDATION_DATA_ID), batch_size=2 * self.batch_size)

        if self.distributional:
            loss_fct = full_cross_entropy
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss_fct = multi_class_loss

        for epoch in range(self.num_epochs_per_train_call):
            self.model.train()
            for batch, sample in enumerate(loader_train):
                self._loss_batch(loss_fct, sample, self.optimizer)

            self.model.eval()
            with torch.no_grad():
                losses, nums = zip(
                    *[self._loss_batch(loss_fct, sample) for sample in loader_val]
                )
                val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
                # print(epoch, val_loss)
        return {METRIC_KEY: val_loss}

    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir

    def _restore(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        self.model.load_state_dict(torch.load(checkpoint_path))


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
        'num_samples',
        type=int,
        help="Number of samples that should be instantaiated."
    )
    parser.add_argument(
        'seed',
        type=int,
        help="Seed for the random number generator."
    )
    parser.add_argument(
        "--learn-embeddings",
        action="store_true",
        help="Learn the embeddings jointly with the outputs."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Finish quickly for testing"
    )
    parsed_args, _ = parser.parse_known_args()

    dataset_class = DistributionalDataSetWithoutEmbedding if parsed_args.learn_embeddings else DistributionalSpfDataSet
    dataset = dataset_class.from_hdf5(parsed_args.embeddings, parsed_args.dataset)
    print(">>>>>>>>>>>>>>>>>>>>>>>>> FINISHED BUILDING DATASET")

    random = np.random.RandomState(seed=1)
    x = np.arange(dataset.neighbors.shape[0])
    random.shuffle(x)
    val_start = int(x.size * 0.9)

    TRAINING_DATA_ID = pin_in_object_store({
        "neighbors": dataset.neighbors[x[:val_start]].copy(),
        "destinations": dataset.destinations[x[:val_start]].copy(),
        "attention_masks": dataset.attention_masks[x[:val_start]].copy(),
        "targets": dataset.targets[x[:val_start]].copy(),
        "current_locations": dataset.current_locations[x[:val_start]].copy()
    })
    VALIDATION_DATA_ID = pin_in_object_store({
        "neighbors": dataset.neighbors[x[val_start:]].copy(),
        "destinations": dataset.destinations[x[val_start:]].copy(),
        "attention_masks": dataset.attention_masks[x[val_start:]].copy(),
        "targets": dataset.targets[x[val_start:]].copy(),
        "current_locations": dataset.current_locations[x[val_start:]].copy()
    })

    asha = ASHAScheduler(
        time_attr='training_iteration',
        metric=METRIC_KEY,
        mode='min',
        max_t=11 if parsed_args.smoke_test else 1000,
        grace_period=5,
        reduction_factor=2
    )

    # For dim_out_fcn lists of length 1-3 are sampled, i.e., the dense part
    # of the model has one to three hidden layers of varying size.
    random = np.random.RandomState(seed=parsed_args.seed)
    search_space = {
        "seed": parsed_args.seed,
        "num_epochs_per_train_call": 1,
        "batch_size": tune.sample_from(lambda _: int(random.randint(64, 256))),
        "distributional": True if type(dataset) == DistributionalSpfDataSet else False,
        # "model": "SpfModel",
        "model": "EmbeddingSpfModel",
        "model_config": {
            "num_nodes": 100,
            "mode": "attn",  # "fcn
            "max_degree": dataset.max_degree,
            "dim_embedding": 24 if parsed_args.learn_embeddings else dataset.embedding_dim,
            # "dim_embedding": tune.sample_from(lambda _: int(
            #     random.randint(8, 25)
            # )) if parsed_args.learn_embeddings else dataset.embedding_dim,
            "num_heads": tune.sample_from(lambda _: int(random.randint(1, 5))),
            "dim_attn_hidden": tune.sample_from(lambda _: int(random.randint(4, 33))),
            "dim_attn_out": tune.sample_from(lambda _: int(random.randint(4, 33))),
            "dim_out_fcn": tune.sample_from(lambda _: random.choice(
                np.arange(8, 100),
                replace=True,
                size=random.choice([1, 2, 3])).tolist())
        },
        "optimizer": {
            "lr": tune.sample_from(lambda _: float(10. ** (-1 * random.uniform(3, 4)))),
            "weight_decay": 0
        }
    }

    analysis = tune.run(
        SpTrainable,
        resources_per_trial={'cpu': 1, 'gpu': 0.25},
        num_samples=parsed_args.num_samples,
        scheduler=asha,
        config=search_space,
        name=parsed_args.name,
        checkpoint_at_end=True,
        checkpoint_freq=10,
        checkpoint_score_attr='min-' + METRIC_KEY,
        local_dir=parsed_args.result_dir,
        sync_on_checkpoint=False
    )
    print("Best config: ", analysis.get_best_config(metric=METRIC_KEY))

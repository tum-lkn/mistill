"""
    Implements functionality to train a network state based model.

    Usage:
        python -m training.stateful <dataset> <embeddings> <name> <result-dir>
            <num-samples> <seed> [--smoke-test --learn-embeddings]

    Args:
        dataset             Path to the HDF5 file that contains the templates for the
                            training data, or a directory that contains multiple
                            files that make up the training data.
        embeddings          Path to the HDF5 file that contains the node embeddings.
        name                Name of the experiment.
        result-dir          Folder in which tune will write its checkpoints and
                            the training results.
        num-samples         The number of trials tune will sample and train.
        seed                Seed for the random number generater used to sample
                            hyperparameters for trials.
        smoke-test          If set to true execute 10 epochs only. Use for testing.
        learn-embeddings    Learn an embedding instead of using a pre-computed
                            one. If this option is set argument <embeddings>
                            is ignored.

    Note:
        The dataset is split into templates and embeddings. The templates contain
        placeholders for the node embeddings. This script replaces the placeholder
        with the actual embeddings.
        The background of doing this is that the computations required
        to get the training data is rather expensive. Instead of doing those computations
        per node embedding, its done once and the node embedding is replaced
        accordingly.
        For further information on the format of the embeddings and the templates
        seed the file `dataprep/input_output.py` and the functions
        `write_link_failure_dataset` and `write_embedding` respectively. In
        addition see `dataprep/link_failures.py` function `get_link_failure_dataset`.

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
from collections import OrderedDict

import models.stateful as stateful
from models.utils import full_cross_entropy, multi_class_loss
from dataprep.datasets import StatefulDataset, StatefulGPUDataset, filter_dataset
from dataprep.input_output import read_graph
from training.utils import expand_stateful_config
from dataprep.link_failures import OUTPUT_HULA, OUTPUT_LCP, OUTPUT_WCMP, OUTPUT_ECMP

METRIC_KEY = 'cross_entropy'
if torch.cuda.is_available():
    DEV = torch.device("cuda:0")
    PARALLEL = torch.cuda.device_count() > 1
else:
    DEV = torch.device("cpu")
    PARALLEL = False
PARALLEL = False


GPU_DATASET = False
CHECKPOINT_FREQ = 5


class StateTrainable(tune.Trainable):
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
        self._given_config = expand_stateful_config(config)
        self.sp_config = stateful.StatefulConfig.from_dict(self._given_config['model_config'])
        self.model = {
            'StatefulModel': stateful.StatefulModel,
            'EmbeddingStatefulModel': stateful.EmbeddingStatefulModel
        }[config['model']](self.sp_config)

        if PARALLEL:
            self.model = torch.nn.DataParallel(self.model)

        self.distributional = config['distributional']
        self.model = self.model.to(DEV)
        self.batch_size = config['batch_size']
        self.num_epochs_per_train_call = config['num_epochs_per_train_call']
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=config['optimizer']['lr'],
            weight_decay=config['optimizer']['weight_decay']
        )
        self.val_loss = 1e9
        self.val_loss_single = 1e9
        self.val_loss_ecmp = 1e9

        if self._given_config['use_gpu_dataset']:
            dataset_t = StatefulGPUDataset.from_hdf5_files(
                full_path_templates_dir=os.path.join(self._given_config['dataset'], 'train'),
                full_path_embeddings=self._given_config['embedding_path']
            )
            dataset_v = StatefulGPUDataset.from_hdf5_files(
                full_path_templates_dir=os.path.join(self._given_config['dataset'], 'val'),
                full_path_embeddings=self._given_config['embedding_path']
            )
        else:
            # dataset_t = StatefulDataset(**{k: ray.get(v) for k, v in TRAINING_DATA_IDS.items()})
            # dataset_v = StatefulDataset(**{k: ray.get(v) for k, v in VALIDATION_DATA_IDS.items()})
            dataset_t = StatefulDataset(**{k: ray.get(v) for k, v in config['train_ids'].items()})
            dataset_v = StatefulDataset(**{k: ray.get(v) for k, v in config['val_ids'].items()})
        if self.sp_config.multiclass:
            dataset_t.binarize_targets = True
            dataset_v.binarize_targets = True

        if self.sp_config.hlsa_model == 'attn':
            dataset_t.expand_state = True
            dataset_v.expand_state = True
        self.loader_train = DataLoader(dataset_t, num_workers=3, batch_size=self.batch_size, shuffle=True)
        self.loader_val = DataLoader(dataset_v, num_workers=3, batch_size=self.batch_size)
        # self.loader_train = DataLoader(dataset_t, batch_size=self.batch_size, shuffle=True)
        # self.loader_val = DataLoader(dataset_v, batch_size=self.batch_size)

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
        pred, pred_ecmp, reg = self.model(
            network_state=sample["network_state"].to(DEV),
            network_state_mask=sample["network_masks"].to(DEV),
            embeddings_neighbors=sample["neighbors"].to(DEV),
            mask_embeddings=sample["neighbor_mask"].to(DEV),
            embd_current_location=sample[self.sp_config.hlsa_attn_key].to(DEV),
            embd_destination=sample["destination"].to(DEV),
            embeddings=sample['embeddings'].to(DEV),
            embd_nodes_state=sample['embd_nodes_state'].to(DEV)
            # hlsa_attn_head_activations=None if sample['hlsa_attn_activations'] is None \
            #     else sample['hlsa_attn_activations'].to(DEV)
        )
        loss_single = loss_fct(
            logits=pred,
            target=sample['target'].to(DEV),
            weights=sample['weights'].to(DEV)
        )
        loss_ecmp = multi_class_loss(
            logits=pred_ecmp,
            target=sample['target_ecmp'].to(DEV),
            weights=sample['weights'].to(DEV)
        )
        loss = loss_single + loss_ecmp

        if opt is not None:
            loss += reg
            loss.backward()
            opt.step()
            opt.zero_grad()

        return loss.cpu().item(), loss_single.cpu().item(), loss_ecmp.cpu().item(), \
               sample['target'].shape[0]

    def _train_with_split_dataset(self):
        """
        Train the model for the specified number of epochs.

        Returns:

        """
        if self.distributional:
            loss_fct = full_cross_entropy
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        samples_trained = 0
        while samples_trained < 100000:
            self.model.train()
            for batch, sample in enumerate(self.loader_train):
                self._loss_batch(loss_fct, sample, self.optimizer)
                samples_trained += self.batch_size
                if samples_trained >= 100000:
                    break

        self.model.eval()
        losses = []
        nums = []
        with torch.no_grad():
            samples_evaluated = 0
            while samples_evaluated < 10000:
                for sample in self.loader_val:
                    loss, num = self._loss_batch(loss_fct, sample)
                    losses.append(loss)
                    nums.append(num)
                    samples_evaluated += num
                    if samples_evaluated >= 10000:
                        break
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            # print(epoch, val_loss)
        return {METRIC_KEY: val_loss}

    def step(self):
        """
        Train the model for the specified number of epochs.

        Returns:

        """
        train_loss = 1e9
        train_loss_single = 1e9
        train_loss_ecmp = 1e9
        if self.sp_config.multiclass:
            loss_fct_train = multi_class_loss
            loss_fct_val = multi_class_loss
        else:
            loss_fct_train = full_cross_entropy
            loss_fct_val = full_cross_entropy

        for epoch in range(self.num_epochs_per_train_call):
            self.model.train()
            losses, losses_single, losses_ecmp, nums = zip(
                *[self._loss_batch(loss_fct_train, sample, self.optimizer) for sample in self.loader_train]
            )
            train_loss = np.sum(np.multiply(losses, nums) / np.sum(nums))
            train_loss_single = np.sum(np.multiply(losses_single, nums) / np.sum(nums))
            train_loss_ecmp = np.sum(np.multiply(losses_ecmp, nums) / np.sum(nums))

            if self.iteration % CHECKPOINT_FREQ == 0:
                self.model.eval()
                with torch.no_grad():
                    losses, losses_single, losses_ecmp, nums = zip(
                        *[self._loss_batch(loss_fct_val, sample) for sample in self.loader_val]
                    )
                    self.val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
                    self.val_loss_single = np.sum(np.multiply(losses_single, nums) / np.sum(nums))
                    self.val_loss_ecmp = np.sum(np.multiply(losses_ecmp, nums) / np.sum(nums))
                    # print(epoch, val_loss)
            self.model.reduce_tau(0.023)
        return {METRIC_KEY: train_loss, '{}-val'.format(METRIC_KEY): self.val_loss,
                "{}_single".format(METRIC_KEY): train_loss_single,
                "{}_ecmp".format(METRIC_KEY): train_loss_ecmp,
                "{}_single-val".format(METRIC_KEY): self.val_loss_single,
                "{}_ecmp-val".format(METRIC_KEY): self.val_loss_ecmp
                }

    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir

    def _restore(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        # The state dict could be saved on a GPU. So we might have to move the
        # data storage to CPU.
        state_dict = torch.load(checkpoint_path, map_location=torch.device(DEV))

        if not PARALLEL:
            # If data parallel is used then the parameters are prefixed with .model.
            # When loading the model not in data parallel mode, then this .model is
            # not expected and results in an error.
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]
                new_state_dict[k] = v

        self.model.load_state_dict(new_state_dict)


if __name__ == '__main__':
    ray.init()
    parser = argparse.ArgumentParser()
    # The template dataset contains placeholders for the embeddings. Computing
    # all the shortest paths and creating the targets is quite time consuming,
    # so it is done once and can then be re-used across experiments. The
    # template data-set contains placeholders that are then replaced by the
    # correct embedding when loading the data.
    parser.add_argument(
        "policy",
        help="Routing policy to train, should be in {ECMP, WCMP, HULA, LCP}.")
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
        '--filter',
        type=str,
        default='None',
        help='Prefix to filter current locations with.'
    )
    parser.add_argument(
        "--learn-embeddings",
        action="store_true",
        help="Learn the embeddings jointly with the outputs."
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
        "--multiclass",
        action="store_true",
        help='Whether to use multi-class instead of distribution.'
    )
    parser.add_argument(
        "--cur-loc-and-dst-q-hlsa",
        action="store_true",
        help='Whether to use combination of the embedding of the current location and destination as key for the attention mechanism over the HLSAs. If not set, current location is used.'
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Finish quickly for testing"
    )
    parsed_args, _ = parser.parse_known_args()
    assert parsed_args.policy.lower() in [OUTPUT_ECMP, OUTPUT_HULA, OUTPUT_LCP, OUTPUT_WCMP]
    print("MULTICLAAS IS", parsed_args.multiclass)

    if not GPU_DATASET:
        dataset_train = StatefulDataset.from_hdf5_files(
            full_path_templates_dir=os.path.join(parsed_args.dataset, 'train'),
            policy=parsed_args.policy.lower(),
            full_path_embeddings=None if parsed_args.learn_embeddings else parsed_args.embeddings,
            ignore_masks=True
        )
        dataset_train.use_embeddings_as_queries_for_attention_over_links = parsed_args.embeddings_as_queries_for_links
        dataset_train.binarize_targets = parsed_args.multiclass
        dataset_train.generate_weights = parsed_args.sample_weights
        dataset_train.load_weights(parsed_args.dataset)
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
            policy=parsed_args.policy.lower(),
            full_path_embeddings=None if parsed_args.learn_embeddings else parsed_args.embeddings,
            ignore_masks=True
        )
        dataset_val.use_embeddings_as_queries_for_attention_over_links = parsed_args.embeddings_as_queries_for_links
        dataset_val.binarize_targets = parsed_args.multiclass
        dataset_val.generate_weights = False
        dataset_val.load_weights(parsed_args.dataset)
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
    else:
        dataset = StatefulDataset.from_hdf5(
            full_path_embeddings=None if parsed_args.learn_embeddings else parsed_args.embeddings,
            full_path_templates=os.path.join(parsed_args.dataset, 'train', 'link-failure-data-0.h5')
        )

    asha = ASHAScheduler(
        time_attr='training_iteration',
        metric=METRIC_KEY + '-val',
        mode='min',
        max_t=11 if parsed_args.smoke_test else 150,
        grace_period=5,
        reduction_factor=2
    )

    # For dim_out_fcn lists of length 1-3 are sampled, i.e., the dense part
    # of the model has one to three hidden layers of varying size.
    tmp = [32, 16, 8]
    nblocks = tmp[parsed_args.seed % 3]
    random = np.random.RandomState(seed=parsed_args.seed)
    search_space = {                                                            
        "val_ids": VALIDATION_DATA_IDS,                                         
        "train_ids": TRAINING_DATA_IDS,                                         
        "use_gpu_dataset": GPU_DATASET,                                         
        "embedding_path": parsed_args.embeddings,                               
        "dataset": parsed_args.dataset,                                         
        "seed": parsed_args.seed,                                               
        "num_epochs_per_train_call": 1,                                         
        "batch_size": tune.sample_from(lambda _: int(random.randint(128, 256))),
        "distributional": True,                                                 
        "use_embeddings_as_queries_for_attention_over_links": parsed_args.embeddings_as_queries_for_links,
        "model": "EmbeddingStatefulModel" if parsed_args.learn_embeddings else "StatefulModel",
        "model_config": {                                                       
            "hlsa_model": "fcn",  # tune.sample_from(lambda x: random.choice(["attn", "fcn"])),
            "policy": parsed_args.policy.lower(),                               
            # "hlsa_model": None,                                               
            "neighbor_model": "fcn",                                            
            "cur_loc_and_dst_q_hlsa": parsed_args.cur_loc_and_dst_q_hlsa,       
            "hlsa_attn_key": "current_loc",                                     
            "multiclass": parsed_args.multiclass,                               
            "max_degree": dataset.max_degree,                                   
            "num_nodes": dataset.num_nodes,                                     
            "num_nodes_with_state": dataset.num_nodes_with_state,               
            "alpha_l1_hlsa_attn_weights": 0.,  # tune.sample_from(lambda _: float(10. ** (-1 * random.randint(3, 7)))),
            "alpha_l1_hlsas": 0.,  #tune.sample_from(lambda _: float(10. ** (-1 * random.randint(2, 6)))),
            "dim_embedding": tune.sample_from(lambda _: int(                    
                random.randint(8, 24)                                           
            )) if parsed_args.learn_embeddings else dataset.embedding_dim,      
            "pool_links": 'squeeze' if parsed_args.embeddings_as_queries_for_links else \
                tune.sample_from(lambda _: str(random.choice(["sum", "average", "max"]))),
            "packets_droppeable": True,                                         
            "final_fcns": tune.sample_from(lambda _: random.randint(50, 129, size=random.randint(2, 4)).tolist()),# [92, 64],
            "link_attns": [                                                     
                {                                                               
                    "num_heads": tune.sample_from(lambda _: int(random.randint(1, 6))),
                    "dim_fcn": tune.sample_from(lambda _: int(random.randint(32, 150))),
                    "dim_hidden": tune.sample_from(lambda _: int(random.randint(12, 33))),
                    "dim_out": tune.sample_from(lambda _: int(random.randint(12, 33))),
                    "dim_in": 13, #dataset.num_features_state,                  
                    "attn_activation": "softmax",                               
                    "dim_q": dataset.embedding_dim if parsed_args.embeddings_as_queries_for_links else None,  #dataset.num_features_state
                } for _ in range(random.randint(1, 2 if parsed_args.embeddings_as_queries_for_links else 4))
            ],                                                                  
            "hlsa_attns": {                                                     
                "num_heads": tune.sample_from(lambda _: int(random.randint(9, 15))),#10,
                "dim_fcn": tune.sample_from(lambda _: int(random.randint(70, 129))), # 70, # 
                "dim_hidden": tune.sample_from(lambda _: int(random.randint(20, 65))), #30,  # 
                "dim_out": tune.sample_from(lambda _: int(random.randint(16, 65))), # 20,  # 
                # Set inside the setup of the trainable. Is the output dim of   
                # the last link_attns config.                                   
                "dim_in": None,                                                 
                "dim_q": None,                                                  
                "dim_k": dataset.embedding_dim,                                 
                "attn_activation": "gs"                                         
            },                                                   
            "neighbor_attns": {                                                 
                "num_heads": 5,                                                 
                "dim_fcn": 90,  # tune.sample_from(lambda _: int(random.randint(64, 129))),
                "dim_hidden": 30,                                               
                "dim_out": 30,                                                  
                "dim_in": None                                                  
            },                                                                  
            "hlsa_gs": {                                                        
                "temperature": 0.6,                                             
                "arity": 2,                                                     
                "num_blocks": 100,  # tune.sample_from(lambda _: int(np.random.randint(1, 63)))
            }                                                                   
            # "neighbor_attns": {                                               
            #     "num_heads": tune.sample_from(lambda _: int(random.randint(1, 6))),
            #     "dim_fcn": tune.sample_from(lambda _: int(random.randint(5, 129))),
            #     "dim_hidden": tune.sample_from(lambda _: int(random.randint(4, 33))),
            #     "dim_out": tune.sample_from(lambda _: int(random.randint(4, 33))),
            #     "dim_in": None                                                
            # }                                                                 
        },                                                                      
        "optimizer": {                                                          
            "lr": tune.sample_from(lambda _: float(10. ** (-1 * random.uniform(3.5, 4.3)))),
            "weight_decay": 0                                                   
        }                                                                       
    }                      

    analysis = tune.run(
        StateTrainable,
        resources_per_trial={'cpu': 1, 'gpu': 0.3},
        num_samples=parsed_args.num_samples,
        scheduler=asha,
        config=search_space,
        name=parsed_args.name,
        checkpoint_at_end=True,
        checkpoint_freq=CHECKPOINT_FREQ,
        checkpoint_score_attr='min-' + METRIC_KEY + '-val',
        local_dir=parsed_args.result_dir,
        sync_on_checkpoint=False,
        trial_name_creator=lambda trial: 'StateTrainable_{:s}'.format(trial.trial_id),
    )
    # print("Best config: ", analysis.get_best_config(metric=METRIC_KEY))

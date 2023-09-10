"""
Iterates recursively into the folder hierarchy, opens data sets and evaluates
the cross entropy therein.
"""
import torch
import argparse
import os
import h5py
import re
import numpy as np
from models.utils import full_cross_entropy
from dataprep.sp_prep import H5_TARGET
import logging
from typing import Any, Dict, List, Union, Tuple
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('baseline.py')


def evaluate_targets(full_path_to_file: str) -> Union[None, float]:
    """
    Takes path to a dataset file and calculates the full cross entropy of
    the target.

    Args:
        full_path_to_file:

    Returns:
        cross_entropy: Average cross entropy across the dataset or None if
            error occured.
    """
    logger.info("Evaluate {}".format(full_path_to_file))
    f = h5py.File(full_path_to_file, 'r')
    cross_entropy = None
    try:
        cross_entropy = np.mean([full_cross_entropy(
            torch.log(torch.tensor(np.expand_dims(f[k][H5_TARGET][()], 0)) + 1e-9),
            torch.tensor(np.expand_dims(f[k][H5_TARGET][()], 0))) for k in f])
    except Exception as e:
        logger.exception(e)
    finally:
        f.close()
    return cross_entropy


def _evaluate_file(pattern: re.Pattern, base_path: str, files: List[str]) -> List[Tuple[str, Union[float, None]]]:
    """
    For every file check if it contains a dataset. If it is a dataset evaluate
    the file. If not, descend deeper into the hierarchy.

    Args:
        pattern:
        base_path:
        files:
        rets:

    Returns:

    """
    rets = []
    for f in files:
        path = os.path.join(base_path, f)
        if os.path.isdir(path):
            rets = rets + _evaluate_file(pattern, path, os.listdir(path))
        elif re.match(pattern, f):
            rets = rets + [(path, evaluate_targets(path))]
        else:
            continue
    return rets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        help="Root folder into which should be descended."
    )
    parser.add_argument(
        "--pattern",
        help="Regular expression identifying dataset names.",
        default="spf-distributional\.h5"
    )
    parsed_args, _ = parser.parse_known_args()
    results = _evaluate_file(parsed_args.pattern, parsed_args.root, os.listdir(parsed_args.root))

    for f, e in results:
        print(f, e)

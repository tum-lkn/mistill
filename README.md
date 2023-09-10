# Source Code for Mistill: Distilling Distributed Network Protocols from Examples
This repository contains the source code accompanying the publication:
> P. Krämer, O. Zeidler, J. Zerwas, A. Blenk, and W. Kellerer, “Mistill: Distilling Distributed Network Protocols from Examples,” IEEE TNSM, pp. 1–16, Mar. 2023, doi: 10.1109/TNSM.2023.3263529.

**Disclaimer**. The source code in this repository has been made publicly
available for transparency and as contribution for the scientific community. The
source code reflects in most parts the state in which the results for the referenced
publications were obtained. The source code has mostly been left as is.

## Repository organization
The repository is organized as follows.

`avxnn`. This folder contains C++-code implementing the NN forward pass, exploiting
AVX512 acceleration and structural properties unique to the used NN structure.

`data_gen`. This folder contains Python scripts that generate data for specific
settings.

`dataprep`. This Python package contains modules that generate and manipulate
training data, embeddings, and supporting functionality for data handling.

`dc-mininet-emulation`. This folder contains Python and C++ code to start and
configure the Mininet emulator. The code configures a Fat-Tree, sets up forwarding
rules, and facilitates the sending of HNSAs from switches in the network.

`docker`. This folder contains the docker file encapsulating the project dependencies.

`ebpf`. This folder contains the code implementing the forwarding behavior for MPLs
on end-hosts using eBPF, and the handling of NN, eBPF Maps and user space functionality.

`embeddings`. This Python package implements various embedding methods for nodes
in graphs.

`eval`. This Python package contains code evaluating trained models and experiments.

`layers`. This Python package implemnts customized NN layers.

`models`. This Python package implements various Neural Architectures (NAs) that
were evaluated during this project.

`scripts`. This folder contains utility functions and stand-alone scripts. For example,
`to_torchscript_model.py` implements the export of Pytorch models using tracing
and a custom export serializing the weights into a binary file.

`topos`. This Python package implements functionality to generate a Fat-Tree topology.

`training`. This Python package contains functionality to train NNs for preconfigured
NAs over a large search space using the Ray Tune library.

## Reproducibility

Describes the step necessary to reproduce the results of the paper.

1) Change to the folder `docker` and run `docker build -f Dockerfile_torch -t pytorch-image`.
2) Update the path in the shell script `run_container.sh` to the location where you cloned the repository.
3) Execute the script `run_container.sh`.
4) In the docker container, change directory to `/opt/project`.
5) Run `python3 plot_results.py`.

This will start an interactive session in which the simulation results with the
trained models are reproduced and the plots from the paper recreated. The interactive
program asks you to enter one of three possible TE policies. Enter them in the order
`hula`, `lcp`, `wcmp`, `hula` (`hula` corresponds to the `MinMax` policy).

After the program finishes (which can take some time), the images will be located
in the `img/gs` and `img/sparsemax` folder. The simulation results are written to
the corresponding folders in `data/results/`.


### Neural Network Model
The final NN model of the publication can be found in `models/stateful.py`. Its the
class `StatefulModel`.


### Training Method
The training of the models can be inspected in the file `training/stateful.py`. At
the bottom is the definition of the search space.


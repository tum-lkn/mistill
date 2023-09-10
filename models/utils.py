import torch
from typing import List, Union
import layers.attn as attn


ACTIVATION_FACTORY = {
    'relu': torch.nn.ELU,
    'elu': torch.nn.ELU,
    'softmax': lambda: torch.nn.Softmax(dim=-1)
}


def full_cross_entropy(logits: torch.Tensor, target: torch.Tensor,
                       weights: torch.Tensor) -> torch.Tensor:
    """
    Compute the cross entropy between the target distribution `target` and the
    unnormalized predicted distribution `logits`. The loss is computed as follows:

    ```
    -\sum_{i} p_i \cdot \log(q_i)
    = -\sum_{i} p_i \cdot \log(\frac{\exp(logits_i)}{\sum_{j}\exp(logits_j)})
    = -\sum_{i} p_i \cdot (\log(\exp(logits_i)) - \log(\sum_{j}\exp(logits_j)))
    = -\sum_{i} p_i \cdot (logits_i - \logSumExp(logits))
    ```

    Args:
        logits: Unnormalized logits for a predicted distribution. Shape (BS, C).
        target: The target distribution. A normalized categorical distribution.
            The distribution elements are stored along the last dimension. Shape (BS, C).
        weights: Weights for individual samples. Has shape (BS, 1).

    Returns:
        loss: The cross entropy loss between the two distributions averaged
            across samples.
    """
    logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
    diff = torch.sub(logits, logsumexp)
    mul = torch.mul(target, diff)
    loss = torch.mul(-1. * torch.sum(mul, dim=-1, keepdim=True), weights)
    return torch.mean(loss)


def _multi_class_loss(probs: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    loss = torch.mul(
        weights,
        torch.sum(
            torch.add(
                # apply label smoothing: https://arxiv.org/pdf/1906.02629.pdf
                torch.mul(torch.clamp(target, 0.05, 0.95), torch.log(probs + 1e-6)),
                torch.mul(torch.clamp(1. - target, 0.05, 0.95), torch.log(1. - probs + 1e-6))
            ),
            dim=-1
        )
    )
    return -1. * torch.mean(loss)


def multi_class_loss(logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return _multi_class_loss(probs, target, weights)


def make_sequential(layer_sizes: List[int], dim_in: int, activation: str) -> torch.nn.Sequential:
    """
    Create a sequential model of fully connected layers with a specific activation
    function.
    Args:
        layer_sizes: Individual layer sizes.
        dim_in: Dimensionality of input of first layer.
        activation: Activation function, e.g torch.nn.ReLU, or torch.sigmoid, etc.
        last_layer_activation: None use linear output, or any other activation
            for the last layer.

    Returns:
        model: Sequential model consisting of linear layers and activations.
    """
    layers = []
    for i, layer_size in enumerate(layer_sizes):
        if i == 0:
            d_in = dim_in
        else:
            d_in = layer_sizes[i - 1]
        layers.append(torch.nn.Linear(d_in, layer_sizes[i], bias=True))
        layers.append(ACTIVATION_FACTORY[activation]())
    return torch.nn.Sequential(*layers)


def make_sequential_from_attn_configs(mha_configs: List[attn.MultiHeadAttentionModuleConfig]) -> torch.nn.Sequential:
    """
    Creates a sequence of dense layers from a list of multi-head attention module
    configs. Uses only the dense configuration part.

    Args:
        mha_configs:

    Returns:

    """
    layers = []
    for i, mha_config in enumerate(mha_configs):
        d_in = mha_config.dim_in if i == 0 else mha_configs[i - 1].dim_fcn
        layers.append(torch.nn.Linear(d_in, mha_config.dim_fcn, bias=True))
        layers.append(ACTIVATION_FACTORY[mha_config.fcn_activation]())
    return torch.nn.Sequential(*layers)


def make_sequential_attn_modules(configs: List[Union[attn.SelfAttentionModuleConfig,
                                 attn.MultiHeadAttentionModuleConfig]]) -> attn.SequentialAttention:
    """
    Make a sequential list of attention modules. Each module consists of a
    multi-head attention layer followed by a dense layer with relu activation.

    Args:
        configs:

    Returns:

    """
    layers = []
    for config in configs:
        layers.append(attn.AttentionModule(config))
    return attn.SequentialAttention(layers)



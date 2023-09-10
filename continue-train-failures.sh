#!/bin/bash
python -m training.continue_statefule \
    /opt/project/data/fat-tree-k8/link-failures-all-hosts \
    /opt/project/data/fat-tree-k8/fat-tree-k8-ip-embedding.h5 \
    ContinueFatTreeK8LinkFailuresIpEmbeddingEmbsAsQsDstsAsQsHost2Host01_2 \
    /opt/project/data/training-results \
    24 \
    --embeddings-as-queries-for-links


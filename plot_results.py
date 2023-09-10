import pandas as pd
import h5py
import numpy as np
import present
import scipy.stats
import os

import eval.lf_eval as lf_eval


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


def get_driver_averages(nom: h5py.Group, denom: h5py.Group) -> np.array:
    vals = []
    for i in range(10):
        idx = 'run_{:d}'.format(i + 1)
        vals.append(np.mean(np.divide(nom[idx][()], denom[idx][()])))
    return np.array(vals)


#==============================================================================
# Evaluate The GumbelSoftmax case
#==============================================================================
#------------------------------------------------------------------------------
# Evaluate MaxMin Policy
#------------------------------------------------------------------------------
if not os.path.exists('./data/results/gs/k8-hula-b64/driver-results.h5'):
    print("Generate driver results for MaxMin")
    lf_eval.run_driver(
        logdir='./data/results/gs/k8-hula-b64/StateTrainable_804cc_00097_97',
        dsetpath='./data/fat-tree-k8',
        filter='',
        result_path='./data/results/gs/k8-hula-b64/'
    )
f = h5py.File('./data/results/gs/k8-hula-b64/driver-results.h5', 'r')
fig, ax = present.get_fig(0.66)
present.compare_cdfs(
    cdfs=[
        [present._make_cdf(np.concatenate([f['ecmp_path_weights']['run_{:d}'.format(i)][()] for i in range(1, 11)]))],
        [present._make_cdf(np.concatenate([f['llsrp_path_weights']['run_{:d}'.format(i)][()] for i in range(1, 11)]))],
        [present._make_cdf(np.concatenate([f['opt_path_weights']['run_{:d}'.format(i)][()] for i in range(1, 11)]))]
    ],
    xlabel='Max Link Weight',
    ylabel="P(X<x)",
    labels=['ECMP', 'MISTILL', 'OPT'],
    alpha=1,
    ax=ax,
    markevery=0.2
)
present.save_fig('./img/gs/', 'hula_cdfs_weights', 'pdf', fig=fig)
f.close()

#------------------------------------------------------------------------------
# Evaluate Least Cost Policy
#------------------------------------------------------------------------------
if not os.path.exists('./data/results/gs/k8-lcp-b64/driver-results.h5'):
    print("Generate driver results for LCP")
    lf_eval.run_driver(
        logdir='./data/results/gs/k8-lcp-b64/StateTrainable_f7c1f_00097_97',
        dsetpath='./data/fat-tree-k8',
        filter='',
        result_path='./data/results/gs/k8-lcp-b64/'
    )
f = h5py.File('./data/results/gs/k8-lcp-b64/driver-results.h5', 'r')
fig, ax = present.get_fig(0.66)
present.compare_cdfs(
    cdfs=[
        [present._make_cdf(np.concatenate([f['ecmp_path_weights']['run_{:d}'.format(i)][()] for i in range(1, 11)]))],
        [present._make_cdf(np.concatenate([f['llsrp_path_weights']['run_{:d}'.format(i)][()] for i in range(1, 11)]))],
        [present._make_cdf(np.concatenate([f['opt_path_weights']['run_{:d}'.format(i)][()] for i in range(1, 11)]))]
    ],
    xlabel='Path Cost',
    ylabel="P(X<x)",
    labels=['ECMP', 'MISTILL', 'OPT'],
    alpha=1,
    ax=ax,
    markevery=0.2
)
present.save_fig('./img/gs/', 'lcp_cdfs_weights', 'pdf', fig=fig)
f.close()

#------------------------------------------------------------------------------
# Evaluate Weighted Cost Multipathing
#------------------------------------------------------------------------------
if not os.path.exists('./data/results/gs/k8-wcmp-b64/wcmp-results-uni.h5'):
    print("Generate driver results for WCMP")
    lf_eval.run_driver(
        logdir='./data/results/gs/k8-wcmp-b64/StateTrainable_b5ca9_00029_29',
        dsetpath='./data/fat-tree-k8',
        filter='',
        result_path='./data/results/gs/k8-wcmp-b64/'
    )
f = h5py.File('./data/results/gs/k8-wcmp-b64/wcmp-results-uni.h5', 'r')
fig, ax = present.get_fig(0.66)
present.compare_cdfs(
    cdfs=[
        [present._make_cdf(np.clip(np.concatenate([f['ecmp']['run_{:d}'.format(i)][()] for i in range(10)]), 0.01, 100))],
        [present._make_cdf(np.clip(np.concatenate([f['rexm']['run_{:d}'.format(i)][()] for i in range(10)]), 0.01, 100))],
    ],
    xlabel='KL Divergence',
    ylabel="P(X<x)",
    labels=['ECMP', 'MISTILL'],
    alpha=1,
    ax=ax,
    markevery=0.2
)
ax.set_xscale('log')
present.save_fig('./img/gs/', 'wcmp_cdfs_weights', 'pdf', fig=fig)
f.close()


#==============================================================================
# Evaluate The Sparsemax case, i.e., attention weights are fixed.
#==============================================================================

#------------------------------------------------------------------------------
# Evaluate MaxMin Policy
#------------------------------------------------------------------------------
if not os.path.exists('./data/results/sparsemax/k8-hula-b64/driver-results.h5'):
    lf_eval.run_driver(
        logdir='./data/results/sparsemax/k8-hula-b64/StateTrainable_a57f5_00007_7',
        dsetpath='./data/fat-tree-k8',
        filter='',
        result_path='./data/results/sparsemax/k8-hula-b64/'
    )
f = h5py.File('./data/results/sparsemax/k8-hula-b64/driver-results.h5', 'r')
fig, ax = present.get_fig(0.66)
present.compare_cdfs(
    cdfs=[
        [present._make_cdf(np.concatenate([f['ecmp_path_weights']['run_{:d}'.format(i)][()] for i in range(1, 11)]))],
        [present._make_cdf(np.concatenate([f['opt_path_weights']['run_{:d}'.format(i)][()] for i in range(1, 11)]))],
        [present._make_cdf(np.concatenate([f['llsrp_path_weights']['run_{:d}'.format(i)][()] for i in range(1, 11)]))],
    ],
    xlabel='Max Link Weight',
    ylabel="P(X<x)",
    labels=['ECMP', 'OPT', 'MISTILL'],
    alpha=0.7,
    ax=ax,
    markevery=500
)
present.save_fig('./img/sparsemax/', 'hula_cdfs_weights', 'pdf', fig=fig)
f.close()





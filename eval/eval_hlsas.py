import eval.utils as evutils
from models.stateful import StatefulConfig, StatefulModel
import numpy as np
import torch
import itertools
import matplotlib.pyplot as plt

policy = 'wcmp'
path = {
    'hula': '/opt/project/data/results/gs/k8-hula-b64/StateTrainable_804cc_00097_97',
    'lcp': '/opt/project/data/results/gs/k8-lcp-b64/StateTrainable_f7c1f_00097_97',
    'wcmp': '/opt/project/data/results/gs/k8-wcmp-b64/StateTrainable_b5ca9_00029_29'
}[policy]

model = evutils.load_model(
    StatefulConfig,
    StatefulModel,
    path,
    64
)
model.train(False)

state = np.repeat(np.random.RandomState(seed=1).uniform(0, 1, size=[1, 1, 8, 6]), repeats=80, axis=1)
edges = np.zeros((2**8, 16))
state = np.repeat(state, repeats=4, axis=0)
for i, (e1, e2, e3, e4, e5, e6, e7, e8) in enumerate(itertools.product([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])):
    d1, d2 = divmod(i, 80)
    for j, e in enumerate([e1, e2, e3, e4, e5, e6, e7, e8]):
        state[d1, d2, j, 0] = e
        state[d1, d2, j, 1] = 1 - e
        state[d1, d2, j, 2] = e
        state[d1, d2, j, 3] = 1 - e
        edges[i, 8 + j] = e
node_state = torch.tensor(state, dtype=torch.float32)
tmp = node_state.numpy()
msg = model.compute_hlsas(
    node_state, None, None
)
print(msg.shape)
all = msg.detach().numpy().reshape([-1, 128])[:256, np.arange(0, 128, 2)]
all = np.concatenate([all, edges], axis=1)
print("done")

ax = plt.subplot()
fig = plt.gcf()
fig.set_figwidth(all.shape[1] / 10.)
fig.set_figheight(all.shape[0] / 10.)
ax.imshow(all, cmap='Greens')
ax.set_yticks(np.arange(-0.5, all.shape[0], 1))
ax.set_xticks(np.arange(-0.5, all.shape[1], 1))
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.grid(color='white')
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.spines['left'].set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
plt.tight_layout()
plt.savefig(f'/opt/project/img/gs/pattern-link-failures-{policy}.pdf')
plt.close('all')

ax = plt.subplot()
fig = plt.gcf()
fig.set_figwidth(all.shape[1] / 10.)
fig.set_figheight(8 / 10. * 1.5)
ax.imshow(all[-8:, :], cmap='Greens')
ax.set_yticks(np.arange(-0.5, 8, 1))
ax.set_xticks(np.arange(-0.5, all.shape[1], 1))
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.grid(color='white')
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.spines['left'].set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.text(0, -2.4, "Bit0", {'fontsize': 10, 'fontfamily': 'serif'})
ax.text(59, -2.4, "Bit63", {'fontsize': 10, 'fontfamily': 'serif'})
ax.text(65, -2.4, "Edge0 Up", {'fontsize': 10, 'fontfamily': 'serif'})
ax.text(72.5, -2.4, "Edge8 up", {'fontsize': 10, 'fontfamily': 'serif'})
ax.arrow(0.4, -2.3, -.25, 1., width=0.1, color='black')
ax.arrow(63.6, -2.3, .25, 1., width=0.1, color='black')
ax.arrow(71.8, -2.3, .25, 1., width=0.1, color='black')
ax.arrow(78.8, -2.3, .25, 1., width=0.1, color='black')
# plt.tight_layout()
plt.subplots_adjust(left=-0.0, right=1., top=0.9, bottom=0.0)
plt.savefig(f'/opt/project/img/gs/pattern-link-failures-part-{policy}.pdf')
plt.close('all')

all = np.expand_dims(np.mean(all[:, :64], axis=0), axis=0)
ax = plt.subplot()
fig = plt.gcf()
fig.set_figwidth(all.shape[1] / 10.)
fig.set_figheight(all.shape[0] / 10.)
ax.imshow(all, cmap='Greens')
ax.set_yticks(np.arange(-0.5, all.shape[0], 1))
ax.set_xticks(np.concatenate((np.arange(-0.5, 65, 1), np.arange(71.5, 80, 1))))
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.grid(color='white', linewidth=1.5)
plt.tight_layout()
plt.savefig(f'/opt/project/img/gs/pattern-link-failures-avg-{policy}.pdf')
plt.close('all')


weights = np.random.RandomState(seed=1).uniform(0, 1, size=[3, 80, 8, 2])
node_state = torch.cat(
    (
        torch.ones([3, 80, 8, 4], dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32)
    ),
    dim=-1
)
msg = model.compute_hlsas(
    node_state, None, None
)
all = msg.detach().numpy().reshape([-1, 128])[:, np.arange(0, 128, 2)]

part = np.expand_dims(all[0, :], axis=0)
distances = np.sum(all != part, axis=1)
order = np.argsort(distances)
order = np.arange(distances.size)

all = np.concatenate((
    all[order, :],
    # np.zeros((all.shape[0], 4)),
    np.reshape(weights, [240, 16])[order, :]
), axis=1)

ax = plt.subplot()
fig = plt.gcf()
fig.set_figwidth(all.shape[1] / 10.)
fig.set_figheight(all.shape[0] / 10.)
ax.imshow(all, cmap='Greens')
ax.set_yticks(np.arange(-0.5, all.shape[0], 1))
ax.set_xticks(np.concatenate((np.arange(-0.5, 65, 1), np.arange(63.5, 80, 1))))
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.grid(color='white', linewidth=1.5)
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.spines['left'].set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
plt.tight_layout()
plt.savefig(f'/opt/project/img/gs/pattern-edge-weights-{policy}.pdf')
plt.close('all')

ax = plt.subplot()
fig = plt.gcf()
fig.set_figwidth(all.shape[1] / 10.)
fig.set_figheight(8 / 10. * 1.5)
ax.imshow(all[-8:, :], cmap='Greens')
ax.set_yticks(np.arange(-0.5, 8, 1))
ax.set_xticks(np.arange(-0.5, all.shape[1], 1))
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.grid(color='white')
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.spines['left'].set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.text(0, -2.4, "Bit0", {'fontsize': 10, 'fontfamily': 'serif'})
ax.text(59, -2.4, "Bit63", {'fontsize': 10, 'fontfamily': 'serif'})
ax.text(64, -2.4, "I/O Edge0", {'fontsize': 10, 'fontfamily': 'serif'})
ax.text(72.5, -2.4, "Edge8 I/O", {'fontsize': 10, 'fontfamily': 'serif'})
ax.arrow(0.4, -2.3, -.25, 1., width=0.1, color='black')
ax.arrow(62.6, -2.3, .25, 1., width=0.1, color='black')
ax.arrow(64.3, -2.3, -.25, 1., width=0.1, color='black')
ax.arrow(65.3, -2.3, -.25, 1., width=0.1, color='black')
ax.arrow(78, -2.3, 0., 1., width=0.1, color='black')
ax.arrow(79, -2.3, 0., 1., width=0.1, color='black')
# plt.tight_layout()
plt.subplots_adjust(left=-0.0, right=1., top=0.9, bottom=0.0)
plt.savefig(f'/opt/project/img/gs/pattern-edge-weights-part-{policy}.pdf')
plt.close('all')

all = np.expand_dims(np.mean(all, axis=0), axis=0)
ax = plt.subplot()
fig = plt.gcf()
fig.set_figwidth(all.shape[1] / 10.)
fig.set_figheight(all.shape[0] / 10.)
ax.imshow(all, cmap='Greens')
ax.set_yticks(np.arange(-0.5, all.shape[0], 1))
ax.set_xticks(np.arange(-0.5, all.shape[1], 1))
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.grid(color='black')
plt.tight_layout()
plt.savefig(f'/opt/project/img/gs/pattern-edge-weights-avg-{policy}.pdf')
plt.close('all')

from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Reddit
import torch
from torch_geometric.data import NeighborSampler
# from NS import NeighborSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from SAGEConv import SAGEConv
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import time

# from torch.utils.data import DataLoader

"""
GraphSAGE的minibatch方法(包含采样)
可选择的数据集：Cora、Citeseer、Pubmed、Reddit
使用方法：GS方法
"""

# dataset = Planetoid(root='./cora/', name='Cora')
# dataset = Planetoid(root='./citeseer',name='Citeseer')
# dataset = Planetoid(root='./pubmed/',name='Pubmed')
dataset = Reddit(root='./reddit/')
print(dataset)

start_time = time.time()
# cora
# train_loader = NeighborSampler(dataset[0].edge_index, node_idx=dataset[0].train_mask,
#                                sizes=[10, 10], batch_size=16, shuffle=True,
#                                num_workers=12)

# Reddit
train_loader = NeighborSampler(dataset[0].edge_index, node_idx=dataset[0].train_mask,
                               sizes=[25, 10], batch_size=1024, shuffle=True,
                               num_workers=12)

end_time = time.time()
init_sample_time = end_time - start_time
# print('NeighborSampler time:{}'.format(end_time - start_time))


subgraph_loader = NeighborSampler(dataset[0].edge_index, node_idx=None, sizes=[-1],
                                  batch_size=1024, shuffle=False,
                                  num_workers=12)


class SAGENet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SAGENet, self).__init__()

        self.num_layers = 2

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.

        lin_times = 0
        mes_times = 0
        aggr_times = 0
        up_times = 0

        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x, linear_time, message_time, aggregate_time, update_time = self.convs[i]((x, x_target), edge_index)
            lin_times += linear_time
            mes_times += message_time
            aggr_times += aggregate_time
            up_times += update_time
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1), lin_times, mes_times, aggr_times, up_times

    def inference(self, x_all):
        # pbar = tqdm(total=x_all.size(0) * self.num_layers)
        # pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x, linear_time, message_time, aggregate_time, update_time = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                # pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        # pbar.close()

        return x_all


# cora
# model = SAGENet(dataset.num_features, 16, dataset.num_classes)

# Reddit
model = SAGENet(dataset.num_features, 256, dataset.num_classes)
print(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
data = dataset[0].to(device)
print(data)

x = data.x.to(device)
y = data.y.squeeze().to(device)

criterion = nn.NLLLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train(epoch):
    model.train()

    # pbar = tqdm(total=int(data.train_mask.sum()))
    # pbar.set_description(f'Epoch {epoch:02d}')

    total_lin_time = 0
    total_mes_time = 0
    total_aggr_time = 0
    total_up_time = 0

    total_sample_time = 0

    total_loss = total_correct = 0
    start_time = time.time()
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        end_time = time.time()
        total_sample_time += (end_time - start_time)

        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out, lin_time, mes_time, aggr_time, up_time = model(x[n_id], adjs)

        total_lin_time += lin_time
        total_mes_time += mes_time
        total_aggr_time += aggr_time
        total_up_time += up_time

        loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        # pbar.update(batch_size)
        start_time = time.time()

    # pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / int(data.train_mask.sum())

    return loss, approx_acc, total_lin_time, total_mes_time, total_aggr_time, total_up_time, total_sample_time


@torch.no_grad()
def test():
    model.eval()

    out = model.inference(x)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    results = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]

    return results


lin_times = []
mes_times = []
aggr_times = []
up_times = []
sample_times = []
for epoch in range(1, 11):
    loss, acc, lin_time, mes_time, aggr_time, up_time, sample_time = train(epoch)

    lin_times.append(lin_time)
    mes_times.append(mes_time)
    aggr_times.append(aggr_time)
    up_times.append(up_time)
    sample_times.append(sample_time)

    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')

    train_acc, val_acc, test_acc = test()
    print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')

print("Average linear time:", 1000 * np.mean(lin_times), 'ms')
print("Average message time:", 1000 * np.mean(mes_times), 'ms')
print("Average aggregate time:", 1000 * np.mean(aggr_times), 'ms')
print("Average update time:", 1000 * np.mean(up_times), 'ms')
print("Average sample time:", (1000 * np.mean(sample_times) + init_sample_time), 'ms')

from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Reddit
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# from torch_geometric.nn import GCNConv
from GATConv import GATConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import JumpingKnowledge

from MP import MessagePassing
from GCNConv import GCNConv
from ChebConv import ChebConv

# from torch_geometric.nn import ChebConv
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T

"""
可选择的模型：GCN、GAT模型
可选择的数据集：ogbn-arxiv、ogbn-products
使用方法：SpMM方法
"""

dataset = PygNodePropPredDataset('ogbn-arxiv', root='./arxiv/', transform=T.ToSparseTensor())
# dataset = PygNodePropPredDataset('ogbn-products', root='./products/', transform=T.ToSparseTensor())
print(dataset)

split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-arxiv')
# evaluator = Evaluator(name='ogbn-products')
data = dataset[0]

train_idx = split_idx['train']
test_idx = split_idx['test']


# baseline：GCN模型（2层）
class GCNNet(nn.Module):
    def __init__(self, dataset, hidden_dim=16):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, dataset.num_classes)

    def forward(self, data):
        x, adj_t = data.x, data.adj_t

        lin_times = 0
        mes_times = 0
        aggr_times = 0
        up_times = 0

        x, linear_time, message_time, aggregate_time, update_time = self.conv1(x, adj_t)

        lin_times += linear_time
        mes_times += message_time
        aggr_times += aggregate_time
        up_times += update_time

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x, linear_time, message_time, aggregate_time, update_time = self.conv2(x, adj_t)
        lin_times += linear_time
        mes_times += message_time
        aggr_times += aggregate_time
        up_times += update_time
        x = F.log_softmax(x, dim=1)

        return x, lin_times, mes_times, aggr_times, up_times


# baseline：GAT模型（2层）
class GATNet(nn.Module):
    def __init__(self, dataset, hidden_dim=8):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(dataset.num_features, hidden_dim, heads=2, dropout=0.6)
        self.conv2 = GATConv(2 * hidden_dim, dataset.num_classes, dropout=0.6)

    def forward(self, data):
        x, adj_t = data.x, data.adj_t

        lin_times = 0
        mes_times = 0
        aggr_times = 0
        up_times = 0

        x = F.dropout(x, p=0.6, training=self.training)
        x, linear_time, message_time, aggregate_time, update_time = self.conv1(x, adj_t)
        lin_times += linear_time
        mes_times += message_time
        aggr_times += aggregate_time
        up_times += update_time
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)

        x, linear_time, message_time, aggregate_time, update_time = self.conv2(x, adj_t)
        lin_times += linear_time
        mes_times += message_time
        aggr_times += aggregate_time
        up_times += update_time

        return F.log_softmax(x, dim=1), lin_times, mes_times, aggr_times, up_times


# model = ChebNet(dataset, hidden_dim=16)
model = GCNNet(dataset, hidden_dim=256)
# model = GATNet(dataset, hidden_dim=128)
print(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
data = dataset[0].to(device)
data.adj_t = data.adj_t.to_symmetric()  # adj_t
print(data)

train_idx = train_idx.to(device)

criterion = nn.NLLLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


def train():
    model.train()
    lin_times = []
    mes_times = []
    aggr_times = []
    up_times = []
    for epoch in range(100):
        out, lin_time, mes_time, aggr_time, up_time = model(data)
        loss = criterion(out[train_idx], data.y.squeeze(1)[train_idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # _, pred = torch.max(out[train_idx], dim=1)
        # correct = (pred == data.y[train_idx]).sum().item()
        # acc = correct / train_idx.size(0)

        lin_times.append(lin_time)
        mes_times.append(mes_time)
        aggr_times.append(aggr_time)
        up_times.append(up_time)

        # print('Epoch {:03d} train_loss: {:.4f} train_acc: {:.4f}'.format(
        #     epoch, loss.item(), acc))

        print('Epoch {:03d} train_loss: {:.4f}'.format(epoch, loss.item()))

        test()

        # val_loss, val_acc = valid()

        # print('Epoch {:03d} train_loss: {:.4f} train_acc: {:.4f} val_loss: {:.4f} val_acc: {:.4f}'.format(
        #     epoch, loss.item(), acc, val_loss, val_acc))

    print("Average linear time:", 1000 * np.mean(lin_times), 'ms')
    print("Average message+aggregate time:", 1000 * np.mean(mes_times), 'ms')
    # print("Average aggregate time:", 1000 * np.mean(aggr_times), 'ms')
    print("Average update time:", 1000 * np.mean(up_times), 'ms')


@torch.no_grad()
def test():
    model.eval()

    out, lin_time, mes_time, aggr_time, up_time = model(data)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    print(f'Train: {train_acc:.4f}, Val: {valid_acc:.4f}, '
          f'Test: {test_acc:.4f}')

    return train_acc, valid_acc, test_acc


if __name__ == '__main__':
    train()

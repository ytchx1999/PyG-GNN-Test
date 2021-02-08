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

"""
可选择的模型：ChebNet、GCN、GAT模型
可选择的数据集：Cora、Citeseer、Pubmed
使用方法：GS方法
"""

dataset = Planetoid(root='./cora/', name='Cora')
# dataset = Planetoid(root='./citeseer',name='Citeseer')
# dataset = Planetoid(root='./pubmed/', name='Pubmed')
print(dataset)


# baseline：GCN模型（2层）
class ChebNet(nn.Module):
    def __init__(self, dataset, hidden_dim=16):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(dataset.num_node_features, hidden_dim, K=2)
        self.conv2 = ChebConv(hidden_dim, dataset.num_classes, K=2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        lin_times = 0
        mes_times = 0
        aggr_times = 0
        up_times = 0

        x, linear_time, message_time, aggregate_time, update_time = self.conv1(x, edge_index)

        lin_times += linear_time
        mes_times += message_time
        aggr_times += aggregate_time
        up_times += update_time

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x, linear_time, message_time, aggregate_time, update_time = self.conv2(x, edge_index)
        lin_times += linear_time
        mes_times += message_time
        aggr_times += aggregate_time
        up_times += update_time
        x = F.log_softmax(x, dim=1)

        return x, lin_times, mes_times, aggr_times, up_times


# baseline：GCN模型（2层）
class GCNNet(nn.Module):
    def __init__(self, dataset, hidden_dim=16):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        lin_times = 0
        mes_times = 0
        aggr_times = 0
        up_times = 0

        x, linear_time, message_time, aggregate_time, update_time = self.conv1(x, edge_index)

        lin_times += linear_time
        mes_times += message_time
        aggr_times += aggregate_time
        up_times += update_time

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x, linear_time, message_time, aggregate_time, update_time = self.conv2(x, edge_index)
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
        self.conv1 = GATConv(dataset.num_features, hidden_dim, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * hidden_dim, dataset.num_classes, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        lin_times = 0
        mes_times = 0
        aggr_times = 0
        up_times = 0

        x = F.dropout(x, p=0.6, training=self.training)
        x, linear_time, message_time, aggregate_time, update_time = self.conv1(x, edge_index)
        lin_times += linear_time
        mes_times += message_time
        aggr_times += aggregate_time
        up_times += update_time
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)

        x, linear_time, message_time, aggregate_time, update_time = self.conv2(x, edge_index)
        lin_times += linear_time
        mes_times += message_time
        aggr_times += aggregate_time
        up_times += update_time

        return F.log_softmax(x, dim=1), lin_times, mes_times, aggr_times, up_times


# model = ChebNet(dataset, hidden_dim=16)
# model = GCNNet(dataset, hidden_dim=16)
model = GATNet(dataset, hidden_dim=8)
print(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
data = dataset[0].to(device)
print(data)

criterion = nn.NLLLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


# 按照60%、20%、20%划分train、valid、test
# if dataset.name == 'Cora':
#     data.train_mask[:1624] = True
#     data.train_mask[1624:2166] = True
#     data.train_mask[2166:] = True
# elif dataset.name == 'Citeseer':
#     data.train_mask[:1995] = True
#     data.train_mask[1995:2661] = True
#     data.train_mask[2661:] = True
# elif dataset.name == 'Pubmed':
#     data.train_mask[:11829] = True
#     data.train_mask[11829:15773] = True
#     data.train_mask[15773:] = True


def train():
    model.train()
    lin_times = []
    mes_times = []
    aggr_times = []
    up_times = []
    for epoch in range(100):
        out, lin_time, mes_time, aggr_time, up_time = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = torch.max(out[data.train_mask], dim=1)
        correct = (pred == data.y[data.train_mask]).sum().item()
        acc = correct / data.train_mask.sum().item()

        lin_times.append(lin_time)
        mes_times.append(mes_time)
        aggr_times.append(aggr_time)
        up_times.append(up_time)

        print('Epoch {:03d} train_loss: {:.4f} train_acc: {:.4f}'.format(
            epoch, loss.item(), acc))

        # val_loss, val_acc = valid()

        # print('Epoch {:03d} train_loss: {:.4f} train_acc: {:.4f} val_loss: {:.4f} val_acc: {:.4f}'.format(
        #     epoch, loss.item(), acc, val_loss, val_acc))

    print("Average linear time:", 1000 * np.mean(lin_times), 'ms')
    print("Average message time:", 1000 * np.mean(mes_times), 'ms')
    print("Average aggregate time:", 1000 * np.mean(aggr_times), 'ms')
    print("Average update time:", 1000 * np.mean(up_times), 'ms')

    test()


# def valid():
#     # model.eval()
#     with torch.no_grad():
#         out = model(data)
#         loss = criterion(out[data.val_mask], data.y[data.val_mask])
#         _, pred = torch.max(out[data.val_mask], dim=1)
#         correct = (pred == data.y[data.val_mask]).sum().item()
#         acc = correct / data.val_mask.sum().item()
#         return loss.item(), acc
#         # print("val_loss: {:.4f} val_acc: {:.4f}".format(loss.item(), acc))


def test():
    model.eval()
    out, lin_time, mes_time, aggr_time, up_time = model(data)
    loss = criterion(out[data.test_mask], data.y[data.test_mask])
    _, pred = torch.max(out[data.test_mask], dim=1)
    correct = (pred == data.y[data.test_mask]).sum().item()
    acc = correct / data.test_mask.sum().item()
    print("test_loss: {:.4f} test_acc: {:.4f}".format(loss.item(), acc))


if __name__ == '__main__':
    train()

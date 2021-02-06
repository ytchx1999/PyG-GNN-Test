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
import time

"""
JK-Nets模型
可选择的数据集：Cora、Citeseer、Pubmed
"""

dataset = Planetoid(root='./cora/', name='Cora')
# dataset = Planetoid(root='./citeseer',name='Citeseer')
# dataset = Planetoid(root='./pubmed/', name='Pubmed')
print(dataset)


# baseline：GCN模型（2层）
# class GCNNet(nn.Module):
#     def __init__(self, dataset):
#         super(GCNNet, self).__init__()
#         self.conv1 = GCNConv(dataset.num_node_features, 16)
#         self.conv2 = GCNConv(16, dataset.num_classes)
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#
#         mes_times = 0
#         aggr_times = 0
#         up_times = 0
#
#         x, message_time, aggregate_time, update_time = self.conv1(x, edge_index)
#
#         mes_times += message_time
#         aggr_times += aggregate_time
#         up_times += update_time
#
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x, message_time, aggregate_time, update_time = self.conv2(x, edge_index)
#         mes_times += message_time
#         aggr_times += aggregate_time
#         up_times += update_time
#         x = F.log_softmax(x, dim=1)
#
#         return x, mes_times, aggr_times, up_times
#
#
# # baseline：GAT模型（2层）
# class GATNet(nn.Module):
#     def __init__(self, dataset):
#         super(GATNet, self).__init__()
#         self.conv1 = GATConv(dataset.num_features, 8, heads=8, dropout=0.6)
#         self.conv2 = GATConv(8 * 8, dataset.num_classes, dropout=0.6)
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         mes_times = 0
#         aggr_times = 0
#         up_times = 0
#
#         x = F.dropout(x, p=0.6, training=self.training)
#         x, message_time, aggregate_time, update_time = self.conv1(x, edge_index)
#         mes_times += message_time
#         aggr_times += aggregate_time
#         up_times += update_time
#         x = F.elu(x)
#         x = F.dropout(x, p=0.6, training=self.training)
#
#         x, message_time, aggregate_time, update_time = self.conv2(x, edge_index)
#         mes_times += message_time
#         aggr_times += aggregate_time
#         up_times += update_time
#
#         return F.log_softmax(x, dim=1), mes_times, aggr_times, up_times


# JK-Nets（6层）
class JKNet(nn.Module):
    def __init__(self, dataset, mode='max', num_layers=6, hidden=16):
        super(JKNet, self).__init__()
        self.num_layers = num_layers
        self.mode = mode

        self.conv0 = GCNConv(dataset.num_node_features, hidden)
        self.dropout0 = nn.Dropout(p=0.5)

        for i in range(1, self.num_layers):
            setattr(self, 'conv{}'.format(i), GCNConv(hidden, hidden))
            setattr(self, 'dropout{}'.format(i), nn.Dropout(p=0.5))

        self.jk = JumpingKnowledge(mode=mode)
        if mode == 'max':
            self.fc = nn.Linear(hidden, dataset.num_classes)
        elif mode == 'cat':
            self.fc = nn.Linear(num_layers * hidden, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        lin_times = 0
        mes_times = 0
        aggr_times = 0
        up_times = 0
        jk_times = 0

        layer_out = []  # 保存每一层的结果
        for i in range(self.num_layers):
            conv = getattr(self, 'conv{}'.format(i))
            dropout = getattr(self, 'dropout{}'.format(i))
            x, linear_time, message_time, aggregate_time, update_time = conv(x, edge_index)
            lin_times += linear_time
            mes_times += message_time
            aggr_times += aggregate_time
            up_times += update_time

            x = dropout(F.relu(x))
            layer_out.append(x)

        start_time = time.time()
        h = self.jk(layer_out)  # JK层
        end_time = time.time()
        jk_times = end_time - start_time

        h = self.fc(h)
        h = F.log_softmax(h, dim=1)

        return h, lin_times, mes_times, aggr_times, up_times, jk_times


model = JKNet(dataset, mode='max')  # max和cat两种模式可供选择
# model = GCNNet(dataset)
# model = GATNet(dataset)
print(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
data = dataset[0].to(device)
print(data)

criterion = nn.NLLLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


def train():
    model.train()
    lin_times = []
    mes_times = []
    aggr_times = []
    up_times = []
    jk_times = []
    for epoch in range(100):
        out, lin_time, mes_time, aggr_time, up_time, jk_time = model(data)
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
        jk_times.append(jk_time)

        print('Epoch {:03d} train_loss: {:.4f} train_acc: {:.4f}'.format(
            epoch, loss.item(), acc))

        # val_loss, val_acc = valid()

        # print('Epoch {:03d} train_loss: {:.4f} train_acc: {:.4f} val_loss: {:.4f} val_acc: {:.4f}'.format(
        #     epoch, loss.item(), acc, val_loss, val_acc))

    print("Average linear time:", 1000 * np.mean(lin_times), 'ms')
    print("Average message time:", 1000 * np.mean(mes_times), 'ms')
    print("Average aggregate time:", 1000 * np.mean(aggr_times), 'ms')
    print("Average update time:", 1000 * np.mean(up_times), 'ms')
    print("Average jk time:", 1000 * np.mean(jk_times), 'ms')

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
    out, lin_time, mes_time, aggr_time, up_time, jk_time = model(data)
    loss = criterion(out[data.test_mask], data.y[data.test_mask])
    _, pred = torch.max(out[data.test_mask], dim=1)
    correct = (pred == data.y[data.test_mask]).sum().item()
    acc = correct / data.test_mask.sum().item()
    print("test_loss: {:.4f} test_acc: {:.4f}".format(loss.item(), acc))


if __name__ == '__main__':
    train()

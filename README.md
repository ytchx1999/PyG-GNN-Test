# PyG-GNN-Test
使用PyTorch Geometric对GNN典型模型的各阶段执行时间进行复现、测试和分析。

[CSDN博客：GNN典型模型的各阶段执行时间与算子分析](https://blog.csdn.net/weixin_41650348/article/details/113090317)

分析的模型包括（后续模型再进行补充）：

+ ChebNet
+ GCN
+ GAT
+ GraphSAGE（minibatch）
+ JK-Nets

### 文件说明

| 文件/文件夹  | 说明                                 |
| :----------- | ------------------------------------ |
| pyg_test.py  | ChebNet、GCN、GAT模型的实验结果      |
| pyg_test2.py | GraphSAGE模型（minibatch）的实验结果 |
| pyg_test3.py | JK-Nets模型的实验结果                |
| MP.py        | 加入计时机制的MessagePassing类       |
| ChebConv.py  | 修改后的ChebConv层的定义             |
| GCNConv.py   | 修改后的GCNConv层的定义              |
| GATConv.py   | 修改后的GATConv层的定义              |
| SAGEConv.py  | 修改后的SAGEConv层的定义             |
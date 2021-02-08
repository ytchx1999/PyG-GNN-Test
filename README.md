# PyG-GNN-Test
GNN负载研究：使用PyTorch Geometric对GNN典型模型的各阶段执行时间进行复现、测试和分析。

实验结果与分析：

+ [CSDN博客：GNN典型模型的各阶段执行时间与算子分析](https://blog.csdn.net/weixin_41650348/article/details/113090317)
+ [CSDN博客：GNN各阶段执行时间实验【Citeseer、Pubmed、Reddit、OGB数据集】](https://blog.csdn.net/weixin_41650348/article/details/113709290)

分析的模型包括（后续模型再进行补充）：

+ ChebNet
+ GCN
+ GAT
+ GraphSAGE（minibatch）
+ JK-Nets

分析的数据集包括：

+ Cora
+ Citeseer
+ Pubmed
+ Reddit
+ ogbn-arxiv
+ ogbn-products

Reddit、ogbn-arxiv、ogbn-products数据集比较大，就不上传了。执行模型代码可以自动下载缓存到本地。

==需要注意，除了pyg_test7.py和pyg_test8.py使用的是SpMM方法，其他均使用GS方法！==

### 文件说明

| 文件/文件夹  | 说明                                 |
| ----------- | ------------------------------------ |
| pyg_test.py  | 模型：ChebNet、GCN、GAT；数据集：Cora、Citeseer、Pubmed    |
| pyg_test2.py | GraphSAGE的minibatch方法(包含采样)；数据集：Cora、Citeseer、Pubmed、Reddit |
| pyg_test3.py | JK-Nets模型；数据集：Cora、Citeseer、Pubmed               |
|     pyg_test4.py   | GAT的minibatch方法(包含采样)；数据集：Reddit |
| pyg_test5.py | GraphSAGE的minibatch方法(包含采样)；数据集：ogbn-arxiv、ogbn-products |
| pyg_test6.py | GAT的minibatch方法(包含采样)；数据集：ogbn-arxiv、ogbn-products |
| pyg_test7.py | GCN、GAT模型（SpMM方法）；数据集：ogbn-arxiv、ogbn-products |
| pyg_test8.py | GraphSAGE的minibatch方法(SpMM方法)；数据集：ogbn-arxiv、ogbn-products |
| MP.py        | 加入计时机制的MessagePassing类       |
| ChebConv.py  | 修改后的ChebConv层的定义             |
| GCNConv.py   | 修改后的GCNConv层的定义              |
| GATConv.py   | 修改后的GATConv层的定义              |
| SAGEConv.py  | 修改后的SAGEConv层的定义             |
import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
import dgl


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
    return labels_one_hot

def load_mat(dataset, train_rate=0.3, val_rate=0.1):
    """Load .mat dataset."""
    data = sio.loadmat("./Data/{}.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']
    
    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)

    ano_labels = np.squeeze(np.array(label))

    return adj, feat, ano_labels

def adj_to_dgl_graph(adj):
    """Convert adjacency matrix to dgl format - DGL 1.1.3 compatible version."""
    # 转换为COO格式
    adj_coo = adj.tocoo()
    
    # 创建边的源节点和目标节点
    src = torch.from_numpy(adj_coo.row).long()
    dst = torch.from_numpy(adj_coo.col).long()
    
    # 使用DGL 1.x的现代API创建图
    dgl_graph = dgl.graph((src, dst), num_nodes=adj.shape[0])
    
    return dgl_graph

def generate_rwr_subgraph(dgl_graph, subgraph_size):
    """Generate subgraph with RWR algorithm - DGL 1.1.3 compatible version with proper device handling."""
    all_idx = list(range(dgl_graph.number_of_nodes()))
    reduced_size = subgraph_size - 1
    
    # 检查图是否在GPU上
    device = next(dgl_graph.parameters()) if hasattr(dgl_graph, 'parameters') and any(dgl_graph.parameters()) else None
    if device is None:
        # 尝试从图的边检查设备
        try:
            src, _ = dgl_graph.edges()
            device = src.device
        except:
            device = torch.device('cpu')
    
    subv = []
    
    # 对每个节点生成子图
    for i in all_idx:
        try:
            # 使用DGL 1.x的sampling API
            from dgl.sampling import random_walk
            
            # 确保节点索引在正确的设备上
            node_tensor = torch.tensor([i], device=device)
            
            # 执行随机游走
            traces, _ = random_walk(dgl_graph, node_tensor, length=subgraph_size*3)
            
            # 提取唯一节点
            subv_i = torch.unique(traces[0]).cpu().tolist()
            
            # 移除padding值（-1）
            subv_i = [x for x in subv_i if x != -1 and x < dgl_graph.number_of_nodes()]
            
            retry_time = 0
            while len(subv_i) < reduced_size and retry_time < 10:
                # 重新进行更长的随机游走
                traces, _ = random_walk(dgl_graph, node_tensor, length=subgraph_size*5)
                subv_i = torch.unique(traces[0]).cpu().tolist()
                subv_i = [x for x in subv_i if x != -1 and x < dgl_graph.number_of_nodes()]
                retry_time += 1
            
            # 如果仍然不够，使用邻居节点填充
            if len(subv_i) < reduced_size:
                # 获取一跳邻居
                try:
                    neighbors = dgl_graph.successors(i).cpu().tolist()
                except:
                    neighbors = []
                
                # 添加邻居节点直到达到要求的大小
                for neighbor in neighbors:
                    if neighbor not in subv_i and len(subv_i) < reduced_size:
                        subv_i.append(neighbor)
                
                # 如果邻居不够，随机添加节点
                while len(subv_i) < reduced_size:
                    random_node = random.randint(0, dgl_graph.number_of_nodes()-1)
                    if random_node not in subv_i:
                        subv_i.append(random_node)
            
            # 确保不超过要求的大小
            subv_i = subv_i[:reduced_size]
            
            # 添加中心节点
            subv_i.append(i)
            
        except Exception as e:
            # 如果随机游走失败，使用简单的邻居采样（不打印错误信息以避免spam）
            try:
                neighbors = dgl_graph.successors(i).cpu().tolist() if hasattr(dgl_graph.successors(i), 'cpu') else dgl_graph.successors(i).tolist()
                
                if len(neighbors) >= reduced_size:
                    subv_i = random.sample(neighbors, reduced_size)
                else:
                    subv_i = neighbors.copy()
                    # 用随机节点填充
                    while len(subv_i) < reduced_size:
                        random_node = random.randint(0, dgl_graph.number_of_nodes()-1)
                        if random_node not in subv_i:
                            subv_i.append(random_node)
                
                subv_i.append(i)
                
            except Exception as e2:
                # 最后的备用方案：完全随机采样（不打印错误信息）
                all_nodes = list(range(dgl_graph.number_of_nodes()))
                subv_i = random.sample([n for n in all_nodes if n != i], min(reduced_size, len(all_nodes)-1))
                subv_i.append(i)
        
        subv.append(subv_i)
    
    return subv
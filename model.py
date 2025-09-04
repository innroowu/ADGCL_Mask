import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        
        return self.act(out)

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)

class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq,1).values

class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values

class WSReadout(nn.Module):
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0,2,1)
        sim = torch.matmul(seq,query)
        sim = F.softmax(sim,dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq,sim)
        out = torch.sum(out,1)
        return out
    

class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl):
        scs = []
        # positive
        scs.append(self.f_k(h_pl, c))

        # negative
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1,:], c_mi[:-1,:]),0)
            scs.append(self.f_k(h_pl, c_mi))

        logits = torch.cat(tuple(scs))

        return logits


class ReconstructionTask(nn.Module):
    """
    reconstruction: mask + reconstruct
    """
    def __init__(self, feature_dim, hidden_dim, mask_ratio=0.3):
        super(ReconstructionTask, self).__init__()
        
        # 可可學習的mask token
        self.mask_token = nn.Parameter(torch.randn(feature_dim))
        
        # reconstruction_decoder
        self.reconstruction_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        self.mask_ratio = mask_ratio
        self.feature_dim = feature_dim
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化權重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # mask token初始化為小的隨機值
        nn.init.normal_(self.mask_token, std=0.02)
    
    def apply_mask(self, features):
        """
        對輸入特徵套用掩碼

        Args:
        features: [batch_size, seq_len, feature_dim] 輸入特徵

        Returns:
        masked_features: 掩碼後的特徵
        original_features: 原始特徵（用於計算重構損失）
        mask_indices: 被遮罩的位置
        """
        batch_size, seq_len, feature_dim = features.shape
        
        # 為每個樣本隨機選擇mask的位置
        mask_indices_list = []
        masked_features = features.clone()
        
        for b in range(batch_size):
            # 隨機選擇要mask的位置（不包括最後一個target節點）
            num_to_mask = max(1, int((seq_len - 1) * self.mask_ratio))
            mask_positions = torch.randperm(seq_len - 1)[:num_to_mask]  # 不掩码target节点
            
            # 應用mask
            masked_features[b, mask_positions, :] = self.mask_token.expand_as(
                masked_features[b, mask_positions, :]
            )
            
            mask_indices_list.append(mask_positions)
        
        return masked_features, features.clone(), mask_indices_list
    
    def compute_reconstruction_loss(self, node_embeddings, original_features, mask_indices_list):
        """
        Args:
        node_embeddings: [batch_size, seq_len, hidden_​​dim] GCN輸出的節點embedding
        original_features: [batch_size, seq_len, feature_dim] 原始特徵
        mask_indices_list: 每個樣本被遮罩的位置列表

        Returns:
        reconstruction_loss
        """
        batch_size = node_embeddings.shape[0]
        total_loss = 0.0
        num_masked = 0
        
        for b in range(batch_size):
            if len(mask_indices_list[b]) == 0:
                continue
                
            mask_positions = mask_indices_list[b]
            
            # 對被遮罩的節點進行特徵重構
            masked_embeddings = node_embeddings[b, mask_positions, :]  # [num_masked, hidden_dim]
            original_masked_features = original_features[b, mask_positions, :]  # [num_masked, feature_dim]
            
            # 透過解碼器重構特徵
            reconstructed_features = self.reconstruction_decoder(masked_embeddings)
            
            # MSE loss
            loss = F.mse_loss(reconstructed_features, original_masked_features)
            total_loss += loss
            num_masked += len(mask_positions)
        
        # 平均重構損失
        if num_masked > 0:
            final_loss = total_loss / batch_size
            #return total_loss / batch_size
            return final_loss * 10.0
        else:
            return torch.tensor(0.0, device=node_embeddings.device)


class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout, discriminator_type, temperature=0.1, learnable_temp=False, enable_reconstruction=True):
        super(Model, self).__init__()
        self.read_mode = readout
        self.gcn = GCN(n_in, n_h, activation)

        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()
            
            
        self.disc = Discriminator(n_h, negsamp_round)

        # 新增：重構任务
        self.enable_reconstruction = enable_reconstruction
        if enable_reconstruction:
            self.reconstruction_task = ReconstructionTask(n_in, n_h)
            print(f"✅ 重構任務已啟用")
        else:
            print(f"✅ 重構任務未啟用")

    def forward(self, seq1, adj, sparse=False, return_reconstruction_loss=False):
        reconstruction_loss = 0.0
        
        # 1. Masked Graph Autoencoders（如果啟用重構）
        if self.enable_reconstruction and return_reconstruction_loss:
            masked_seq1, original_features, mask_indices = self.reconstruction_task.apply_mask(seq1)
            input_features = masked_seq1
        else:
            input_features = seq1

        #h_1 = self.gcn(seq1, adj, sparse)
        h_1 = self.gcn(input_features, adj, sparse)  # 注意：这里应该用input_features

        if self.read_mode != 'weighted_sum':
            c = self.read(h_1[:,: -1,:])
            h_mv = h_1[:,-1,:]
        else:
            h_mv = h_1[:, -1, :]
            c = self.read(h_1[:,: -1,:], h_1[:,-2: -1, :])

        ret = self.disc(c, h_mv)
        
        #計算重構損失
        if self.enable_reconstruction and return_reconstruction_loss:
            reconstruction_loss = self.reconstruction_task.compute_reconstruction_loss(
                h_1, original_features, mask_indices
            )

        if return_reconstruction_loss:
            return ret, h_mv, c, reconstruction_loss
        else:
            return ret, h_mv, c
        #return ret, h_mv, c


class Gene(nn.Module):
    def __init__(self, nb_nodes, hid_dim, out_dim):
        super(Gene, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)
        self.layer1 = GraphConv(hid_dim, hid_dim)
        self.layer2 = GraphConv(hid_dim, out_dim)
        self.sigmoid = nn.Sigmoid()

        self.batchnorm = nn.BatchNorm1d(out_dim)

        self.epsilon = torch.nn.Parameter(torch.Tensor(nb_nodes))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.epsilon, 0.5)
        
    def forward(self, g, x, adj):
        h1 = self.fc(x)
        h2 = F.relu(self.layer1(g, x))
        h2 = self.layer2(g, h2)

        h = (1 - self.epsilon.view(-1,1)) * h1 + self.epsilon.view(-1,1) * h2

        ret = (torch.mm(h, h.t()) + torch.mm(x, x.t())) / 2

        h = self.batchnorm(h)

        return ret, h

class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConv, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, g, features):
        g = g.local_var()
        g.ndata['h'] = features
        g.update_all(message_func=dgl.function.copy_src(src='h', out='m'), reduce_func=dgl.function.sum(msg='m', out='h'))
        h = g.ndata['h']
        return self.linear(h)

class Disc(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(Disc, self).__init__()
        self.f_k = nn.Bilinear(hid_dim, hid_dim, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        logits = self.f_k(x1, x2)
        logits = self.sigmoid(logits)

        return logits
"""
utils.py
数据加载与工具函数 (支持 ESM-2 3B 2560维特征自动加载)
"""
import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle
import pandas as pd


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


# 蛋白质序列字典
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000


def seq_cat(prot):
    """将蛋白质序列转换为数字列表"""
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        if ch in seq_dict:
            x[i] = seq_dict[ch]
    return x


class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis',
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None, feature_stats=None,
                 use_precomputed_esm2=False):

        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.feature_stats = feature_stats
        self.use_precomputed_esm2 = use_precomputed_esm2

        # 加载预计算的ESM2特征 (智能检测)
        self.esm2_features = None
        if use_precomputed_esm2:
            base_dataset = dataset.replace('_train', '').replace('_test', '')
            
            # 优先级: 5120 (15B) > 2560 (3B) > 1280 (650M)
            file_5120 = os.path.join(root, f'esm2_features_{base_dataset}_5120.pkl')
            file_2560 = os.path.join(root, f'esm2_features_{base_dataset}_2560.pkl')
            file_1280 = os.path.join(root, f'esm2_features_{base_dataset}_1280.pkl')
            
            target_file = None
            if os.path.exists(file_5120):
                target_file = file_5120
                print(f'✓ 检测到旗舰版 ESM-2 15B 特征 (5120维): {target_file}')
            elif os.path.exists(file_2560):
                target_file = file_2560
                print(f'✓ 检测到高性能 ESM-2 3B 特征 (2560维): {target_file}')
            elif os.path.exists(file_1280):
                target_file = file_1280
                print(f'✓ 检测到 ESM-2 特征 (1280维): {target_file}')
            
            if target_file:
                with open(target_file, 'rb') as f:
                    self.esm2_features = pickle.load(f)
                print(f'  已加载 {len(self.esm2_features)} 个蛋白质特征')
            else:
                print(f'⚠ 警告: 未找到 esm2_features_{base_dataset}_[5120|2560|1280].pkl')
                print(f'  请先运行 precompute_ESM2_features.py')

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, xt, y, smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)

        if self.feature_stats is None:
            stats = {'bde': {'mean': 0.0, 'std': 1.0}, 'inv_bl': {'mean': 0.0, 'std': 1.0}}
        else:
            stats = self.feature_stats
        epsilon = 1e-8

        base_dataset = self.dataset.replace('_train', '').replace('_test', '')
        split = 'train' if '_train' in self.dataset else 'test'
        csv_file = f'data/{base_dataset}_{split}.csv'

        protein_sequences = []
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            protein_sequences = df['target_sequence'].tolist()
        
        # 获取当前加载特征的维度，用于处理未找到的序列
        current_esm_dim = 1280 # 默认
        if self.esm2_features and len(self.esm2_features) > 0:
            current_esm_dim = list(self.esm2_features.values())[0].shape[0]

        for i in range(data_len):
            if (i + 1) % 500 == 0:
                print('Converting SMILES to graph: {}/{}'.format(i + 1, data_len))

            smiles = xd[i]
            target = xt[i]
            labels = y[i]

            num_atoms, edge_index_list, bond_energies, bond_lengths = smile_graph[smiles]

            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)

            ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            ecfp_tensor = torch.FloatTensor(np.array(ecfp, dtype=float)).unsqueeze(0)

            atom_feats_list = []
            for atom in mol.GetAtoms():
                atom_feats_list.append(atom_features(atom))
            features = np.asarray(atom_feats_list, dtype=float)

            edge_index = torch.LongTensor(edge_index_list).transpose(1, 0)

            np_bde = np.array(bond_energies)
            np_bl = np.array(bond_lengths)
            feat_bde = (np_bde - stats['bde']['mean']) / stats['bde']['std']
            np_inv_bl = 1.0 / (np_bl + epsilon)
            feat_inv_bl = (np_inv_bl - stats['inv_bl']['mean']) / stats['inv_bl']['std']
            edge_attr = torch.FloatTensor(np.stack([feat_bde, feat_inv_bl], axis=1))

            GNData = DATA.Data(
                x=torch.FloatTensor(features),
                edge_attr=edge_attr,
                edge_index=edge_index,
                y=torch.FloatTensor([labels])
            )

            GNData.ecfp = ecfp_tensor
            GNData.target = torch.LongTensor([target])

            if i < len(protein_sequences):
                GNData.target_seq = protein_sequences[i]
                if self.esm2_features is not None:
                    protein_seq = protein_sequences[i]
                    if protein_seq in self.esm2_features:
                        feat = self.esm2_features[protein_seq]
                        # 确保是 2D Tensor [1, dim]
                        GNData.protein_features = torch.FloatTensor(feat).unsqueeze(0)
                    else:
                        # 使用动态获取的维度创建零向量
                        GNData.protein_features = torch.zeros(current_esm_dim).unsqueeze(0)

            GNData.__setitem__('c_size', torch.LongTensor([num_atoms]))
            data_list.append(GNData)

        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def rmse(y, f): return sqrt(((y - f) ** 2).mean(axis=0))
def mse(y, f): return ((y - f) ** 2).mean(axis=0)
def pearson(y, f): return np.corrcoef(y, f)[0, 1]
def spearman(y, f): return stats.spearmanr(y, f)[0]
def ci(y, f):
    ind = np.argsort(y)
    y = y[ind]; f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0; S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0: S = S + 1
                elif u == 0: S = S + 0.5
            j = j - 1
        i = i - 1; j = i - 1
    ci = S / z
    return ci

def r2(y, f):
    """R-squared (Coefficient of Determination)"""
    sst = np.sum((y - np.mean(y)) ** 2)
    if sst == 0:
        return 0.0
    ssr = np.sum((y - f) ** 2)
    return 1 - (ssr / sst)

def rm2(y, f):
    """
    rm2 index (Modified Squared Correlation Coefficient)
    衡量模型外部预测能力的严格指标，要求回归线不仅相关性高，且接近 Y=X。
    Paper: Roy et al. (2008/2013)
    """
    y = np.array(y)
    f = np.array(f)

    # 1. 计算标准 r2 (Squared Pearson Correlation)
    # 注意：rm2定义中的r2通常指Pearson r的平方
    r = np.corrcoef(y, f)[0, 1]
    r_squared = r ** 2

    # 2. 计算 r02 (强制通过原点的回归系数)
    # 拟合 y = k * f (无截距)
    # 斜率 k = sum(y*f) / sum(f^2)
    num = np.sum(y * f)
    den = np.sum(f ** 2)
    
    if den == 0:
        k = 0
    else:
        k = num / den
    
    # 计算通过原点的预测值 f_origin
    f_origin = k * f
    
    # 计算 r02
    ssr_origin = np.sum((y - f_origin) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    
    if sst == 0:
        r02 = 0.0
    else:
        r02 = 1 - (ssr_origin / sst)

    val = r_squared * (1 - np.sqrt(np.abs(r_squared - r02)))
    
    return val
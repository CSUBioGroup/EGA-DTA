import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import AllChem
import networkx as nx
# from utils import *
from alfabet.model import predict
from functools import lru_cache

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))



DEFAULT_BOND_ENERGIES = {
    'C=C': 602, 'C=N': 615, 'C=O': 745, 'C=S': 477,
    'N=N': 418, 'N=O': 607, 'O=O': 498, 'S=S': 425, 'S=O': 522,
    'C≡C': 835, 'C≡N': 891, 'N≡N': 946, # triple
    'C:C': 518, 'C:N': 615, 'N:N': 485 # aromatic
}

def validate_smile(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smile}")
    return smile
@lru_cache(maxsize=None)
def cached_predict(smile):
    validate_smile(smile)  # valid SMILES
    return predict([smile], drop_duplicates=False, verbose=True)


def _get_bond_energy(bond, bde_prediction_map, multi_bond_energies):
    begin_atom = bond.GetBeginAtom()
    end_atom = bond.GetEndAtom()
    begin_symbol = begin_atom.GetSymbol()
    end_symbol = end_atom.GetSymbol()
    bond_order = bond.GetBondTypeAsDouble()
    is_aromatic = bond.GetIsAromatic()

    if is_aromatic:
        bond_key = f"{min(begin_symbol, end_symbol)}:{max(begin_symbol, end_symbol)}"
        return multi_bond_energies.get(bond_key, 400)
    elif bond_order == 1.0:
        bond_idx = bond.GetIdx()
        if bond_idx in bde_prediction_map:
            return round(4.184 * bde_prediction_map[bond_idx], 2)
        # print(f"Warning: No BDE prediction for bond index {bond_idx}, using default value.")
        return 400 if begin_symbol == 'H' or end_symbol == 'H' else 350
    elif bond_order == 2.0:
        bond_key = f"{min(begin_symbol, end_symbol)}={max(begin_symbol, end_symbol)}"
        return multi_bond_energies.get(bond_key, 600)
    elif bond_order == 3.0:
        bond_key = f"{min(begin_symbol, end_symbol)}≡{max(begin_symbol, end_symbol)}"
        return multi_bond_energies.get(bond_key, 800)
    return 400


def _get_bond_length(conf, bond):
    if conf is None:
        bond_order = bond.GetBondTypeAsDouble()
        default_lengths = {1.0: 1.5, 2.0: 1.3, 3.0: 1.2, 1.5: 1.4}
        return round(default_lengths.get(bond_order, 1.5), 3)
    pos_i = conf.GetAtomPosition(bond.GetBeginAtomIdx())
    pos_j = conf.GetAtomPosition(bond.GetEndAtomIdx())
    return round(pos_i.Distance(pos_j), 3)


def smile_to_graph(smile):
    """Convert a SMILES string to a molecular graph with bond dissociation energies and bond lengths."""

    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smile}")
    mol = Chem.AddHs(mol)

    # 预测键能
    try:
        alfabet_predictions = cached_predict(smile)
        if not all(col in alfabet_predictions.columns for col in ['molecule', 'bond_index']):
            # print(f"Warning: Missing required columns in alfabet predictions for SMILES: {smile}")
            bde_prediction_map = {}
        else:
            bde_prediction_map = {int(row['bond_index']): row['bde_pred'] for _, row in alfabet_predictions.iterrows()}
    except Exception as e:
        print(f"Error in alfabet prediction for SMILES {smile}: {e}")
        bde_prediction_map = {}

    # 生成 3D 构象
    conf = None
    try:
        if AllChem.EmbedMolecule(mol) != -1:
            AllChem.UFFOptimizeMolecule(mol)
            conf = mol.GetConformer()
    except Exception:
        conf = None

    num_atoms = mol.GetNumAtoms()
    edges, edges_BDE, edges_BL = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append([i, j])
        edges_BDE.append(_get_bond_energy(bond, bde_prediction_map, DEFAULT_BOND_ENERGIES))
        edges_BL.append(_get_bond_length(conf, bond))

    # 生成双向边
    edge_index, bond_energies, bond_lengths = [], [], []
    for (i, j), energy, length in zip(edges, edges_BDE, edges_BL):
        edge_index.extend([[i, j], [j, i]])
        bond_energies.extend([energy, energy])
        bond_lengths.extend([length, length])

    # ========== 新增处理：处理孤立原子/无边的情况 ==========
    if len(edge_index) == 0:
        # 如果没有边（例如 Li+ 离子），添加一个自环 [i, i]
        for i in range(num_atoms):
            edge_index.append([i, i])
            bond_energies.append(0.0)
            # 键长设为 1.5 (典型碳碳键长)，防止标准化时数值爆炸
            bond_lengths.append(1.5)

    return num_atoms, edge_index, bond_energies, bond_lengths

compound_iso_smiles = []
# [修改] 更新了数据集列表
target_datasets = ['Metz', 'kiba', 'davis', 'bindingdb']

print(f"正在读取数据集 SMILES: {target_datasets}")
for dt_name in target_datasets:
    opts = ['train', 'test']
    for opt in opts:
        file_path = 'data/' + dt_name + '_' + opt + '.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if 'compound_iso_smiles' in df.columns:
                compound_iso_smiles += list(df['compound_iso_smiles'])
            else:
                print(f"Warning: 'compound_iso_smiles' column not found in {file_path}")
        else:
            print(f"Warning: File not found {file_path}")

compound_iso_smiles = set(compound_iso_smiles)
compound_iso_smiles = sorted(list(compound_iso_smiles))

smile_graph = {}
print("Starting to generate smile_graph...") 
for i, smile in enumerate(compound_iso_smiles):
    if (i+1) % 100 == 0:
        print(f"Processing SMILES {i+1}/{len(compound_iso_smiles)}")
    try:
        g = smile_to_graph(smile)   #转换为图
        smile_graph[smile] = g
    except Exception as e:
        print(f"Failed to process SMILES: {smile}, Error: {e}")

print("Saving smile_graph.pkl...") 
with open("smile_graph.pkl", "wb") as f:
    pickle.dump(smile_graph, f)

print("smile_graph.pkl has been successfully generated.")
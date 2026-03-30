"""
: python convert_pytorch_data.py --use-precomputed-esm2

1. Label 'affinity'
2.  ['Metz', 'kiba', 'davis']
"""
import pandas as pd
import numpy as np
import os
import pickle
import argparse

from utils import TestbedDataset, seq_cat


def check_precomputed_features(dataset_name):
    files_to_check = [
        f'data/esm2_features_{dataset_name}_5120.pkl', # 15B
        f'data/esm2_features_{dataset_name}_2560.pkl', # 3B
        f'data/esm2_features_{dataset_name}_1280.pkl'  # 650M
    ]
    
    for f in files_to_check:
        if os.path.exists(f):
            print(f"  ✓ Feature file detected: {f}")
            return True
            
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-precomputed-esm2', action='store_true',
                        help='Use pre-computed ESM2 features')
    args = parser.parse_args()

    print("=" * 60)
    print(" (Multi-Dataset Support)")
    print("=" * 60)


    print("\n smile_graph.pkl...")
    try:
        with open("smile_graph.pkl", "rb") as f:
            smile_graph = pickle.load(f)
        print(f"✓ smile_graph.pkl  ({len(smile_graph)} )")
    except FileNotFoundError:
        print("✗ Not found 'smile_graph.pkl'!")
        print("  Please run your `drug_graph_construct.py` ")
        exit(1)

    print("\n Calculating global values BDE/BL ...")
    all_bdes = []
    all_bls = []
    for smile in smile_graph:
        num_edges_half = len(smile_graph[smile][2]) // 2
        all_bdes.extend(smile_graph[smile][2][:num_edges_half])
        all_bls.extend(smile_graph[smile][3][:num_edges_half])

    all_bdes = np.array(all_bdes)
    bde_mean = np.mean(all_bdes)
    bde_std = np.std(all_bdes)

    epsilon = 1e-8
    all_inv_bls = 1.0 / (np.array(all_bls) + epsilon)
    inv_bl_mean = np.mean(all_inv_bls)
    inv_bl_std = np.std(all_inv_bls)

    feature_stats = {
        'bde': {'mean': bde_mean, 'std': bde_std},
        'inv_bl': {'mean': inv_bl_mean, 'std': inv_bl_std}
    }
    print(f"  BDE (mean, std): ({bde_mean:.2f}, {bde_std:.2f})")
    print(f"  Inv-BL (mean, std): ({inv_bl_mean:.2f}, {inv_bl_std:.2f})")


    datasets = ['Metz', 'kiba', 'davis']
    
    for dataset in datasets:
        print(f"\n{'=' * 60}")
        print(f"处理数据集: {dataset}")
        print(f"{'=' * 60}")

        processed_data_file_train = f'data/processed/{dataset}_train.pt'
        processed_data_file_test = f'data/processed/{dataset}_test.pt'

        os.makedirs('data/processed', exist_ok=True)


        if args.use_precomputed_esm2:
            if not check_precomputed_features(dataset):
                print(f"Warning: No ESM2 feature files found (5120/2560/1280).pkl")
                print(f"  Please run the pre-computation script for {dataset} first (precompute_ESM2_features.py)。")
                print(f"  Skip  {dataset} ...")
                continue


        train_csv = f'data/{dataset}_train.csv'
        test_csv = f'data/{dataset}_test.csv'
        if not os.path.exists(train_csv) or not os.path.exists(test_csv):
            print(f"✗ Not found {train_csv} or {test_csv}")
            print(f"  Skip {dataset} ...")
            continue

        if ((not os.path.isfile(processed_data_file_train)) or
                (not os.path.isfile(processed_data_file_test))):

            print(f"\n Loading CSV...")

            df_train = pd.read_csv(train_csv)
            train_drugs = list(df_train['compound_iso_smiles'])
            train_prots_raw = list(df_train['target_sequence'])

            try:
                train_Y = list(df_train['affinity'])
            except KeyError:
                print(f"✗ Error: The ‘affinity’ column was not found in {train_csv}!")
                print(f"  Current columns: {list(df_train.columns)}")
                continue

            XT_train = [seq_cat(t) for t in train_prots_raw]
            train_drugs = np.asarray(train_drugs)
            train_prots = np.asarray(XT_train)
            train_Y = np.asarray(train_Y)

            # 读取测试集
            df_test = pd.read_csv(test_csv)
            test_drugs = list(df_test['compound_iso_smiles'])
            test_prots_raw = list(df_test['target_sequence'])
            
            try:
                test_Y = list(df_test['affinity']) 
            except KeyError:
                print(f"✗ Error: The ‘affinity’ column was not found in {train_csv}!")
                continue

            XT_test = [seq_cat(t) for t in test_prots_raw]
            test_drugs = np.asarray(test_drugs)
            test_prots = np.asarray(XT_test)
            test_Y = np.asarray(test_Y)

            print(f'  Train: {len(train_drugs)} ')
            print(f'  Test: {len(test_drugs)} ')


            print(f'\n {dataset}_train.pt ...')
            train_data = TestbedDataset(
                root='data',
                dataset=dataset + '_train',
                xd=train_drugs,
                xt=train_prots,
                y=train_Y,
                smile_graph=smile_graph,
                feature_stats=feature_stats,
                use_precomputed_esm2=args.use_precomputed_esm2
            )

            print(f' {dataset}_test.pt ...')
            test_data = TestbedDataset(
                root='data',
                dataset=dataset + '_test',
                xd=test_drugs,
                xt=test_prots,
                y=test_Y,
                smile_graph=smile_graph,
                feature_stats=feature_stats,
                use_precomputed_esm2=args.use_precomputed_esm2
            )

            print(f'✓ {processed_data_file_train} and {processed_data_file_test} created')
        else:
            print(f'  The file {processed_data_file_train} already exists; generation will be skipped.')
            print(f'  (If you need to regenerate it, please delete the files in the data/processed/ directory first.)')

    print("\n" + "=" * 60)
    print("All tasks completed!")
    print("=" * 60)

    if args.use_precomputed_esm2:
        print("\n✓ The data has been configured to use pre-computed ESM2 features")
    else:
        print("\n✓ The data has been configured for standard mode")

if __name__ == "__main__":
    main()

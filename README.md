# EGA-DTA
EGA-DTA: An Energetic-Geometric Augmented Graph Neural Network with Target-Conditional Gating for DTA Prediction
* `BDEGNN.py`: Core model definition including `EnhancedGATNet`, `TCGI`, and `Multi-Strategy Fusion`.
* `DTA_train_test.py`: Main training script with Cosine Annealing scheduler and MSE loss.
* `precompute_ESM2_3B.py`: Script to extract 2560-dim protein features using ESM-2 3B.
* `drug_graph_construct.py`: Generates molecular graphs with BDE/BL attributes (requires `alfabet`).
* `convert_pytorch_data.py`: Prepares PyTorch Geometric datasets.
* `utils.py`: Metrics (CI, MSE, $r_m^2$) and data loading utilities.

# EGA-DTA
EGA-DTA: An Energetic-Geometric Augmented Graph Neural Network with Target-Conditional Gating for DTA Prediction
* `BDEGNN.py`: Core model definition including `EnhancedGATNet`, `TCGI`, and `Multi-Strategy Fusion`.
* `DTA_train_test.py`: Main training script with Cosine Annealing scheduler and MSE loss.
* `precompute_ESM2_3B.py`: Script to extract 2560-dim protein features using ESM-2 3B.
* `drug_graph_construct.py`: Generates molecular graphs with BDE/BL attributes (requires `alfabet`).
* `convert_pytorch_data.py`: Prepares PyTorch Geometric datasets.
* `utils.py`: Metrics (CI, MSE, $r_m^2$) and data loading utilities.


Step 1: Drug Graph Construction
Note: The BDE calculation relies on the TensorFlow-based alfabet library. To save time, we provide the pre-computed graph dictionary:
smile_graph.pkl: Contains graph structures, BDE, and Bond Lengths for the datasets.
Place this file in the root directory.
python drug_graph_construct.py


Step 2: Protein Feature Extraction (ESM-2 3B)
Extract the 2560-dimensional embeddings for protein sequences. This requires a GPU with approx. 16GB+ VRAM (or reduce batch size).
python precompute_ESM2_3B.py davis 4


Step 3:Convert to PyTorch Format
Combine the drug graphs and protein features into .pt files for training.
python convert_pytorch_data.py --use-precomputed-esm2

Since the ALFABET model is based on TensorFlow, we have placed all preprocessed data and trained model weights at the following link: https://drive.google.com/drive/folders/1eQPO3dr4NfqUja_oW_bSAq6DUVAIyGWl?usp=drive_link

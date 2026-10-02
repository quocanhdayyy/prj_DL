import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import os

# Cấu hình
COL_DRUG1 = 'Smiles1'
COL_DRUG2 = 'Smiles2'
COL_LABEL = 'Label'

def get_fingerprints(smiles_list, n_bits=2048):
    rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=5)
    np_fps = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                fp_array = np.zeros(n_bits, dtype=np.int32)
            else:
                fp = rdkit_gen.GetFingerprint(mol)
                fp_array = np.zeros(n_bits, dtype=np.int32)
                fp_array[list(fp.GetOnBits())] = 1
        except:
             fp_array = np.zeros(n_bits, dtype=np.int32)
        np_fps.append(fp_array)
    return np.array(np_fps)

def process_and_save(csv_path, output_name):
    print(f"Đang xử lý {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Tạo Features
    fp1 = get_fingerprints(df[COL_DRUG1].values)
    fp2 = get_fingerprints(df[COL_DRUG2].values)
    X = np.concatenate((fp1, fp2), axis=1)
    # 2. Xử lý Label (Chuyển về 0-85 luôn tại đây)
    y = df[COL_LABEL].values
    if np.min(y) == 1:
        y = y - 1
        
    # 3. Lưu nén dạng .npz
    np.savez_compressed(output_name, X=X, y=y)
    print(f" -> Đã lưu: {output_name}.npz (Shape X: {X.shape})")

# Chạy cho 3 file
process_and_save('train_final.csv', 'train_data')
process_and_save('valid.csv', 'valid_data')
process_and_save('test.csv',  'test_data')
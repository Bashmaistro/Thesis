import os
from utils.preprocessing import Preprocess_3D
import pandas as pd
from config import Config
import torch
import glob2
from dataset.dataset import DataGenerator
from train.train import Train
from sklearn.model_selection import train_test_split

from multiprocessing import freeze_support

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = info, 2 = warning, 3 = error only

def extract_id_from_path(path):

    # Extract the ID based on your folder or filename structure
    # Example: if path = ".../C3N-01196/image_001.npy", then extract "C3N-01196"
    parts = path.split(os.sep)
    
    # Try returning parts[-2], or change based on your folder layout
    basename = os.path.basename(parts[-1])
    id_ = os.path.splitext(basename)[0]  # remove '.npy' -> 'UPENN-GBM-00264'

    return id_

def find_pairs(df, f_path):
    
    
    all_files = glob2.glob(f_path + "*.npy")  # or *.png, etc.

    # Create a list of (file_path, id) tuples
    file_id_pairs = [(f, extract_id_from_path(f)) for f in all_files]

    # Filter files that have IDs present in the CSV
    valid_file_id_pairs = [(f, pid) for f, pid in file_id_pairs if pid in df['id'].values]

    # Build a DataFrame of valid files with labels
    df_files = pd.DataFrame(valid_file_id_pairs, columns=["filepath", "id"])
    return df_files
        
if __name__ == "__main__":


    
    conf = Config()
    freeze_support()
 
    hp = conf.get("hp")
    if conf.get("preprocess_run"):
        dataset = conf.get("paths")
        proc = Preprocess_3D(conf, dataset)
        proc()
    if conf.get():
        
        df = pd.read_csv("dataset/meta2.csv")
        # Count category frequencies in 'gt'
        value_counts = df["gt"].value_counts()

        # Keep only categories with at least 5 instances
        valid_categories = value_counts[value_counts >= 5].index
        # Filter the dataframe
        df = df[df["gt"].isin(valid_categories)]
        print(df.head())
        
        mri_path = conf.get("paths", "mri_features")
        his_path = conf.get("paths", "his_features")
        
        mri_df_files = find_pairs(df, mri_path)
        his_df_files = find_pairs(df, his_path)
        print("MRI head:\n", mri_df_files.head())
        print("HIS head:\n", his_df_files.head())
        
        #histopathology pretraining section
        paired_his = pd.merge(df, his_df_files, on='id', how='inner')
        print(len(paired_his))
        paired_his.to_csv("his_debug.csv")
        paired_his["stratify_key"] = paired_his["grade"].astype(str) + "_" + paired_his["gt"].astype(str)
        df_his_train, df_his_val = train_test_split(paired_his, test_size=0.3, stratify=paired_his["gt"] , random_state=42)

    
        #Mri pretraining section
        paired_mri = pd.merge(df, mri_df_files, on='id', how='inner')
        paired_mri.to_csv("mri_debug.csv")
        print(len(paired_mri))
        paired_mri["stratify_key"] = paired_mri["grade"].astype(str) + "_" + paired_mri["gt"].astype(str)
        df_mri_train, df_mri_val = train_test_split(paired_mri, test_size=0.3, stratify=paired_mri["stratify_key"] , random_state=42)

        
        for d in [mri_df_files, his_df_files, df]:
            d['id'] = d['id'].astype(str).str.strip()
        
        # Merge on the common identifier (e.g., 'patient_id'), keeping only entries present in both
        paired_df = pd.merge(mri_df_files, his_df_files, on='id', how='inner')
        df_merged = pd.merge(df, paired_df, on='id', how='inner')
        
        print("MErged head:\n", len(df_merged))
        df_merged.to_csv('debug_merged.csv')
        df_merged["stratify_key"] = df_merged["grade"].astype(str) + "_" + df_merged["gt"].astype(str)


        df_train, df_temp = train_test_split(df_merged, test_size=0.3, stratify=df_merged["stratify_key"] , random_state=42)
        df_temp["stratify_key"] = df_temp["grade"].astype(str) + "_" + df_temp["gt"].astype(str)
        df_val, df_test = train_test_split(df_temp, test_size=0.5, stratify=df_temp["stratify_key"], random_state=42)

        train_gen = DataGenerator([mri_path, his_path], df_train, hp["IMG_SIZE"], batch_size=hp["batch_size"], shuffle=True)
        val_gen = DataGenerator([mri_path, his_path], df_val, hp["IMG_SIZE"], batch_size=hp["batch_size"], shuffle=False)
        test_gen = DataGenerator([mri_path, his_path], df_test, hp["IMG_SIZE"], batch_size=hp["batch_size"], shuffle=False)
        trainer = Train(hp, train_gen, val_gen, test_gen)
        
        
        model_his = None
        model_mri = None
        
        if conf.get("pretrain"):

            print("Iam activated")
            train_his = DataGenerator([mri_path, his_path], df_his_train, hp["IMG_SIZE"], pretrain=1, batch_size=hp["batch_size"], shuffle=True)
            val_his = DataGenerator([mri_path, his_path], df_his_val, hp["IMG_SIZE"], pretrain=1, batch_size=hp["batch_size"], shuffle=False)
            model_his = trainer.pretrain(train_his, val_his, train_type="his")
            
            train_mri = DataGenerator([mri_path, his_path], df_mri_train, hp["IMG_SIZE"], pretrain=1, batch_size=hp["batch_size"], shuffle=True)
            val_mri = DataGenerator([mri_path, his_path], df_mri_val, hp["IMG_SIZE"], pretrain=1, batch_size=hp["batch_size"], shuffle=False)
            model_mri = trainer.pretrain(train_mri, val_mri, train_type="mri")
        
if conf.get("train"):
        trainer.fit(model_his, model_mri)
        
if conf.get("inf"):   
        trainer.evaluate_multilabel_model()
        
if conf.get("shap"):
        trainer.shap_analysis("death")
    
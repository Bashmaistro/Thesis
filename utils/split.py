import os
import glob2
import pandas as pd
from sklearn.model_selection import  train_test_split
import time

path = '../dataset/raw/'
datasets = ['CPTAC-GBM', 'TCGA-GBM', 'UPENN-GBM', 'TCGA-LGG',]

part_contains_id = 4

def manipulate_csv():
    path =  '../dataset/metadata1.csv'
    df = pd.read_csv(path)

    df = df.replace({'female' : 1,
                       'male': 0,
                       'N' : 0,
                       'Y' : 1,
                       "_21": "",
                       "_11": ""                       
                      })
    df = df.rename(columns={'patient_id': 'id', 'gender': 'sex', 'days_to_death': 'death', "age_bins": "age"})


    return(df)

selected = []
count= []

def select_patients(df):

    image_list = []
    for dataset in datasets:
        path1 = path + dataset
        subfolders = [f for f in glob2.glob(path1 + '/**/') if os.path.isdir(f)]
        for folder_name in subfolders:
            dcm_list = os.listdir(folder_name)
            if any(file.endswith('.dcm') for file in dcm_list):
                print(folder_name)
                if folder_name not in image_list:  
                    image_list.append(folder_name)
        
                    parts = folder_name.split('/')
                    df_row = df[df["id"] == parts[part_contains_id]]
                    print(parts[part_contains_id])
        
                    # prevent duplicate DataFrame rows (based on ID)
                    if not df_row.empty and not df_row.iloc[0].id in [row.iloc[0].id for row in selected]:
                        selected.append(df_row)
    df = pd.concat(selected, ignore_index=True)
    print(selected)
    
    with open("selected_dirs.txt", "w") as f:
        for folder in image_list:
            f.write(folder + "\n")

    return df, image_list



def filter_cases(image_dirs):
    keywords = [
        't1%20axial%20stealth-post%20%20ProcessedCaPTk',
        "C%20Ax%20T1%20MP%20SPGR",
        "ax%203d%20fspgr%20c",
        "stealth-post",
        '3D%20WAND',
        'stealthvecstealth',
        "T1%203D%20fSPGR-IR",
        '3D%20SPGR%20POST',
        "MP%20RAGE%20AXIAL",
        "AX%20FSPGR%20C",
        "AX%203D%20SPGRC",
        "FSPGR%203D",
        "AX%203D%20STRYKER",
        "AxT1-thin%20for%20surgery",
        "Ax%20FSPGR%203DC",
        "FSPGR%20BRAVO%201.0mm%20AXC",
        "FSPGR%20BRAVO%201.0mm%20AX",
        "VIBE%203D%20AX%20POST",
        "T1%203D%20MP%20RAGE",
        "3D%20T1%20TFEWAND",
        "-ax%20spgr%",
        "-3D-MP-RAGE",
        "ISOTROPIC%20CONTRAST",
        "01.000000-STRYKER",
        "RAGE-STRY",
        "AXIAL T1 POST GD",
        "AXIAL%20T1%20POST%20GD",
        "AXIAL%20T1%20PRE-GAD",
        "T1mprageAx%20Gd",
        "POST%20AX%20T1%20BRAIN%20LAB%201MM",
        "Ax%20T1%202.5mm%20for%20surgery",
        "Post%20AX%20T1WIRTSE",
        "t1mprnssagisoce",
        "t1mprnstraisoce",
        "t1mprtraiso",
        "T1%20AXIAL%20Gd",
        "AXIAL%20T1%20GD",
        "Ax%20T1%20SE",
        "t1mpragetra%20Gd",
        "T1W3Dstryker",
        "T1W3DSTRYKER",
        "AX%203D%20SPGRC",
        "18.000000-3D%20AXIALIRSPGRFast",
        "16.000000-3D%20AXIALIRSPGRFast",
        "3D%20AXIALIRSPGRFast",
        "T1%20MP%20SPGR",
        "-ax%20t1",
        "AX%20T1C",
        "AXIALIRSPGR",
        "t1AX",
        "%20SPGR",
        "AX%20FSPGR",
        "AX%20T1%20POST%20GD%20FLAIR",
        "AX%20T1%20pre%20gd",
        "Axial%20T1%20FSE%20Post%20Gad",
        "Axial%20T1%20FSE",
        "%20AXIAL%20Gd",
        "AX%203D%20SPGR",
        "VOLUMETRIC%20AXIAL",
        "T1%20AX",
        "T1AXIAL",
        "AXIAL%20T1",
        "Ax%20T1%20SE",
        "t1mpragetra",
        "AX%20T1",
        "Ax%20T1%20FS%20BRAIN%20POST",
        "Ax%20T1",
    ]
    

    ct_list = []

    for folder in image_dirs:
        control_list = []
        folders = [f for f in glob2.glob(os.path.dirname(os.path.dirname(folder)) + '/*/') if os.path.isdir(f)]
        for folder_name in folders:
    
            control_list.append(folder_name)

        flag = False   
        for key in keywords:
            for cntrl in control_list:
                if key in cntrl:
                   ct_list.append(cntrl)
                   flag = True
                   break
            if flag:
               break
               
        # for item in control_list :
        #     if any(keyword in item for keyword in keywords):
        #             #print(item)
        #             ct_list.append(item)
        #             break
        #     else:
        #         continue
    print(len(ct_list))
    return ct_list

def dataset_out(df, dirs):
    df['dataset'] = df['id'].str.contains('AR')
    for folder_dir in dirs:
        p_id = folder_dir.split('/')[part_contains_id]
        df.loc[df['id'] == p_id, 'CT_dir'] = folder_dir
        continue
    df = df.dropna(subset=['CT_dir'])
    return df

def split(df):

    # Create the 'stratify_group' column by combining 'dataset' and 'icu'
    df['stratify_group'] = df[['dataset', "death", "sex"]].astype(str).agg('-'.join, axis=1)
    df = df.drop(['sex', 'dataset'], axis=1)
    df.to_csv('../dataset/temp.csv')
    # First split: train (60%) and val_test (40%)
    train, val_test = train_test_split(
        df,
        test_size=0.4,
        stratify=df['stratify_group'],
        random_state=42
    )
    # Second split: val_test into val (50%) and test (50%)
    val, test = train_test_split(
        val_test,
        test_size=0.5,
        stratify=val_test['stratify_group'],
        random_state=42
    )


    train = train.drop(['stratify_group'], axis=1)    
    val = val.drop(['stratify_group'], axis=1)
    test = test.drop(['stratify_group'], axis=1)
    print(train.head())
    
    return train, val, test


if __name__ == '__main__':
    # preprocessing the metadata
    df = manipulate_csv()
    #CTs are selected
    selected, image_dirs =select_patients(df)
    df_sorted = selected.sort_values(by='id').reset_index(drop=True)
    ct_dir = filter_cases(image_dirs)
    df = dataset_out(selected, ct_dir)
    df.to_csv('../dataset/result.csv')
    df.to_csv('../dataset/selected.csv')
    train, val, test = split(df)

    print('Train dataset is number: ' + str(len(train)) )
    print('Validation dataset is number: ' + str(len(val)) )
    print('Test dataset is number: ' + str(len(test)) )
    train.to_csv('../dataset/train.csv')
    val.to_csv('../dataset/val.csv')
    test.to_csv('../dataset/test.csv')
    
    
    





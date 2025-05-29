import os
import pandas as pd
import tensorflow as tf


from dataset.dataloader import DataGenerator
from utils.lung_segment import *
from train.train import Train
from augmentation.augment import Augment_3D

set_dataset = 1

" hyper-parameters..."
hp = 

    "MAX_SEQ_LENGTH": 500, 
    "NUM_FEATURES": 960}

os.environ["CUDA_DEVICES_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir="/home/lab-pc-1/miniconda3/envs/p310/lib"'
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



datasets = ['train', 'val', 'test']

if set_dataset:
    for dataset in datasets:
        df = pd.read_csv('dataset/' + dataset +'.csv')

        crop = crop_dataset(df, dataset)
        crop()

        if not dataset == "test":
            aug = Augment_3D(df, dataset)
            aug()


df_train = pd.read_csv('dataset/train.csv')
df_val = pd.read_csv('dataset/val.csv')
df_test = pd.read_csv('dataset/test.csv')

train_gen = DataGenerator('train', df_train, hp["IMG_SIZE"], batch_size=hp["batch_size"])
val_gen = DataGenerator('val', df_val, hp["IMG_SIZE"], batch_size=hp["batch_size"])
test_gen = DataGenerator('test', df_test, hp["IMG_SIZE"],)

train_rnn = Train(hp, train_gen, val_gen, test_gen)

train_rnn.fit()
train_rnn.evaluate()

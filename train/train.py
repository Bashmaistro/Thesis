import os
import numpy as np
import tensorflow as tf
from keras import layers
from keras.optimizers import Adam
import re
import datetime
import io
import sys
import pandas as pd
import matplotlib.pyplot as plt
from inference import plot_loss, saveResults, plot_roc_curve_binary
import inference
import keras
from utils import *
from utils.loss import focal_loss

from keras.models import Model
from tensorflow.keras.layers import Input
from model.model import RNN_Model, CNN_RNN_Model, CNN_3D, Multimodel, build_his_block_2, build_rnn_block, build_his_block
from keras.losses import CategoricalFocalCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import shap
from keras.models import load_model
  
def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig("plot/" + class_names[0] + "_" + str(timestamp), dpi=600)
    plt.show()
    
def plot_history_fn(history, base_key, timestamp):
    """
    Plots training and validation history for all output labels using the given base metric (e.g., "loss", "accuracy").

    Args:
        history: Keras History object or a dict with training history.
        base_key: The base metric to search for (e.g., 'loss', 'accuracy').
    """
    if not history:
        print("No history to plot.")
        return

    hist = history.history if hasattr(history, "history") else history

    # Match keys like 'emotion_loss', 'val_emotion_loss', etc.
    pattern = re.compile(rf"^(val_)?(.+?)_{base_key}$")

    matched = [k for k in hist if pattern.match(k)]
    if not matched:
        print(f"No entries found for base key: '{base_key}'")
        return

    # Find unique output labels
    labels = sorted(set(match.group(2) for k in matched if (match := pattern.match(k))))

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    color_cycle = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    for label in labels:
        color = next(color_cycle)
        train_key = f"{label}_{base_key}"
        val_key = f"val_{label}_{base_key}"

        if train_key in hist:
            ax.plot(hist[train_key], label=f"{label} - Train", linestyle='-', color=color)

        if val_key in hist:
            ax.plot(hist[val_key], label=f"{label} - Val", linestyle='--', color=color)

    plt.title(f"Training and Validation {base_key.capitalize()} for All Outputs")
    plt.xlabel("Epochs")
    plt.ylabel(base_key.capitalize())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot/" + str(base_key) + "_"+ str(timestamp), dpi=600)
    plt.show()
    
class Train():

    def __init__(self, hp, train_gen, val_gen, test_gen = None):
        import tensorflow as tf
        os.environ["CUDA_DEVICES_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = info, 2 = warning, 3 = error only
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        # os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir="/home/lab-pc-1/miniconda3/envs/p310/lib"'
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        self.hp = hp
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.test_gen = test_gen
        self.history = None
        self.model = None
        # self.model = CNN_RNN_Model(self.hp["MAX_SEQ_LENGTH"])
        #passing the hyperparameters
        
    def pretrain(self,train_gen, val_gen, train_type="his"):
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        checkpoint = ModelCheckpoint(
        filepath = f'plot/{train_type}_best_model.keras',         # or use '.keras' if preferred
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
        # loss=CategoricalFocalCrossentropy(alpha=0.25, gamma=3.0),
        

        if train_type == "his":
            opt = Adam(learning_rate=self.hp["lr_his"])
            focal_loss = CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0)
            
            his_features_input = keras.Input(shape=(self.hp["NUM_FRAMES"], self.hp["FRAME_SIZE"]), name="his_sequence")
            base_model = build_his_block(his_features_input)  # this returns a Keras Model
            
            
        elif train_type == "mri":
            opt = Adam(learning_rate=self.hp["lr_mri"])
            focal_loss = CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0)
            frame_features_input = keras.Input(shape=(self.hp["MAX_SEQ_LENGTH"], self.hp["NUM_FEATURES"]), name="image_sequence")
            base_model = build_rnn_block(frame_features_input)  # this returns a Keras Model
            
            
        else:
            print("Model name is unknown")
          
        # Add a basic Dense output layer to the selected block
        x = base_model.output
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='relu')(x)
        
        # Output branches
        death = layers.Dense(3, activation="softmax", name="death")(x)
        grade = layers.Dense(2, activation="softmax", name="grade")(x)
        mortality = layers.Dense(2, activation="softmax", name="mortality")(x)
        gt = layers.Dense(3, activation="softmax", name="gt")(x)


        model = Model(inputs=base_model.input, outputs=[death, grade, mortality, gt])
        
        model.compile(
            optimizer=opt,
            loss={
                "death": focal_loss,
                "grade": focal_loss,
                "mortality": focal_loss,
                "gt": focal_loss
            },
            metrics={
                "death": "accuracy",
                "grade": "accuracy",
                "mortality": "accuracy",
                "gt": "accuracy"
            }
        )
        history = model.fit(
            train_gen, 
            validation_data=val_gen, 
            epochs = self.hp["epochs"],
            callbacks=[reduce_lr, early_stop, checkpoint],
            verbose=2
            )
        return model
        
    
    def fit(self, model_his=None, model_mri=None):
        try:
            best_model_path = f'plot/his_best_model.keras'
            model_his = load_model(best_model_path)
            
            best_model_path = f'plot/mri_best_model.keras'
            model_mri = load_model(best_model_path)
            
            self.model = Multimodel(self.hp, model_his, model_mri)
            print(f"pretrained model from previous attempts loaded ")
        except Exception as e:
            print(f"Failed to load best model: {e}")
            
        #if not  model_his == None and not model_his == None:
        #    self.model = Multimodel(self.hp, model_his, model_mri)
        
        
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        checkpoint = ModelCheckpoint(
        filepath='plot/best_model.keras',         # or use '.keras' if preferred
        monitor='val_death_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
        # loss=CategoricalFocalCrossentropy(alpha=0.25, gamma=3.0),
        
        opt = Adam(learning_rate=self.hp["lr"])
        focal_loss = CategoricalFocalCrossentropy(alpha=0.25, gamma=3.0)
        self.model.compile(
                            optimizer=opt, 
                            loss={
                                    "death": focal_loss,
                                    "grade": focal_loss,
                                    "mortality": focal_loss,
                                    "gt": focal_loss
                                },
                            metrics={
                                    "death": "accuracy",
                                    "grade": "accuracy",
                                    "mortality": "accuracy",
                                    "gt": "accuracy"
                                }
)

        self.history = self.model.fit(
            self.train_gen, 
            validation_data=self.val_gen, 
            epochs = self.hp["epochs"],
            callbacks=[reduce_lr, early_stop, checkpoint],
            verbose=2
            )
        return self.model
    
    def evaluate_multilabel_model(self, threshold=0.5):
        """
        Evaluates a multi-output model with categorical predictions.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


        label_names_dict = {
            "death": ["0-370", "370-1448", "1448"],
            "grade": ["LGG", "HGG"],
            "mortality": ["Survived", "Died"],
            "gt": ["Oliogendroma", "Astrocytoma", "Glioblastoma"]
        }

        self.model = keras.models.load_model("plot/best_model.keras")
        test_gen = self.test_gen
        output_names = list(label_names_dict.keys())
        
        if not type(self.history) == None:
            plot_history_fn(self.history, "loss", timestamp)
            plot_history_fn(self.history, "accuracy", timestamp)

        # Collect all inputs and labels
        num_inputs = len(test_gen[0][0])
        X_lists = [[] for _ in range(num_inputs)]
        y_lists = {key: [] for key in output_names}

        for i in range(len(test_gen)):
            X_batch, y_batch_dict = test_gen[i]
            for j, x in enumerate(X_batch):
                X_lists[j].append(x)
            for key in output_names:
                y_lists[key].append(y_batch_dict[key])

        X_all = [np.concatenate(x_list, axis=0) for x_list in X_lists]
        y_true_dict = {key: np.concatenate(y_lists[key], axis=0) for key in output_names}

        y_pred_dict = self.model.predict(X_all, batch_size=1)
        if isinstance(y_pred_dict, list):
            y_pred_dict = dict(zip(output_names, y_pred_dict))
            
        for key in output_names:
            print(f"\n========== Evaluating: {key.upper()} ==========")
            
            y_true = y_true_dict[key]
            y_pred = y_pred_dict[key]

            y_true_cls = np.argmax(y_true, axis=1)
            y_pred_cls = np.argmax(y_pred, axis=1)

            print("Class distribution in ground truth:")
            unique, counts = np.unique(y_true_cls, return_counts=True)
            for cls, count in zip(unique, counts):
                name = label_names_dict[key][cls]
                print(f"  {name}: {count} samples")

            print("\nConfusion Matrix:")
            print(confusion_matrix(y_true_cls, y_pred_cls))
            plot_confusion_matrix(y_true_cls, y_pred_cls, label_names_dict[key], title=f"{key.upper()} Confusion Matrix")
            print("\nClassification Report:")
            print(classification_report(
                y_true_cls, y_pred_cls, target_names=label_names_dict[key], zero_division=0
        ))

        print(f"\n========== Evaluating: {key.upper()} ==========")
        
        y_true = y_true_dict[key]
        y_pred = y_pred_dict[key]

        y_true_cls = np.argmax(y_true, axis=1)
        y_pred_cls = np.argmax(y_pred, axis=1)

        print("Class distribution in ground truth:")
        unique, counts = np.unique(y_true_cls, return_counts=True)
        for cls, count in zip(unique, counts):
            name = label_names_dict[key][cls]
            print(f"  {name}: {count} samples")

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true_cls, y_pred_cls))
        plot_confusion_matrix(y_true_cls, y_pred_cls, label_names_dict[key], title=f"{key.upper()} Confusion Matrix")
        print("\nClassification Report:")
        print(classification_report(
            y_true_cls, y_pred_cls, target_names=label_names_dict[key], zero_division=0
        ))
            
    def shap_analysis(self, output_name, sample_size=50):
        for (X_sample, _), (background, _) in zip(self.test_gen, self.train_gen):
            shap_values, [mri_pct, clinical_pct, his_pct] = inference.shapley(self.model, X_sample, background, output_name)
    
 


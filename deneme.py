import os
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import cv2

# Ayarlar
DATASET_PATH = 'dataset'  # dataset dizini yolu
BATCH_SIZE = 200
EPOCHS = 10
IMAGE_SHAPE = (224, 224, 3)  # VGG16 input boyutu

# 1. Klasörlerden class label'larını ve dosya yollarını oku
def load_dataset_info(dataset_path):
    class_names = []
    data_files = []

    for class_dir in sorted(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, class_dir)
        
        if not os.path.isdir(class_path) or class_dir.startswith('__'):
            continue
        if os.path.isdir(class_path):
            class_names.append(class_dir)
            npy_files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.npy')]
            for f in npy_files:
                data_files.append((f, class_dir))

    return class_names, data_files

# 2. Custom data generator (memory dostu)
class NPYDataGenerator(Sequence):
    def __init__(self, data_files, label_encoder, batch_size=BATCH_SIZE, shuffle=True):
        self.data_files = data_files
        self.label_encoder = label_encoder
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Data index listesi [(file_path, class_name, start_idx, end_idx)]
        self.index_list = []
        self.build_index_list()

    def build_index_list(self):
        self.index_list.clear()
        for file_path, class_name in self.data_files:
            arr_shape = np.load(file_path, mmap_mode='r').shape  # (num_slices, 240, 240)
            num_slices = arr_shape[0]

            for start in range(0, num_slices, self.batch_size):
                end = min(start + self.batch_size, num_slices)
                self.index_list.append((file_path, class_name, start, end))

        if self.shuffle:
            np.random.shuffle(self.index_list)

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):
        file_path, class_name, start_idx, end_idx = self.index_list[index]

        # npy dosyasından sadece ilgili slice'ı yükle
        npy_data = np.load(file_path, mmap_mode='r')
        X_slices = npy_data[start_idx:end_idx]  # shape: (slice_count, 240, 240)

        # Resize ve kanal boyutu ekleme (224, 224, 3)
        X_resized = np.array([
            cv2.resize(slice, (224, 224)) for slice in X_slices
        ])

        # Grayscale ise (slice, 224, 224) → (slice, 224, 224, 1)
        if len(X_resized.shape) == 3:
            X_resized = np.expand_dims(X_resized, axis=-1)

        # (slice, 224, 224, 1) → (slice, 224, 224, 3) (channel repeat)
        if X_resized.shape[-1] == 1:
            X_resized = np.repeat(X_resized, 3, axis=-1)

        # Normalize
        X_resized = X_resized.astype('float32') / 255.0

        # Label
        y_label = self.label_encoder.transform([class_name])[0]
        y = np.full((X_resized.shape[0],), y_label)

        return X_resized, y
    def on_epoch_end(self):
        self.build_index_list()

# 3. Model oluştur (Transfer Learning)
def create_model(num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=IMAGE_SHAPE)
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# 4. Main
if __name__ == "__main__":
    class_names, data_files = load_dataset_info(DATASET_PATH)
    print(f"Sınıflar: {class_names}")
    print(f"Toplam NPY dosyası: {len(data_files)}")

    # Label encode (class isimlerini sayıya çevir)
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    
    print(class_names)

    # Data generator tanımla
    train_generator = NPYDataGenerator(data_files, label_encoder)

    # Modeli oluştur
    model = create_model(num_classes=len(class_names))
    model.summary()

    # Eğitimi başlat
    model.fit(train_generator, epochs=EPOCHS)

    # Modeli kaydet
    model.save("vgg16_incremental_model.keras")

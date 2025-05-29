import os
import numpy as np
import pydicom
import nibabel as nib

from cuml.cluster import KMeans as cuKMeans
import cupy as cp
import gc
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import numpy as np
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path




def center_crop(slice_array, size=240):
    """Slice'ı ortasından crop'la."""
    
    if slice_array.ndim == 2:
        h, w = slice_array.shape
        top = max((h - size) // 2, 0)
        left = max((w - size) // 2, 0)
        return slice_array[top:top + size, left:left + size]

    elif slice_array.ndim == 3:
        h, w, c = slice_array.shape
        top = max((h - size) // 2, 0)
        left = max((w - size) // 2, 0)
        return slice_array[top:top + size, left:left + size, :]


def load_and_filter(file_path):
    """Dosya tipine göre yükle, slice'ları filtrele ve crop yap."""
    ext = file_path.lower()
    

    if ext.endswith('.svs') or ext.endswith('.ndpi'):
        return load_svs_ndpi(file_path)

    elif ext.endswith('.dcm'):
        ds = pydicom.dcmread(file_path)
        slices = []
        arr = ds.pixel_array
        
        print("arr shape:", arr.shape)
        print("arr dtype:", arr.dtype)
        print("arr min/max:", arr.min(), arr.max())
        
        print("deneme1 " + str(arr.shape))
        print("deneme2 " , arr.shape[0])

        plt.imsave("uploads/send/sample_crop_plt.png", arr[2500])
        
        for i in range(arr.shape[0]):
            if arr[i].mean() < 230:
                cropped = center_crop(arr[i])
                slices.append(cropped)
        return np.array(slices)

    elif ext.endswith('.npy'):
        arr = np.load(file_path)
        slices = []
        for i in range(arr.shape[0]):
            cropped = center_crop(arr[i])
            slices.append(cropped)
        return np.array(slices)

    elif ext.endswith('.nii') or ext.endswith('.nii.gz'):
        nii = nib.load(file_path)
        arr = nii.get_fdata()
        if arr.ndim == 4:
            arr = arr[..., 0]
        slices = []
        for i in range(arr.shape[2]):
            slice_img = arr[:, :, i]
            if slice_img.mean() < 230:
                cropped = center_crop(slice_img)
                slices.append(cropped)
        return np.array(slices)

    else:
        raise ValueError(f"Desteklenmeyen dosya tipi: {file_path}")


def load_svs_ndpi(file_path, level=0, tile_size=240):
    """OpenSlide ile SVS veya NDPI'den 240x240 crop'lar çıkar."""
    slide = OpenSlide(file_path)
    width, height = slide.level_dimensions[level]
    step = tile_size
    slices = []

    for y in range(0, height, step):
        for x in range(0, width, step):
            if x + tile_size <= width and y + tile_size <= height:
                tile = slide.read_region((x, y), level, (tile_size, tile_size)).convert("L")
                tile_np = np.array(tile)
                if tile_np.mean() < 230:
                    slices.append(tile_np)
    slide.close()
    return np.array(slices)


def sample_from_clusters(data_array: np.ndarray,
                         saved_centroids: np.ndarray,
                         n_clusters: int,
                         target_clusters: list,
                         total_slices: int,
                         batch_size: int = 128,
                         target_shape=(240, 240)) -> np.ndarray:

    data = data_array.reshape(data_array.shape[0], -1)
    centroids_cp = cp.asarray(saved_centroids)
    kmeans_model = cuKMeans(n_clusters=n_clusters, init=centroids_cp, max_iter=100)

    all_labels = []
    cluster_to_samples = {c: [] for c in target_clusters}

    for i in range(0, data.shape[0], batch_size):
        batch = data[i:i + batch_size]
        labels = cp.asnumpy(kmeans_model.predict(cp.asarray(batch.astype(np.float32))))
        all_labels.extend(labels.tolist())
        for j, label in enumerate(labels):
            if label in cluster_to_samples:
                cluster_to_samples[label].append(batch[j])

    all_labels_np = np.array(all_labels)
    total_cluster_counts = {c: (all_labels_np == c).sum() for c in target_clusters}
    total_count_in_targets = sum(total_cluster_counts.values())

    final_samples = []
    for c in target_clusters:
        ratio = total_cluster_counts[c] / total_count_in_targets if total_count_in_targets > 0 else 0
        target_count = int(round(total_slices * ratio))
        used_count = min(len(cluster_to_samples[c]), target_count)
        final_samples.extend(cluster_to_samples[c][:used_count])

    remaining = total_slices - len(final_samples)
    if remaining > 0:
        other_clusters = [i for i in range(n_clusters) if i not in target_clusters]
        extras = {c: [] for c in other_clusters}
        for i, label in enumerate(all_labels_np):
            if label in other_clusters:
                extras[label].append(data[i])
        for c in other_clusters:
            if remaining <= 0:
                break
            take = min(remaining, len(extras[c]))
            final_samples.extend(extras[c][:take])
            remaining -= take

    result = np.array([np.resize(x, target_shape) for x in final_samples])
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    return result

def convert_to_grayscale(data):
    # data: (slices, height, width, channels)
    if data.shape[-1] == 3:
        # RGB -> Grayscale
        grayscale_data = np.dot(data[...,:3], [0.2989, 0.5870, 0.1140])
    elif data.shape[-1] == 1:
        # Zaten grayscale
        grayscale_data = data[..., 0]
    else:
        raise ValueError("Desteklenmeyen kanal sayısı: {}".format(data.shape[-1]))
    
    return grayscale_data

def extract_features_from_numpy(processed_data: np.ndarray,
                                 model_path: str = 'utils/vgg_model.keras',
                                 target_shape=(224, 224)) -> np.ndarray:
    """
    Verilen (N, H, W) formatındaki grayscale numpy array'den
    VGG16 ile öznitelik çıkarır. Model bu fonksiyon içinde yüklenir.
    """

    print("📦 Özellik çıkarımı başladı...")

    # Model yükle
    vgg16 = load_model(model_path)
    feature_extractor = tf.keras.Model(
        inputs=vgg16.input,
        outputs=vgg16.get_layer("global_average_pooling2d").output
    )

    # (N, 240, 240) → (N, 224, 224)
    resized = tf.image.resize(tf.convert_to_tensor(processed_data[..., np.newaxis]), target_shape)

    # (N, 224, 224, 1) → (N, 224, 224, 3)
    rgb_like = tf.repeat(resized, repeats=3, axis=-1)

    # Normalizasyon
    rgb_like = tf.cast(rgb_like, tf.float32) / 255.0

    # Özellik çıkar
    features = feature_extractor.predict(rgb_like, verbose=1)

    print(f"✅ Özellik çıkarımı tamamlandı. Şekil: {features.shape}")
    return features

def normalize_slice_float64(img):
    img = img.astype(np.float64)
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val - min_val < 1e-12:  # Çok küçük varyans varsa
        return np.zeros_like(img, dtype=np.uint8)
    norm_img = (img - min_val) / (max_val - min_val)
    norm_img = (norm_img * 255).astype(np.uint8)
    return norm_img

def save_slices_direct_uint8(clustered_slices, output_path, num_slices=10):
    os.makedirs(output_path, exist_ok=True)
    selected_indices = random.sample(range(clustered_slices.shape[0]), num_slices)

    for i, idx in enumerate(selected_indices):
        slice_img = clustered_slices[idx]
        
        # Küçük değerler olsa da 0-255 aralığında float, direk uint8 yap
        img_uint8 = slice_img.astype(np.uint8)
        
        image = Image.fromarray(img_uint8, mode='L')
        image.save(os.path.join(output_path, f"image_{i}.png"))


def preprocess_his_file(file_path: str,
                 cluster_centroid_path: str = "utils/test_inc_cluster_centers_10.npy",
                 n_clusters: int = 10,
                 target_clusters: list = [8, 6, 1],
                 total_slices: int = 2500,
                 target_shape: tuple = (240, 240)) -> np.ndarray:
    """
    Tüm işlemleri yapar: yükleme, filtreleme, crop, kümeleme → np.ndarray olarak döner.
    """
    print(f"\n📁 İşlem Başladı: {file_path}")
    raw_data = load_and_filter(file_path)
    
    
    raw_data = convert_to_grayscale(raw_data)
    
    print("deneme")

    if raw_data.shape[0] < total_slices:
        raise ValueError(f"Yetersiz slice sayısı: {raw_data.shape[0]} < {total_slices}")

    centroids = np.load(cluster_centroid_path)
    clustered = sample_from_clusters(
        data_array=raw_data,
        saved_centroids=centroids,
        n_clusters=n_clusters,
        target_clusters=target_clusters,
        total_slices=total_slices,
        target_shape=target_shape
    )
    
    print("Clustered dtype:", clustered.dtype)
    print("Clustered min:", np.min(clustered))
    print("Clustered max:", np.max(clustered))
    print("Clustered mean:", np.mean(clustered))
    print("Clustered shape:", clustered.shape)

    print(f"✅ İşlem tamamlandı. Şekil: {clustered.shape}")
    
    print(f"clustered min: {clustered.min()}, max: {clustered.max()}, dtype: {clustered.dtype}")


    save_slices_direct_uint8(clustered, "uploads/send", 10)
   
    
    
    result = extract_features_from_numpy(clustered)
    return result




# === Örnek kullanım ===
if __name__ == "__main__":
    path = "uploads/histo/TCGA-CS-4942_9b57f33e-7e3a-4ea0-ad16-cd77ee964722.dcm"  # ← Buraya herhangi bir .svs, .ndpi, .dcm, .npy, .nii ver
    try:
        result = preprocess_his_file(path)
        print("Sonuç shape:", result.shape)
    except Exception as e:
        print("Hata:", e)

import os
import time
import numpy as np
import pydicom as pd
import cupy as cp
import cv2
import math
from cuml.cluster import KMeans as cuKMeans
import matplotlib.pyplot  as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


width= 476
height= 276
def process_dcm_file(dcm_path, save_dir):
    try:
        # Read using force to speed up + stop after pixel data
        ds = pd.dcmread(dcm_path, force=True, stop_before_pixels=False)
        
        data = []
        
        for arr in pd.iter_pixels(ds):
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            data.append(arr)

        # # Decode JPEG2000 more efficiently (if applicable)
        # arr = ds.pixel_array  # Pydicom will use pylibjpeg/gdcm if installed

        # # Optional: apply VOI LUT if necessary
        # # arr = apply_voi_lut(arr, ds)
        # if arr.shape[-1] == 3:
        #     arr = (
        #             0.2989 * arr[..., 0] + 
        #             0.5870 * arr[..., 1] + 
        #             0.1140 * arr[..., 2]
        #             )
            
        # else:
        #     raise ValueError("Expected RGB images with shape (..., 3)")
        arr = np.array(data)   
        del data 
        ver_cent = arr.shape[1]//2
        hor_cent = arr.shape[2]//2
        
        arr = arr[:,ver_cent-120:ver_cent+120,hor_cent-120:hor_cent+120]
        # Build output path
        pid = os.path.basename(dcm_path)
        output_path = save_dir + ".npy"
        
        # Save as NumPy file
        np.save(output_path, arr)
        del arr
        width = ds.TotalPixelMatrixColumns//240
        height = ds.TotalPixelMatrixRows//240
        return output_path

    except Exception as e:
        print(f"Failed to process {dcm_path}: {e}")
        
def predict_cluster(input_path, model_path, batch_size=1000, n_clusters=10, show_centroids=False):
    if input_path.endswith(".dcm"):
        input_path = process_dcm_file(input_path, input_path)
   
    data = np.load(input_path)
    print(data.shape)
    data = data.reshape(data.shape[0], -1)
    
    
    saved_centroids = np.load(model_path)  # shape = (10, 57600)

    # Move to GPU
    saved_centroids_cp = cp.asarray(saved_centroids)

        # Shape: (10, 57600) → reshape each to (240, 240)
    centroids = saved_centroids
    plt.figure(figsize=(15, 5))
    for i in range(n_clusters):
        plt.subplot(2, 5, i + 1)
        plt.imshow(centroids[i].reshape(240, 240), cmap='gray')
        plt.title(f"Cluster {i}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


    kmeans_loaded = cuKMeans(
        n_clusters=n_clusters,
        init=saved_centroids_cp,   # ✅ Pass the centroids directly here!
        max_iter=100
    )
    all_labels = []
    
    for i in range(0, data.shape[0], batch_size):
        batch = data[i:i+batch_size]
        batch = batch.astype(np.float32)
        batch_cp = cp.asarray(batch)
        labels_cp = kmeans_loaded.predict(batch_cp)
        labels_np = cp.asnumpy(labels_cp)
        all_labels.append(labels_np)

    # Concatenate all batch results
    all_labels = np.concatenate(all_labels)
    print("All labels predicted:", all_labels.shape)

    return all_labels



batch_size = 5000
input_path = "TCGA-DH-A7UT_25298a64-8ec7-4b0d-a391-2526e61c0fb4.dcm.npy"
model_path = "test_inc_cluster_centers_10.npy"
n_clusters = 10


all_labels = predict_cluster(input_path, model_path, batch_size=batch_size, n_clusters=n_clusters)  
  
  
  
unique, counts = np.unique(all_labels, return_counts=True)

plt.figure(figsize=(7, 7))
plt.pie(counts, labels=[f"Cluster {i}" for i in unique], autopct='%1.1f%%', startangle=140)
plt.title("Proportion of Images per Cluster")
plt.axis('equal')
plt.show()



# num_patches = len(all_labels)  # 2883 in your case

# # Dynamically determine canvas size
# rows = int(math.sqrt(num_patches))
# cols = math.ceil(num_patches / rows)

# # Create canvas filled with -1
# canvas = -1 * np.ones((rows, cols), dtype=int)

# # Fill canvas with actual labels
# canvas.flat[:num_patches] = all_labels

# # Optional colormap setup
# cmap = ListedColormap(plt.cm.tab10.colors)  # or another suitable colormap

# Define the number of patches
num_patches = len(all_labels)

np.save('cluster_labels.npy' , np.array(all_labels))

# Trim the array to match the target size
trimmed_labels = all_labels[:height * width]

# Reshape the trimmed array
canvas = np.reshape(trimmed_labels, (height, width))

# Create an empty canvas filled with -1 (unused area will be -1)
canvas = 0 * np.ones((height, width), dtype=int)

# Fill canvas with cluster labels
canvas.flat[:num_patches] = all_labels

# Define a colormap (e.g., 10 colors)
cmap = ListedColormap(plt.cm.tab10.colors)
# Example label array
# Assume all_labels is already defined

# Plot it
plt.figure(figsize=(12, 12))
plt.imshow(canvas, cmap=cmap, vmin=0, vmax=9)
cbar = plt.colorbar(ticks=np.arange(n_clusters))
cbar.set_label("Cluster ID")
plt.title("Cluster Distribution Map of All Images (1 Patch = 1 Image)")
plt.axis('off')
plt.show()



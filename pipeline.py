import os
import sys
import shutil
from Pipeline.Filterize import Filterizer,Filterizer_ndpi,Filterizer_svs
from Pipeline.Kmeanspredict import sample_from_clusters
import numpy as np
import cupy as cp
from cuml.cluster import KMeans as cuKMeans

klasor_yolu = 'PKG - CPTAC-GBM_v16/GBM'  # Girdi klasörü
tasinan_dosyalar_klasoru = 'islenen_svsler_cptac'  # İşlenen dosyaların taşınacağı klasör
dusuk = '2500dendusuk_cptac'
os.makedirs(tasinan_dosyalar_klasoru, exist_ok=True)
os.makedirs(dusuk, exist_ok=True)
save_dir = "cptac_2500_mlp"

dcm_path_listesi = []

for root, _, files in os.walk(klasor_yolu):
    for file in files:
        if file.lower().endswith('.svs'):
            tam_yol = os.path.join(root, file)
            
            name = file.split("/")[0].split(".")[0]
            print("Dosya Isleniyor:" + name)
            
            result = Filterizer_svs(tam_yol)
            print("Dosya Filtrelendi:" + name)
            
            if result.shape[0] < 2500:
                print(f"❌ {name} dosyası yetersiz slice sayısına sahip ({result.shape[0]} < 2500). Atlanıyor.")
                  # Orijinal .dcm dosyasını başka klasöre taşı
                hedef_yol = os.path.join(dusuk, file)
                shutil.move(tam_yol, hedef_yol)
                print("Dosya Tasindi:" + file)
                
                continue  # Bu dosyayı atla, döngüde devam et
            
            result = sample_from_clusters(
                data_array=result,
                saved_centroids=np.load("test_inc_cluster_centers_10.npy"),
                n_clusters=10,
                target_clusters=[8, 6, 1 ],
                total_slices=2500
            )
            print("Dosya Kumelendi:" + name)
            
           
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, name), result)
            print("Dosya Kaydedildi")
            
            # Orijinal .dcm dosyasını başka klasöre taşı
            hedef_yol = os.path.join(tasinan_dosyalar_klasoru, file)
            shutil.move(tam_yol, hedef_yol)
            print("Dosya Tasindi:" + file)
            
            
            
            
        else:
            tam_yol = os.path.join(root, file)
            name = file.split("/")[0].split(".")[0]
            
            np_file = np.load(tam_yol)
            
            ver_cent = np_file.shape[1]//2
            hor_cent = np_file.shape[2]//2

            size = 240//2
            result = np_file[:,ver_cent-size:ver_cent+size,hor_cent-size:hor_cent+size]

            
            
            
            print(f"✅  {name} dosyası numpy olarak yakalandi")
            
            result = sample_from_clusters(
                data_array=result,
                saved_centroids=np.load("test_inc_cluster_centers_10.npy"),
                n_clusters=10,
                target_clusters=[8, 6, 1],
                total_slices=2500
            )
            print("Dosya Kumelendi:" + name)
            
            
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, name), result)
            print("Dosya Kaydedildi")
            
            # Orijinal .dcm dosyasını başka klasöre taşı
            hedef_yol = os.path.join(tasinan_dosyalar_klasoru, file)
            shutil.move(tam_yol, hedef_yol)
            print("Dosya Tasindi:" + file)
            print(result.shape)
            

    del result
    del name
    del file
    cp.get_default_memory_pool().free_all_blocks()
    import gc
    gc.collect()
            
    python = sys.executable
    os.execv(python, [python] + sys.argv)
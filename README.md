
# 🧠 Brain Tumor State Detection and Life Expectancy Prediction with Explainable Multimodal AI

This project is a Bachelor of Science thesis developed at Eskişehir Technical University that introduces an **explainable AI-based multimodal system** to classify brain tumors and predict patient life expectancy. It combines **MRI images**, **histopathology slides**, and **clinical data** using a modular deep learning pipeline, and provides interpretability using **SHAP**. The final system is deployed via a web interface for real-world usability.

## 📌 Project Highlights

- **Multimodal architecture** using MRI, histopathology, and clinical data.
- **Explainability** via SHAP to enhance clinical trust.
- **Web interface** for real-time predictions.
- Based on 2021 WHO glioma classification.

---

## 📂 Folder Structure

```
Thesis/
├── backend/                # FastAPI backend for prediction
├── frontend/               # ReactJS-based frontend for uploading data
├── models/                 # Pretrained and final model weights
├── utils/                  # Preprocessing, postprocessing, and helpers
├── dataset/                # Preprocessed MRI, pathology, and metadata
└── README.md               # This file
```

---

## 📊 Technologies Used

- **Python** (FastAPI, PyTorch, NumPy, Pandas)
- **Deep Learning:** CNN (InceptionV3, VGG16), RNN (BiLSTM), Transformer
- **Libraries:** SHAP, SimpleITK, OpenSlide, scikit-learn
- **Frontend:** React.js
- **Deployment:** Uvicorn, Docker (optional)

---

## 🧬 Data Sources

- **MRI:** UPENN-GBM, TCGA-GBM, TCGA-LGG, CPTAC-GBM
- **Histopathology:** H&E stained slides (NDPI, DICOM)
- **Clinical Data:** Age, gender, tumor type, survival status

> ⚠️ **Note:** Access to some datasets required a TCIA Restricted License.

---

## 🧠 Model Architecture

- **MRI Module:**
  - Preprocessing: Skull stripping, resampling, bias correction
  - Feature extraction: InceptionV3 + Bidirectional LSTM
- **Histopathology Module:**
  - Patch extraction + clustering (K-Means)
  - Feature extraction: VGG16 → Transformer
- **Clinical Integration:**
  - Combined via MLP for final prediction
- **Explainability:**
  - SHAP values calculated per modality and feature

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Bashmaistro/Thesis.git
cd Thesis
```

### 2. Setup Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### 3. Setup Frontend
```bash
cd ../frontend
npm install
npm start
```

### 4. Upload Data
- Upload `.nii` MRI files, histopathology tiles, and patient info through the web interface.

---

## 🧪 Results Summary

| Task                         | Accuracy |
|------------------------------|----------|
| Tumor Grade Prediction       | 93.75%   |
| Mortality Prediction         | 96.88%   |
| Tumor Type Classification    | 93.94%   |
| Life Expectancy (macro avg) | 81.25%   |

- Histopathology contribution: **53.31%**
- MRI contribution: **44.89%**
- Clinical data contribution: **1.80%**

> 📊 Evaluated using F1-score, precision, recall, confusion matrices.

---

## 🌐 Website Integration

An intuitive web platform allows clinicians to:
- Upload patient MRI, pathology images, age & gender
- View predicted tumor type, grade, mortality, and life expectancy
- Access SHAP-based explainability results per patient

---

## 🎓 Authors

- Ahmet Caner Tat  
- **Emirhan Yıldız**  
- Salih Kızılışık  

**Advisor:** Assist. Prof. Dr. Sema Candemir

---

## 📄 Acknowledgments

This project was supported by the **TÜBİTAK 2209-A** Research Grant (1919B012466872). Special thanks to our advisor, Dr. Sema Candemir.

---

## 📃 License

This project is open-source under the MIT License. See `LICENSE` file for more information.

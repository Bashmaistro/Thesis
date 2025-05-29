from fastapi import FastAPI, File, UploadFile, Form,Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import shutil
import time
import sys
from predict import predict
from fastapi.staticfiles import StaticFiles


app = FastAPI()

# CORS ayarı (React frontend'den gelen istekleri kabul et)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Sadece React frontend'ine izin ver
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Upload klasörü
UPLOAD_DIR = "uploads"
os.makedirs(os.path.join(UPLOAD_DIR, "mri"), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_DIR, "histo"), exist_ok=True)

@app.post("/upload")
async def upload_files(
    mri: UploadFile = File(...),
    histo: UploadFile = File(...),
    age: int = Body(...),
    gender: str = Form(...)
):
    
    mri_file = mri
    histopathology_file = histo
    # MRI dosyasını kaydet
    mri_path = os.path.join(UPLOAD_DIR, "mri", mri_file.filename)
    with open(mri_path, "wb") as f:
        shutil.copyfileobj(mri_file.file, f)

    # Histopatoloji dosyalarını kaydet
    histo_path = os.path.join(UPLOAD_DIR, "histo", histopathology_file.filename)
    with open(histo_path, "wb") as f:
        shutil.copyfileobj(histopathology_file.file, f)

    # Giriş bilgilerini logla
    print(f"Age: {age}, Gender: {gender}")
    print(f"Saved MRI to: {mri_path}")
    print(f"Saved {histo_path} histopathology files.")
    
    if (gender == 'Male'):
        gender = 1
    else:
        gender = 0    

    result = predict(
    mri_path=mri_path,
    histo_path=histo_path,
    clinical_data=[gender, age]
        )
    
    print(result)

    return {
        "status": "success",
        "mri_saved": mri_file.filename,
        
        "age": age,
        "gender": gender,
    }



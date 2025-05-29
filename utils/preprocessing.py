#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 17:00:07 2025

@author: ai1
"""
import time
# import torch
import numpy as np
import pandas as pd
import dicom2nifti as d2n
from utils.utils import create_destroy_dic
import SimpleITK as sitk
import glob2
import matplotlib
import matplotlib.pyplot as plt
import os
import torch
import random

from utils.HD_BET.checkpoint_download import maybe_download_parameters
from utils.HD_BET.hd_bet_prediction import get_hdbet_predictor, hdbet_predict


def plot_operations_2_rows(img, denoised_img, corrected_img, resampled_img):
    """
    Plot denoised, bias-corrected, and resampled images in axial and sagittal views.
    Sagittal slices are flipped horizontally for correct orientation.
    """
    # Convert images to numpy arrays
    img_np = sitk.GetArrayFromImage(img)
    denoised_np = sitk.GetArrayFromImage(denoised_img)
    corrected_np = sitk.GetArrayFromImage(corrected_img)
    resampled_np = sitk.GetArrayFromImage(resampled_img)

    # Calculate the slice index in the resampled image that corresponds to the mid slice of the original image
    mid_z = img_np.shape[0] // 2
    mid_x = img_np.shape[1] // 2
    
    original_origin = img.GetOrigin()
    original_spacing = img.GetSpacing()
    
    resampled_origin = resampled_img.GetOrigin()
    resampled_spacing = resampled_img.GetSpacing()
    # Find corresponding index in resampled image
    mid_z_resampled = int((original_origin[2] + mid_z * original_spacing[2] - resampled_origin[2]) / resampled_spacing[2])
    mid_x_resampled = int((original_origin[0] + mid_x * original_spacing[0] - resampled_origin[0]) / original_spacing[0])
    # Axial slices
    axial_img = img_np[mid_z, :, :]
    axial_denoised = denoised_np[mid_z, :, :]
    axial_corrected = corrected_np[mid_z, :, :]
    axial_resampled = resampled_np[mid_z_resampled, :, :]

    # Sagittal slices (flipped for proper orientation)
    sagittal_img = np.flipud(img_np[:, :, mid_x])
    sagittal_denoised = np.flipud(denoised_np[:, :, mid_x])
    sagittal_corrected = np.flipud(corrected_np[:, :, mid_x])
    sagittal_resampled = np.flipud(resampled_np[:, :, mid_x_resampled])

    # Create a figure with 2 rows (axial on top, sagittal below)
    fig, axes = plt.subplots(2, 4, figsize=(16, 12))

    # Plot axial row
    axes[0, 0].imshow(axial_img, cmap='gray')
    axes[0, 0].set_title('Original (Axial)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(axial_denoised, cmap='gray')
    axes[0, 1].set_title('Denoised (Axial)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(axial_corrected, cmap='gray')
    axes[0, 2].set_title('Bias Field Corrected (Axial)')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(axial_resampled, cmap='gray')
    axes[0, 3].set_title('Resampled (Axial)')
    axes[0, 3].axis('off')

    # Plot sagittal row (flipped)
    axes[1, 0].imshow(sagittal_img, cmap='gray')
    axes[1, 0].set_title('Original (Sagittal)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(sagittal_denoised, cmap='gray')
    axes[1, 1].set_title('Denoised (Sagittal)')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(sagittal_corrected, cmap='gray')
    axes[1, 2].set_title('Bias Field Corrected (Sagittal)')
    axes[1, 2].axis('off')

    axes[1, 3].imshow(sagittal_resampled, cmap='gray')
    axes[1, 3].set_title('Resampled (Sagittal)')
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.show()
    
def plot_before_after_single_row(original, reoriented, cropped, resized):
    """
    Plot original, reoriented, cropped, and resized images in a single row.
    Each image is displayed with a title in the same figure.
    """
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    
    img_np = sitk.GetArrayFromImage(original)
    denoised_np = sitk.GetArrayFromImage(reoriented)
    corrected_np = sitk.GetArrayFromImage(cropped)
    resampled_np = sitk.GetArrayFromImage(resized)
    
    # Calculate the slice index in the resampled image that corresponds to the mid slice of the original image
    mid_z = img_np.shape[0] // 2
    mid_x = img_np.shape[1] // 2
    
    original_origin = original.GetOrigin()
    original_spacing = original.GetSpacing()
    
    reoriented_origin = reoriented.GetOrigin()
    reoriented_spacing = reoriented.GetSpacing()
    
    cropped_origin = cropped.GetOrigin()
    cropped_spacing = cropped.GetSpacing()
    
    resized_origin = resized.GetOrigin()
    resized_spacing = resized.GetSpacing()
    
    # Find corresponding index in resampled image
    mid_z_reoriented = int((original_origin[2] + mid_z * original_spacing[2] - reoriented_origin[2]) / reoriented_spacing[2])
    
    mid_z_cropped = int((original_origin[2] + mid_z * original_spacing[2] - cropped_origin[2]) / cropped_spacing[2])
    
    mid_z_resized = int((original_origin[2] + mid_z * original_spacing[2] - resized_origin[2]) / resized_spacing[2])


    # Axial slices
    # Plot Original Image
    ax[0].imshow(sitk.GetArrayFromImage(original)[mid_z, :, :], cmap='gray')
    ax[0].set_title("Original")
    ax[0].axis('off')

    # Plot Reoriented ImageS
    ax[1].imshow(sitk.GetArrayFromImage(reoriented)[mid_z_reoriented, :, :], cmap='gray')
    ax[1].set_title("Reoriented")
    ax[1].axis('off')

    # Plot Cropped Image
    ax[2].imshow(sitk.GetArrayFromImage(cropped)[mid_z_cropped, :, :], cmap='gray')
    ax[2].set_title("Cropped")
    ax[2].axis('off')

    # Plot Resized Image
    ax[3].imshow(sitk.GetArrayFromImage(resized)[mid_z_resized, :, :], cmap='gray')
    ax[3].set_title("Resized")
    ax[3].axis('off')

    plt.tight_layout()
    plt.show()
    
def bias_field_correction(img, mask):
    img = sitk.Cast(img, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50])
    return corrector.Execute(img, mask)

def denoise_image(img):
    denoiser = sitk.CurvatureFlowImageFilter()
    denoiser.SetTimeStep(0.0625)
    denoiser.SetNumberOfIterations(3)
    return denoiser.Execute(img)

def resample_spacing(img, new_spacing=(1.0, 1.0, 3.0), interpolator=sitk.sitkBSpline):
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()

    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)

    return resampler.Execute(img)


def generate_brain_mask(img: sitk.Image, visualization=0, threshold=0):
    """
    Generate a binary brain mask from a skull-stripped image.
    
    Parameters:
    - img: Skull-stripped image (SimpleITK Image)
    - threshold: Intensity threshold to separate brain from background
    
    Returns:
    - Binary mask (SimpleITK Image) with same size and spacing
    """
    # # Get image statistics
    # stats = sitk.StatisticsImageFilter()
    # stats.Execute(img)
    # min_intensity = stats.GetMinimum()
    # max_intensity = stats.GetMaximum()
    
    # # Stretching: scale the intensity values to the full range
    # new_img = sitk.IntensityWindowing(img, windowMinimum=min_intensity, windowMaximum=max_intensity, 
    #                                   outputMinimum=0, outputMaximum=255)

    # img_np = sitk.GetArrayFromImage(new_img)
    # avg_intensity = int(np.mean(img_np)) + 15
    # if visualization:
    #     # Plot histogram
    #     plt.hist(img_np.flatten(), bins=100)
    #     plt.title("Image Intensity Histogram")
    #     plt.xlabel("Intensity")
    #     plt.ylabel("Voxel Count")
    #     plt.show()
    #     print(avg_intensity)
    #     # Apply simple thresholding to get brain vs background
    # mask = sitk.BinaryThreshold(new_img, lowerThreshold=avg_intensity, upperThreshold=sitk.GetArrayViewFromImage(img).max(), insideValue=1, outsideValue=0)

    # Optional: clean up noise
    
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    
    mask = sitk.BinaryMorphologicalClosing(mask, [2]*3)  # fill small holes
    mask = sitk.BinaryFillhole(mask)

    cc = sitk.ConnectedComponent(mask)

    # Get statistics to find the largest component
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    largest_label = max(stats.GetLabels(), key=lambda l: stats.GetPhysicalSize(l))
    
    # Threshold to keep only the largest label
    mask = sitk.BinaryThreshold(cc, 
                                      lowerThreshold=largest_label, 
                                      upperThreshold=largest_label, 
                                      insideValue=1, 
                                      outsideValue=0)
    if visualization:
        mask_np = sitk.GetArrayFromImage(mask)
        resampled_z = mask_np.shape[0] // 2
        plt.imshow(mask_np[resampled_z,:,:])
        plt.savefig("mask.jpg")
        plt.show()

    return mask    

def resample_mask(mask_img, reference_img, transform):
    """
    Resample the mask to align with the reference image using the given transform.
    """
    return sitk.Resample(mask_img,
                         reference_img,
                         transform,
                         sitk.sitkNearestNeighbor,  # use nearest neighbor to preserve binary mask
                         0,
                         mask_img.GetPixelID())
    
def crop_to_mask(img, mask):
    """
    Crop the image and mask to the bounding box of the brain mask.
    """
    mask_array = sitk.GetArrayFromImage(mask)
    indices = np.array(np.nonzero(mask_array))
    
    min_indices = indices.min(axis=1)
    max_indices = indices.max(axis=1) + 1  # +1 to include last index

    cropped_img = img[min_indices[2]:max_indices[2],
                      min_indices[1]:max_indices[1],
                      min_indices[0]:max_indices[0]]

    cropped_mask = mask[min_indices[2]:max_indices[2],
                        min_indices[1]:max_indices[1],
                        min_indices[0]:max_indices[0]]
    
    return cropped_img, cropped_mask

def resize_image(image, new_size=(256, 256, 52), interpolator=sitk.sitkBSpline):
    """
    Resample image to new size with consistent physical spacing.
    """
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()

    new_size = np.array(new_size, dtype=np.int64)
    new_spacing = [
        (original_size[i] * original_spacing[i]) / new_size[i]
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([int(s) for s in new_size])
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)

    return resampler.Execute(image)


def rigid_register_to_template(moving_img, fixed_img):
    """
    Apply rigid registration to align moving image (e.g., patient) to fixed image (e.g., MNI).
    Returns both aligned image and transform.
    """
    # Ensure that both images are 3D
    if moving_img.GetDimension() != 3 or fixed_img.GetDimension() != 3:
        raise ValueError("Both images must be 3D images.")

    # Ensure the images have the same pixel type (e.g., both float32)
    if moving_img.GetPixelID() != fixed_img.GetPixelID():
        moving_img = sitk.Cast(moving_img, fixed_img.GetPixelID())
    
    registration = sitk.ImageRegistrationMethod()
    
    # Set Metric: Mattes Mutual Information
    registration.SetMetricAsMattesMutualInformation()
    
    # Set Optimizer: Regular Step Gradient Descent
    registration.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0,
                                                          minStep=1e-4,
                                                          numberOfIterations=100,
                                                          gradientMagnitudeTolerance=1e-8)
    
    # Set Interpolator: BSpline
    registration.SetInterpolator(sitk.sitkBSpline)
    
    # Initialize the transform (Euler 3D for rigid registration)
    initial_transform = sitk.Euler3DTransform()
    
    # Centered Transform Initializer
    registration.SetInitialTransform(sitk.CenteredTransformInitializer(
        fixed_img, moving_img, initial_transform, sitk.CenteredTransformInitializerFilter.GEOMETRY))

    # Perform the registration
    final_transform = registration.Execute(fixed_img, moving_img)
    
    # Apply the final transform to the moving image
    aligned_img = sitk.Resample(moving_img, fixed_img, final_transform, sitk.sitkLinear, 0.0, moving_img.GetPixelID())

    return aligned_img, final_transform


# def nyul_standardize_array(np_img, reference_paths, spacing=(1.0, 1.0, 1.0), mask_method='otsu'):
#     """
#     Standardize an MRI array using Nyul histogram standardization.
    
#     Args:
#         np_img (np.ndarray): Input image (3D array) from SimpleITK.
#         reference_paths (list): Paths to reference .nii.gz files to build Nyul histogram landmarks.
#         spacing (tuple): Voxel spacing of the input image (default is 1.0 mm isotropic).
#         mask_method (str): Masking method for brain region ('otsu' is recommended).
        
#     Returns:
#         np.ndarray: Histogram standardized image.
#     """
#     # Step 1: Build the transform using reference scans
#     subjects = [tio.Subject(t1=tio.ScalarImage(p)) for p in reference_paths]
#     dataset = tio.SubjectsDataset(subjects)

#     hist_std = tio.HistogramStandardization(
#         landmarks_dict={}, 
#         masking_method=mask_method
#     )
#     hist_std.fit(dataset)

#     # Step 2: Convert the NumPy array to a temporary NIfTI file for TorchIO
#     with tempfile.TemporaryDirectory() as tmpdir:
#         temp_path = os.path.join(tmpdir, 'temp.nii.gz')
#         affine = np.diag(list(spacing) + [1])  # Build affine using spacing
#         nib_img = nib.Nifti1Image(np_img.astype(np.float32), affine)
#         nib.save(nib_img, temp_path)

#         # Step 3: Apply Nyul transformation
#         tio_img = tio.ScalarImage(temp_path)
#         standardized = hist_std(tio_img)

#         # Step 4: Return back as NumPy array
#         return standardized.t1.numpy().squeeze()

def bet(inp, conf, override=0):
    #Load model once
    
    maybe_download_parameters()
    predictor = get_hdbet_predictor(
        use_tta=not conf["disable_tta"],
        device=torch.device(conf["device"]),
        verbose=conf["verbose"]
    )
    
    img_list = glob2.glob(inp + "/*/*-operated.nii.gz")
    
    for img in img_list:
        if not override and any("processed-bet.nii.gz" in f for f in os.listdir(os.path.dirname(img))):
            continue
        else:
            out = os.path.dirname(img)
            out = out + "/processed-bet.nii.gz"
            print(img)
            print(out)
            hdbet_predict(img, out, predictor, keep_brain_mask=conf["save_bet_mask"],
                    compute_brain_extracted_image=not conf["no_bet_image"])
    return

def plot_full_preprocessing_pipeline(
    original_img,
    corrected_img,
    resampled_img,
    extracted_img,
    matched_img,
    denoised_img,
    cropped_img,
    resized_img
):
    """
    Visualize each major step of the MRI preprocessing pipeline in a 2x4 layout.
    All images are shown in the axial plane using the middle slice.
    """
    
    def get_mid_axial_slice(img):
        try:
            np_img = sitk.GetArrayFromImage(img)
            if np_img.ndim != 3:
                raise ValueError(f"Expected 3D image, got shape: {np_img.shape}")
            mid_z = np_img.shape[0] // 2
            print(f"Image shape: {np_img.shape}, mid_z: {mid_z}")
            return np_img[mid_z, :, :]
        except Exception as e:
            print(f"Error extracting slice: {e}")
            return np.zeros((10, 10))  # fallback to show empty image

    print(">>> Starting plot_full_preprocessing_pipeline")

    images = [
        (original_img, "Original"),
        (corrected_img, "Bias Field Corrected"),
        (resampled_img, "Resampled"),
        (extracted_img, "BET Extracted"),
        (matched_img, "Histogram Matched"),
        (denoised_img, "Denoised"),
        (cropped_img, "Cropped"),
        (resized_img, "Resized")
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for i, (img, title) in enumerate(images):
        row = i // 4
        col = i % 4

        if not isinstance(img, sitk.Image):
            print(f"WARNING: '{title}' is not a SimpleITK image. Skipping.")
            axes[row, col].imshow(np.zeros((10, 10)), cmap='gray')
            axes[row, col].set_title(f"{title} (Invalid)")
            axes[row, col].axis('off')
            continue

        slice_img = get_mid_axial_slice(img)
        axes[row, col].imshow(slice_img, cmap='gray')
        axes[row, col].set_title(title, fontsize=18)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig("debug_preprocessing_pipeline.png", dpi=600)
    plt.close()

    
    
class preprocess_mri_file():
    
    def __init__(self, config):
        self.config = config
        
        img = sitk.ReadImage("utils/templates/4_c_ax_t1_mp_spgr.nii-operated.nii.gz")
        reference_img = sitk.Cast(img, sitk.sitkFloat32)
        self.reference_image = reference_img
        self.visualization = False
        self.resample_size = 3.0
        self.pred_folder = "test/pred"
        
    def __call__(self, inp:str):
        
        if os.path.isdir(inp):
            print(f"'{inp}' is a directory. Processing NIfTI files in folder...")
            img_dir = self.dicom2nii(inp, compression= True, reorient=True)
            img = sitk.ReadImage(img_dir)

            img_dir = os.path.dirname(inp)
            img_dir = self.pred_folder
            create_destroy_dic(img_dir)
            print(img_dir)
            
        elif inp.endswith(('.nii', '.nii.gz')):
            print(f"'{inp}' is a NIfTI file. Processing file...")
            img = sitk.ReadImage(inp)
            img_dir = os.path.dirname(inp)
            img_dir = self.pred_folder
            create_destroy_dic(img_dir)
            print(img_dir)
        else:
            raise ValueError(f"Invalid input path: '{inp}'. Must be a NIfTI file or folder containing NIfTI files.")
        
        denoised = denoise_image(img)
        mask = generate_brain_mask(img, visualization=self.visualization)   
        corrected = bias_field_correction(denoised, mask)

        spacing = corrected.GetSpacing()
        resampled = resample_spacing(corrected, new_spacing=(spacing[0], spacing[1], self.resample_size))
        
        operated_dir = img_dir +"/operated.nii.gz"
        
        print(img_dir)
        sitk.WriteImage(resampled, operated_dir)
        print("A file is created as  " + operated_dir)
        
        out, mask = self.bet(img_dir, self.config.get("bet_conf"))
        
        extracted = sitk.ReadImage(out)
        mask = sitk.ReadImage(mask)
        
        matched = sitk.HistogramMatching(extracted, self.reference_image, numberOfHistogramLevels=50, numberOfMatchPoints=10)
        
        denoised = denoise_image(matched)
        # Step 4: Crop to Mask
        cropped, mask = crop_to_mask(denoised, mask)
        
        # Step 5: Resize Image
        resized = resize_image(cropped)
        ready_dir = img_dir +"/ready.nii.gz"
        sitk.WriteImage(resized, ready_dir)
        
        plot_full_preprocessing_pipeline(
        original_img=img,
        corrected_img=corrected,
        resampled_img=resampled,
        extracted_img=extracted,
        matched_img=matched,
        denoised_img=denoised,
        cropped_img=cropped,
        resized_img=resized
    )
        img = sitk.GetArrayFromImage(resized)
        return img



    def dicom2nii(self, inp_dir, compression=True, reorient=True):
        
        files = os.listdir(inp_dir)
        nifti_files = [f for f in files if f.endswith('.nii') or f.endswith('.nii.gz')]
        if len(nifti_files) > 0:
            for f in nifti_files:
                file_path = os.path.join(inp_dir, f)  # use out_folder here, not inp_dir
                try:
                    os.remove(file_path)
                    print(f"Deleted existing file: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

        print(inp_dir)
        out_folder = inp_dir
        d2n.convert_directory(inp_dir, out_folder, compression, reorient)
        
        files = os.listdir(out_folder)

        # To guess the output filename:
        nifti_files = [f for f in files if f.endswith('.nii') or f.endswith('.nii.gz')]
        if not len(nifti_files) == 0:
            return os.path.join(out_folder, nifti_files[0])
        else:
            return None  # or raise error if preferred
        
        
    def bet(self, inp, conf, override=0):
        maybe_download_parameters()
        predictor = get_hdbet_predictor(
            use_tta=not conf["disable_tta"],
            device=torch.device(conf["device"]),
            verbose=conf["verbose"]
        )
        
        files = os.listdir(inp)
        
        nifti_files = [f for f in files if f.endswith('operated.nii') or f.endswith('operated.nii.gz')]
        
        img = inp + "/" + nifti_files[0]
        out = inp + "/extracted.nii.gz"
        mask = inp + "/extracted_bet.nii.gz"
        print(img)
        print(out)
        hdbet_predict(img, out, predictor, keep_brain_mask=conf["save_bet_mask"],
                compute_brain_extracted_image=not conf["no_bet_image"])
        return out, mask



















class Preprocess_3D():
    
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset["dataset"]
        self.out_dir = self.config.get("paths")["dataset_cvt"]
        self.ready = self.config.get("paths")["ready"]
        self.reference_image = sitk.ReadImage("dataset/converted/UPENN-GBM-00609/processed-bet.nii.gz")

        return
    
    def __call__(self,):
        run_conf = self.config.get("preprocess")
        
        if run_conf["d2n"]:
            self.dicom2nii(self.dataset, self.out_dir)
            
        if run_conf["filter_ops"]:
            self.filter_ops(denoising=run_conf["denoising"],
                            bfc_flag=run_conf["bias_field_correction"],
                            resample=run_conf["resample"],
                            resample_size=run_conf["resample_size"],
                            override = run_conf["override"],
                            )
            
        if run_conf["bet_run"]:
            bet(self.out_dir, run_conf["bet"], run_conf["override"])
            
        if run_conf["rrr"]:
            self.reorient_crop_resize()
                
        return
        
    def dicom2nii(self, inp_dir, out_dir, compression=False, reorient=True):
        print(inp_dir)
        df = pd.read_csv(inp_dir +"/../result.csv")
        create_destroy_dic(out_dir)
        for row in df.itertuples(index=True):
            
            out_folder = out_dir + row.id 
            create_destroy_dic(out_folder)
            print(row.CT_dir[1:])
            d2n.convert_directory(row.CT_dir[1:], out_folder, compression, reorient)
            print(f"Index: {row.Index}, {row.id}")
        return
        
    def filter_ops(self, denoising=1, bfc_flag=1, resample=1, resample_size= 3.0, override= 1, visualization=0):
        img_list = glob2.glob(self.out_dir + "/*/*.nii")
        
        for img_path in img_list:
           if "operated.nii" not in img_path:
               if not override and any("operated.nii" in f for f in os.listdir(os.path.dirname(img_path))):
                   continue
               else:
                    try:
                        img = sitk.ReadImage(img_path)
                        print(f"Successfully read: {img_path}")
                    except RuntimeError as e:
                        print(f"Failed to read: {img_path}")
                        print(f"Error: {e}")
                        continue  # Skip to next image on read failure
                    temp = img
                    
                    mask = generate_brain_mask(img, visualization=visualization)   
                    if denoising:
                        img = denoise_image(img)
                        temp1 = img
        
                
                    if bfc_flag:
                        # ct = sitk.HistogramMatching(ct, self.reference_image, numberOfHistogramLevels=50, numberOfMatchPoints=10)
                        img = bias_field_correction(img,mask)
                        temp2 = img
        
                        
                    if resample:
                        spacing = img.GetSpacing()
                        img = resample_spacing(img, new_spacing=(spacing[0], spacing[1], resample_size))
                        temp3 = img

                    if visualization:
                        plot_operations_2_rows(temp, temp1, temp2, temp3)
                        
                    else:
                        del temp, temp1, temp2, temp3
                        
                        
                    out_path = img_path + "-operated.nii.gz"
                    mask_path = img_path + "-mask.nii.gz"
                    
                    sitk.WriteImage(img, out_path)
                    sitk.WriteImage(mask, mask_path)
        return
    


    
    def reorient_crop_resize(self,  override=1, template="utils/templates/processed-bet.nii.gz"):
        temp_img = sitk.ReadImage(template)
            
        img_list = glob2.glob(self.out_dir + "/*/processed-bet.nii.gz")
        ready_path =self.ready
        if override:
            create_destroy_dic(ready_path)
        
        for img_path in img_list:
            file_name = img_path.split("/")[-2]
            file_path= ready_path +  file_name + ".npy"
            if not override and any(file_name + ".nii.gz" in f for f in os.listdir(ready_path)):
                continue
            else:
                print(img_path)
                # Step 1: Read Image and Mask
                img = sitk.ReadImage(img_path)
                mask = sitk.ReadImage(img_path)
                temp = img

                # # Step 2: Rigid Registration
                # img, transform = rigid_register_to_template(img, temp_img)
                # temp1 = img
                # # Step 3: Resample the Mask
                # mask = resample_mask(mask, temp_img, transform)
                img = sitk.HistogramMatching(img, self.reference_image, numberOfHistogramLevels=50, numberOfMatchPoints=10)

                # Step 4: Crop to Mask
                img, mask = crop_to_mask(img, mask)
                temp2 = img
                # Step 5: Resize Image
                img = resize_image(img)
                temp3 = img
                # Now plot the images (Original, Reoriented, Cropped, Resized) in one row
                plot_before_after_single_row(temp, temp, temp2, temp3)
                

                np.save(file_path, sitk.GetArrayFromImage(img))

                
        return
        

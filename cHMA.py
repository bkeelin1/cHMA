# Canonical Holistic Morphometric Analysis (cHMA)
# Author: Brian Anthony Keeling

import os
import numpy as np
import SimpleITK as sitk
import logging
import gc
import traceback
import time
import multiprocessing
import pandas as pd
import math
import skimage.measure
import pyvista as pv
import skimage.measure
import vtk
import re
import tetgen
from vtk.util.numpy_support import numpy_to_vtk
from scipy.spatial.transform import Rotation as R
from datetime import datetime
from skimage import measure
from scipy import ndimage
from skimage import measure, morphology, filters
from vtk.util import numpy_support
from scipy.ndimage import binary_closing, binary_opening, binary_dilation, binary_erosion, generate_binary_structure, map_coordinates
from scipy.spatial import Delaunay, cKDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_robust_outer_hull(bone_mask):
    """
    Uses orthogonal 2D slice-by-slice hole filling to perfectly seal
    open-ended bones (like condyle necks) without losing 3D concavities.
    """
    arr = sitk.GetArrayFromImage(bone_mask).astype(bool)

    # 1. Fill holes slice-by-slice along Z
    fill_z = np.zeros_like(arr)
    for z in range(arr.shape[0]):
        fill_z[z, :, :] = ndimage.binary_fill_holes(arr[z, :, :])

    # 2. Fill holes slice-by-slice along Y
    fill_y = np.zeros_like(arr)
    for y in range(arr.shape[1]):
        fill_y[:, y, :] = ndimage.binary_fill_holes(arr[:, y, :])

    # 3. Fill holes slice-by-slice along X
    fill_x = np.zeros_like(arr)
    for x in range(arr.shape[2]):
        fill_x[:, :, x] = ndimage.binary_fill_holes(arr[:, :, x])

    # The true solid hull is the intersection of all three orthogonal fills
    hull_arr = fill_z & fill_y & fill_x

    hull_img = sitk.GetImageFromArray(hull_arr.astype(np.uint8))
    hull_img.CopyInformation(bone_mask)
    return hull_img


def slicer_surface_wrap_solidify(bone_mask, wrap_radius=6):
    """
    A pure Python implementation of 3D Slicer's 'Surface Wrap Solidify' extension.
    It seals open cut planes (like a condyle neck) and micro-cracks by expanding,
    isolating the scanner background, inverting, and shrinking back.
    """
    #logging.info(f"  -> Slicer Wrap: Dilating by {wrap_radius} to seal surface leaks...")
    dilated_bone = sitk.BinaryDilate(bone_mask, [wrap_radius, wrap_radius, wrap_radius], sitk.sitkBall)

    #logging.info("  -> Slicer Wrap: Isolating the scanner background...")
    # Everything that is 0 is air (both inside and outside)
    air_mask = dilated_bone == 0
    cc_air = sitk.ConnectedComponent(air_mask)
    stats_air = sitk.LabelShapeStatisticsImageFilter()
    stats_air.Execute(cc_air)

    # The largest pocket of air is the infinite scanner background
    largest_air_label = max(stats_air.GetLabels(), key=lambda l: stats_air.GetPhysicalSize(l))
    outside_air = sitk.BinaryThreshold(
        cc_air, lowerThreshold=largest_air_label,
        upperThreshold=largest_air_label, insideValue=1, outsideValue=0
    )

    #logging.info("  -> Slicer Wrap: Inverting background and eroding back to anatomical boundary...")
    # Everything NOT outside air is the solid condyle
    solid_dilated = outside_air == 0

    # Shrink it back to exactly fit the original bone
    solid_condyle = sitk.BinaryErode(solid_dilated, [wrap_radius, wrap_radius, wrap_radius], sitk.sitkBall)
    return solid_condyle


def Segment(input_dir, output_dir):
    """Automated pipeline strictly following the 10-Step Slicer Workflow."""

    log_file = os.path.join(output_dir, f"Segmentation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

    logging.info(f"Starting 10-Step Segmentation pipeline. Input: {input_dir} | Output: {output_dir}")

    dirs = {
        'Raw': os.path.join(output_dir, 'Raw'),
        'Filled': os.path.join(output_dir, 'Filled'),
        'Cortical': os.path.join(output_dir, 'Cortical'),
        'Trabecular': os.path.join(output_dir, 'Trabecular'),
        'Debug': os.path.join(output_dir, 'Debug')
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    valid_extensions = ('.tiff', '.tif')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

    if not files:
        logging.error(f"No files found in {input_dir}.")
        return False

    for filename in files:
        bone_id = os.path.splitext(filename)[0]
        input_path = os.path.join(input_dir, filename)

        try:
            logging.info(f"--- Processing {bone_id} ---")
            raw_image = load_image(input_path)
            sitk.WriteImage(raw_image, os.path.join(dirs['Raw'], f"{bone_id}.tiff"))

            # --- STEP 1: THRESHOLD THE BONE ---
            np_view = sitk.GetArrayViewFromImage(raw_image)
            if len(np.unique(np_view)) <= 256:
                img_min, img_max = np.min(np_view), np.max(np_view)
                bone_mask = sitk.Cast(raw_image > float(img_min + 0.5 * (img_max - img_min)), sitk.sitkUInt8)
            else:
                logging.info(f"Applying Triangle Thresholding...")
                sharpened = sitk.Cast(raw_image, sitk.sitkFloat32)
                sharpened = sitk.LaplacianSharpening(sitk.LaplacianSharpening(sharpened))
                triangle = sitk.TriangleThresholdImageFilter()
                triangle.SetInsideValue(0)
                triangle.SetOutsideValue(1)
                bone_mask = triangle.Execute(sharpened)

            # (Housekeeping: Clean main bone islands)
            cc = sitk.ConnectedComponent(bone_mask)
            stats = sitk.LabelShapeStatisticsImageFilter()
            stats.Execute(cc)
            largest_label = max(stats.GetLabels(), key=lambda l: stats.GetPhysicalSize(l))
            clean_bone_mask = sitk.BinaryThreshold(cc,
                                                   lowerThreshold=largest_label, upperThreshold=largest_label,
                                                   insideValue=1, outsideValue=0)

            sitk.WriteImage(sitk.Cast(clean_bone_mask * 255, sitk.sitkUInt8),
                            os.path.join(dirs['Filled'], f"{bone_id}.tiff"))

            # --- STEP 2: FILL BONE (SurfaceWrapSolidify) ---
            logging.info("STEP 1: Executing Surface Wrap Solidify on the external shell...")
            solid_condyle = slicer_surface_wrap_solidify(clean_bone_mask, wrap_radius=6)

            # --- STEP 3: SUBTRACT TO GET PORES ---
            logging.info("STEP 2: Subtracting bone to isolate pores...")
            raw_pores = solid_condyle & ~clean_bone_mask

            # --- STEP 4: REMOVE SMALL ISLANDS (500 VOXELS) ---
            logging.info("STEP 3: Removing small pore islands (< 500 voxels)...")
            pores_array = sitk.GetArrayFromImage(raw_pores).astype(bool)
            cleaned_pores_array = morphology.remove_small_objects(pores_array, min_size=500)
            cleaned_pores = sitk.GetImageFromArray(cleaned_pores_array.astype(np.uint8))
            cleaned_pores.CopyInformation(raw_pores)

            # --- STEP 5: SURFACE WRAP SOLIDIFY THE PORES ---
            logging.info("STEP 4: Executing Surface Wrap Solidify on the pore network...")
            # We use a radius of 8 here to cleanly bridge the thick trabecular struts
            medullary_cavity = slicer_surface_wrap_solidify(cleaned_pores, wrap_radius=8)

            # --- STEP 6: MEDIAN SMOOTHING (1, 1, 1) ---
            logging.info("STEP 5: Applying median smoothing to the Medullary Cavity...")
            median_filter = sitk.MedianImageFilter()
            median_filter.SetRadius([1, 1, 1])
            medullary_cavity = median_filter.Execute(medullary_cavity)

            # Safety check: Cavity cannot exceed the solid outer bone
            medullary_cavity = medullary_cavity & solid_condyle

            # --- STEP 7: EXTRACT TRABECULAR BONE ---
            logging.info("STEP 6: Intersecting to extract Trabecular Bone...")
            trabecular_mask = clean_bone_mask & medullary_cavity
            trabecular_array = sitk.GetArrayFromImage(trabecular_mask).astype(bool)
            clean_trabecular_array = morphology.remove_small_objects(trabecular_array, min_size=100)
            trabecular_mask_clean = sitk.GetImageFromArray(clean_trabecular_array.astype(np.uint8))
            trabecular_mask_clean.CopyInformation(trabecular_mask)
            trabecular_mask = trabecular_mask_clean

            # --- STEP 8: EXTRACT CORTICAL BONE ---
            logging.info("STEP 7: Subtracting to extract Cortical Bone...")
            cortical_mask = clean_bone_mask & ~medullary_cavity
            cort_array = sitk.GetArrayFromImage(cortical_mask).astype(bool)
            clean_cort_array = morphology.remove_small_objects(cort_array, min_size=500)
            cortical_mask_clean = sitk.GetImageFromArray(clean_cort_array.astype(np.uint8))
            cortical_mask_clean.CopyInformation(cortical_mask)
            cortical_mask = cortical_mask_clean

            # --- STEP 10: SAVE SEGMENTATIONS ---
            logging.info("STEP 8: Saving Cortical and Trabecular outputs...")
            sitk.WriteImage(sitk.Cast(cortical_mask * 255, sitk.sitkUInt8),
                            os.path.join(dirs['Cortical'], f"{bone_id}.tiff"))
            sitk.WriteImage(sitk.Cast(trabecular_mask * 255, sitk.sitkUInt8),
                            os.path.join(dirs['Trabecular'], f"{bone_id}.tiff"))

            # Cleanup
            del clean_bone_mask, solid_condyle, raw_pores, pores_array
            del cleaned_pores_array, cleaned_pores, medullary_cavity
            del trabecular_mask, cortical_mask, cortical_mask_clean
            gc.collect()

        except Exception as e:
            logging.error(f"Error processing {bone_id}: {e}")
            traceback.print_exc()
            continue

    logging.info("Segmentation pipeline completed.")
    return True

def Segment2(input_dir, output_dir, erosion_radius=1):
    """
    Automated pipeline to process raw image stacks, apply double Laplacian
    sharpening, isolate the main bone structure, apply light smoothing, and
    perform an Erosion-Bounded Shrinkwrap to cleanly separate cortical from
    trabecular bone in complex geometries like the mandibular condyle.
    """

    log_file = os.path.join(output_dir, f"Segmentation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Starting Segmentation pipeline. Input: {input_dir} | Output: {output_dir}")

    dirs = {
        'Raw': os.path.join(output_dir, 'Raw'),
        'Filled': os.path.join(output_dir, 'Filled'),
        'Cortical': os.path.join(output_dir, 'Cortical'),
        'Trabecular': os.path.join(output_dir, 'Trabecular')
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    valid_extensions = ('.tiff', '.tif')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

    if not files:
        logging.error(f"No .tiff or .tif files found in {input_dir}.")
        return False

    for filename in files:
        bone_id = os.path.splitext(filename)[0]
        input_path = os.path.join(input_dir, filename)

        try:
            logging.info(f"--- Processing {bone_id} ---")

            # --- LOAD & SAVE RAW ---
            raw_image = load_image(input_path)
            if raw_image is None:
                logging.error(f"Failed to load {input_path}")
                continue

            raw_out_path = os.path.join(dirs['Raw'], f"{bone_id}.tiff")
            sitk.WriteImage(raw_image, raw_out_path)

            # --- CHECK IF ALREADY BINARY ---
            np_view = sitk.GetArrayViewFromImage(raw_image)
            unique_vals = np.unique(np_view)
            is_binary = len(unique_vals) <= 256

            if is_binary:
                logging.info(f"Image {bone_id} is binary. Handling Binary Input.")
                img_min = np.min(np_view)
                img_max = np.max(np_view)
                threshold_val = img_min + 0.5 * (img_max - img_min)

                bone_mask = raw_image > float(threshold_val)
                bone_mask = sitk.Cast(bone_mask, sitk.sitkUInt8)

                #bone_mask = raw_image > 0
                #bone_mask = sitk.Cast(bone_mask, sitk.sitkUInt16)

            else:
                logging.info(f"Image {bone_id} is Non-binary. Handling Non-Binary Input.")
                sharpened_img = sitk.Cast(raw_image, sitk.sitkFloat32)
                sharpened_img = sitk.Cast(sharpened_img, raw_image.GetPixelID())
                sharpened_img = sitk.LaplacianSharpening(sharpened_img)
                sharpened_img = sitk.LaplacianSharpening(sharpened_img)
                triangle_filter = sitk.TriangleThresholdImageFilter()
                triangle_filter.SetInsideValue(0)
                triangle_filter.SetOutsideValue(1)
                bone_mask = triangle_filter.Execute(sharpened_img)

                del sharpened_img

            # --- PROCESS SMALL ISLANDS ---
            logging.info("Cleaning artifacts outside the main bone structure...")
            cc = sitk.ConnectedComponent(bone_mask)
            stats = sitk.LabelShapeStatisticsImageFilter()
            stats.Execute(cc)

            labels = stats.GetLabels()
            if not labels:
                logging.warning(f"No bone detected in {bone_id}. Skipping.")
                continue

            largest_label = max(labels, key=lambda l: stats.GetPhysicalSize(l))
            clean_bone_mask = sitk.BinaryThreshold(
                cc, lowerThreshold=largest_label,
                upperThreshold=largest_label,
                insideValue=1, outsideValue=0
            )
            del raw_image, bone_mask, cc, stats

            # --- VERY LIGHT SMOOTHING ---
            logging.info("Applying very light smoothing (Median Radius=1) to the cleaned bone mask...")
            median_filter = sitk.MedianImageFilter()
            median_filter.SetRadius([1, 1, 1])
            clean_bone_mask = median_filter.Execute(clean_bone_mask)

            # --- SAVE FILLED ---
            filled_safe = sitk.Cast(clean_bone_mask * 255, sitk.sitkUInt8)
            filled_out_path = os.path.join(dirs['Filled'], f"{bone_id}.tiff")
            sitk.WriteImage(filled_safe, filled_out_path)

            # --- EROSION-BOUNDED SHRINKWRAP ---
            logging.info("Identifying the boundaries of internal trabecular bone...")

            # 1. Build the absolute outer hull using the robust Orthogonal Fill
            closed_bone = sitk.BinaryMorphologicalClosing(clean_bone_mask, [5, 5, 5], sitk.sitkBall)
            outer_hull = generate_robust_outer_hull(closed_bone)

            # 2. Create the protective firewall
            internal_hull = sitk.BinaryErode(outer_hull, [erosion_radius, erosion_radius, erosion_radius], sitk.sitkBall)

            # 3. Find pores STRICTLY inside the protective internal hull
            internal_pores = internal_hull & ~clean_bone_mask

            # 4. Clean up the internal pores
            closed_pores = sitk.BinaryMorphologicalClosing(internal_pores, [3, 3, 3], sitk.sitkBall)
            filled_pores = sitk.BinaryFillhole(closed_pores, fullyConnected=True)

            pores_array = sitk.GetArrayFromImage(filled_pores).astype(bool)
            cleaned_pores_array = morphology.remove_small_objects(pores_array, min_size=500)
            cleaned_pores = sitk.GetImageFromArray(cleaned_pores_array.astype(np.uint8))
            cleaned_pores.CopyInformation(filled_pores)

            logging.info("Segmenting the trabecular bone...")

            # 5. Build the final medullary cavity
            medullary_base = cleaned_pores > 0
            medullary_closed = sitk.BinaryMorphologicalClosing(medullary_base, [10, 10, 10], sitk.sitkBall)
            medullary_cavity = sitk.BinaryFillhole(medullary_closed, fullyConnected=True)

            # Intersect with internal_hull to guarantee it never touches the outer cortical plate
            medullary_cavity = medullary_cavity & internal_hull

            cavity_safe = sitk.Cast(medullary_cavity * 255, sitk.sitkUInt8)
            sitk.WriteImage(cavity_safe, os.path.join("B:/", f"{bone_id}_02_Medullary_Cavity.tiff"))

            #medullary_cavity = internal_hull

            # 6. Extract the final structures
            trabecular_mask = clean_bone_mask & medullary_cavity
            cortical_mask = clean_bone_mask & ~medullary_cavity

            logging.info("Saving all segmentations...")

            # --- SAVE CORTICAL AND TRABECULAR ---
            cortical_safe = sitk.Cast(cortical_mask * 255, sitk.sitkUInt8)
            trabecular_safe = sitk.Cast(trabecular_mask * 255, sitk.sitkUInt8)

            cortical_out_path = os.path.join(dirs['Cortical'], f"{bone_id}.tiff")
            trabecular_out_path = os.path.join(dirs['Trabecular'], f"{bone_id}.tiff")

            sitk.WriteImage(cortical_safe, cortical_out_path)
            sitk.WriteImage(trabecular_safe, trabecular_out_path)

            logging.info(f"Saved Cortical bone to {cortical_out_path}")
            logging.info(f"Saved Trabecular bone to {trabecular_out_path}")

            # Cleanup
            del clean_bone_mask, closed_bone, outer_hull, internal_hull
            del internal_pores, closed_pores, filled_pores, pores_array
            del cleaned_pores_array, cleaned_pores, medullary_base, medullary_closed
            del medullary_cavity, trabecular_mask, cortical_mask
            del filled_safe, cortical_safe, trabecular_safe
            gc.collect()

        except Exception as e:
            logging.error(f"Error processing {bone_id}: {e}")
            traceback.print_exc()
            continue

    logging.info("Segmentation pipeline completed.")
    return True

###############################
# Resampling Function
###############################

def load_image1(file_path, pixel_type=sitk.sitkFloat32):
    """Load an image and preserve its original spacing."""
    try:
        if not os.path.exists(file_path):
            logging.error(f"File does not exist: {file_path}")
            return None

        image = sitk.ReadImage(file_path, pixel_type)

        # Log the ACTUAL spacing from the file
        actual_spacing = image.GetSpacing()
        logging.info(f"Loaded {file_path} with original spacing: {actual_spacing}")

        # Set standard orientation
        image.SetDirection([1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0])
        image.SetOrigin((0.0, 0.0, 0.0))

        # Clean after loading (optional)
        # image = cleanup1(image)

        return image
    except Exception as e:
        logging.error(f"Error loading image {file_path}: {e}")
        traceback.print_exc()
        return None

def resample_image1(image, target_spacing=(0.06, 0.06, 0.06)):
    """Resample the image to the target spacing using linear interpolation."""
    try:
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        new_size = [
            int(np.round(original_size[i] * (original_spacing[i] / target_spacing[i])))
            for i in range(3)
        ]
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(0)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampled_image = resampler.Execute(image)
        logging.info(f"Resampled image to spacing: {target_spacing}, new size: {new_size}")
        return resampled_image
    except Exception as e:
        logging.error(f"Error resampling image: {e}")
        traceback.print_exc()
        return None

def cleanup1(image):
    """
    Clean an intensity image by creating a binary mask, cleaning it,
    then applying it to the original image to preserve intensities.
    """
    try:
        # Get image stats
        stats = sitk.StatisticsImageFilter()
        stats.Execute(image)
        
        min_value = stats.GetMinimum()
        max_value = stats.GetMaximum()
        mean_value = stats.GetMean()
        
        # For empty images, return as is
        if max_value < 1e-6:
            return image
        elif max_value <= 1:
            max_value = 1

        # Robust thresholding based on mean
        threshold = mean_value * 0.50

        if threshold >= max_value:
            threshold = max_value - 0.001
        
        # Create binary mask from thresholding
        binary = sitk.BinaryThreshold(image, threshold, max_value, 1, 0)

        # Step 2: Clean binary mask to remove small components
        min_comp_size = 0.01
        bone_volume = int(sitk.GetArrayFromImage(binary).sum())
        min_comp_size = max(10, int(bone_volume * min_comp_size))  
        
        # Find connected components
        cc = sitk.ConnectedComponent(binary)
        
        # Keep only significant components
        relabel = sitk.RelabelComponentImageFilter()
        relabel.SetMinimumObjectSize(min_comp_size)  
        cleaned_mask = relabel.Execute(cc)
        
        # Convert back to binary
        cleaned_mask = sitk.BinaryThreshold(cleaned_mask, 1, relabel.GetNumberOfObjects(), 1, 0)
        
        # Check if we removed everything
        mask_stats = sitk.StatisticsImageFilter()
        mask_stats.Execute(cleaned_mask)
        
        if mask_stats.GetSum() < 1:
            logging.warning("Cleaning removed all content. Using original mask.")
            cleaned_mask = binary
        
        # Step 3: Apply cleaned mask to Filled mask to remove background noise
        cleaned_image = sitk.Mask(image, cleaned_mask)
        cleaned_image = sitk.Cast(cleaned_image, sitk.sitkUInt8)
        
        return cleaned_image
        
    except Exception as e:
        logging.error(f"Error in clean_intensity_image: {e}")
        traceback.print_exc()
        return image  # Return original if cleaning fails


def find_bounding_box(filled_image, padding_mm=10.0):
    """
    Finds the bounding box of the non-zero regions with padding to prevent volume loss.

    Parameters:
    -----------
    filled_image : sitk.Image
        Binary filled image
    padding_mm : float
        Padding in millimeters to add around the bounding box
    """
    try:
        # Label connected components
        labeled_mask = sitk.ConnectedComponent(filled_image)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(labeled_mask)

        # Check if any labels are present
        labels = stats.GetLabels()
        if not labels:
            logging.warning("No bone regions found in Filled image.")
            return None

        # Get the largest connected component
        largest_label = max(labels, key=lambda l: stats.GetPhysicalSize(l))
        bounding_box = stats.GetBoundingBox(largest_label)

        # Convert padding from mm to voxels
        spacing = filled_image.GetSpacing()
        padding_voxels = [int(padding_mm / s + 0.5) for s in spacing]

        # Add padding to bounding box while respecting image boundaries
        image_size = filled_image.GetSize()
        padded_box = list(bounding_box)

        # Adjust start indices and sizes with padding
        for i in range(3):
            # Lower bound with padding
            new_start = max(0, bounding_box[i] - padding_voxels[i])
            # Upper bound with padding
            new_end = min(image_size[i], bounding_box[i] + bounding_box[3 + i] + padding_voxels[i])
            # Update box
            padded_box[i] = new_start
            padded_box[3 + i] = new_end - new_start

        logging.info(f"Original bounding box: {bounding_box}, Padded: {padded_box}")
        return tuple(padded_box)

    except Exception as e:
        logging.error(f"Error finding bounding box: {e}")
        traceback.print_exc()
        return None


def process_bones(data_dir, target_spacing=(0.06, 0.06, 0.06), padding_mm=10.0):
    """
    Processes all Bones: load -> resample to target spacing -> crop -> pad.
    Preserves physical dimensions during resampling.
    """
    try:
        filled_dir = os.path.join(data_dir, 'Filled')
        raw_dir = os.path.join(data_dir, 'Raw')
        cortical_dir = os.path.join(data_dir, 'Cortical')
        trabecular_dir = os.path.join(data_dir, 'Trabecular')

        for dir_path in [filled_dir, raw_dir, cortical_dir, trabecular_dir]:
            if not os.path.exists(dir_path):
                logging.error(f"Required directory '{dir_path}' does not exist.")
                return {}, [0, 0, 0]

        all_files = os.listdir(filled_dir)
        bone_files = [f for f in all_files if f.lower().endswith('.tiff') or f.lower().endswith('.tif')]

        if not bone_files:
            logging.error(f"No .tiff or .tif files found in {filled_dir}")
            return {}, [0, 0, 0]

        bone_names = [os.path.splitext(f)[0] for f in bone_files]
        logging.info(f"Found {len(bone_names)} Bones")

        bone_info = {}
        max_size = [0, 0, 0]
        processed_count = 0

        for bone, bone_file in zip(bone_names, bone_files):
            try:
                filled_path = os.path.join(filled_dir, bone_file)

                # Load with ORIGINAL spacing preserved
                filled_image = load_image(filled_path)
                if filled_image is None:
                    continue

                original_spacing = filled_image.GetSpacing()
                logging.info(f"{bone}: Original spacing {original_spacing}")

                # NOW resample to target spacing (preserves physical size)
                filled_image_resampled = resample_image1(filled_image, target_spacing=target_spacing)
                if filled_image_resampled is None:
                    continue

                # Verify physical size is preserved
                orig_phys_size = [filled_image.GetSize()[i] * original_spacing[i] for i in range(3)]
                new_phys_size = [filled_image_resampled.GetSize()[i] * target_spacing[i] for i in range(3)]
                logging.info(f"{bone}: Physical size preserved: {orig_phys_size} -> {new_phys_size}")

                # Find bounding box with padding
                bounding_box = find_bounding_box(filled_image_resampled, padding_mm=padding_mm)
                if bounding_box is None:
                    continue

                # Update max size
                for i in range(3):
                    if bounding_box[3 + i] > max_size[i]:
                        max_size[i] = bounding_box[3 + i]

                # Process all stacks
                cropped_images = {}
                for stack_name, stack_dir in zip(
                        ['Raw', 'Cortical', 'Trabecular', 'Filled'],
                        [raw_dir, cortical_dir, trabecular_dir, filled_dir]
                ):
                    stack_files = os.listdir(stack_dir)
                    matching = [f for f in stack_files if os.path.splitext(f)[0] == bone and
                                (f.lower().endswith('.tiff') or f.lower().endswith('.tif'))]

                    if not matching:
                        continue

                    stack_path = os.path.join(stack_dir, matching[0])

                    # Load preserving original spacing
                    image = load_image(stack_path)
                    if image is None:
                        continue

                    # Resample to target spacing
                    image_resampled = resample_image1(image, target_spacing=target_spacing)
                    if image_resampled is None:
                        continue

                    # Crop
                    cropped_image = crop_image(image_resampled, bounding_box)
                    if cropped_image is None:
                        continue

                    cropped_images[stack_name] = cropped_image

                bone_info[bone] = {
                    'bounding_box': bounding_box,
                    'cropped_images': cropped_images
                }

                processed_count += 1
                logging.info(f"Processed {bone} ({processed_count}/{len(bone_names)})")

                del filled_image, filled_image_resampled, cropped_images
                gc.collect()

            except Exception as e:
                logging.error(f"Error processing {bone}: {e}")
                traceback.print_exc()
                continue

        logging.info(f"Maximum bounding box: {max_size}")
        return bone_info, max_size

    except Exception as e:
        logging.error(f"Error in process_bones: {e}")
        traceback.print_exc()
        return {}, [0, 0, 0]

def find_bounding_box67(filled_image):
    """
    Finds the bounding box of the non-zero regions in the Filled binary image.
    """
    try:
        # Label connected components
        labeled_mask = sitk.ConnectedComponent(filled_image)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(labeled_mask)

        # Check if any labels are present
        labels = stats.GetLabels()
        if not labels:
            logging.warning("No bone regions found in Filled image.")
            return None

        # Assume the largest connected component represents the bone
        largest_label = max(labels, key=lambda l: stats.GetPhysicalSize(l))
        bounding_box = stats.GetBoundingBox(largest_label)  # (x_min, y_min, z_min, size_x, size_y, size_z)

        return bounding_box
    except Exception as e:
        logging.error(f"Error finding bounding box: {e}")
        traceback.print_exc()
        return None

def crop_image(image, bounding_box):
    """
    Crops the image based on the provided bounding box and updates the origin.
    """
    try:
        extractor = sitk.RegionOfInterestImageFilter()
        extractor.SetIndex(bounding_box[:3])
        extractor.SetSize(bounding_box[3:])
        cropped_image = extractor.Execute(image)

        # Update the origin to match the new physical location
        original_origin = image.GetOrigin()
        original_spacing = image.GetSpacing()
        new_origin = [
            original_origin[i] + bounding_box[i] * original_spacing[i]
            for i in range(len(original_origin))
        ]
        cropped_image.SetOrigin(new_origin)
        cropped_image.SetDirection(image.GetDirection())
        cropped_image.SetSpacing(image.GetSpacing())

        logging.info(f"Cropped image to bounding box: {bounding_box}, new origin: {new_origin}")
        return cropped_image
    except Exception as e:
        logging.error(f"Error cropping image: {e}")
        traceback.print_exc()
        return None

def pad_image(image, target_size):
    """
    Pads the image to match the target size and adjusts the origin if necessary.
    """
    try:
        current_size = image.GetSize()
        padding_lower = []
        padding_upper = []
        new_origin = list(image.GetOrigin())
        spacing = image.GetSpacing()

        for i in range(len(current_size)):
            total_padding = target_size[i] - current_size[i]
            if total_padding < 0:
                logging.error(f"Target size {target_size[i]} is smaller than current size {current_size[i]}. Cannot pad.")
                return None
            pad_lower = total_padding // 2
            pad_upper = total_padding - pad_lower
            padding_lower.append(pad_lower)
            padding_upper.append(pad_upper)

            # Adjust the origin for padding on the lower side
            new_origin[i] -= pad_lower * spacing[i]

        # Apply padding
        padded_image = sitk.ConstantPad(image, padding_lower, padding_upper, constant=0)
        padded_image.SetOrigin(tuple(new_origin))
        padded_image.SetDirection(image.GetDirection())
        padded_image.SetSpacing(image.GetSpacing())

        logging.info(f"Padded image to target size: {target_size}, new origin: {new_origin}")
        return padded_image
    except Exception as e:
        logging.error(f"Error padding image: {e}")
        traceback.print_exc()
        return None

def process_bones67(data_dir, expected_spacing=(0.06,0.06,0.06)):
    """
    Processes all Bones by trimming based on the Filled image bounding box and determining the maximum bounding box size.
    """
    try:
        # Define stack directories
        filled_dir = os.path.join(data_dir, 'Filled')
        raw_dir = os.path.join(data_dir, 'Raw')
        cortical_dir = os.path.join(data_dir, 'Cortical')
        trabecular_dir = os.path.join(data_dir, 'Trabecular')

        # Ensure all directories exist
        for dir_path in [filled_dir, raw_dir, cortical_dir, trabecular_dir]:
            if not os.path.exists(dir_path):
                logging.error(f"Required directory '{dir_path}' does not exist.")
                return {}, [0, 0, 0]

        # Check what's actually in the Filled directory
        all_files = os.listdir(filled_dir)
        logging.info(f"All files in {filled_dir}: {all_files[:10]}...")  # Show first 10 files
        
        # List of Bones based on Filled images - handle case insensitivity
        bone_files = [f for f in all_files if f.lower().endswith('.tiff') or f.lower().endswith('.tif')]
        
        if len(bone_files) == 0:
            logging.error(f"No .tiff or .tif files found in {filled_dir}")
            logging.error(f"Directory contains {len(all_files)} files total")
            # Check if files might have different extensions
            other_extensions = set([os.path.splitext(f)[1] for f in all_files if os.path.splitext(f)[1]])
            if other_extensions:
                logging.error(f"Found files with these extensions: {other_extensions}")
            return {}, [0, 0, 0]
        
        # Extract bone names (just the ID without extension)
        bone_names = [os.path.splitext(f)[0] for f in bone_files]

        logging.info(f"Found {len(bone_names)} Bones in Filled directory")
        logging.info(f"bone IDs: {', '.join(bone_names[:10])}...")  # Show first 10

        bone_info = {}
        max_size = [0, 0, 0]
        processed_count = 0

        # First Pass: Determine bounding boxes and maximum size
        for bone_idx, (bone, bone_file) in enumerate(zip(bone_names, bone_files)):
            try:
                # Use the actual filename from the directory
                filled_path = os.path.join(filled_dir, bone_file)
                
                if not os.path.exists(filled_path):
                    logging.warning(f"Filled image file not found for {bone} at {filled_path}. This shouldn't happen!")
                    continue

                logging.debug(f"Loading filled image for {bone} from {filled_path}")

                # Load and resample Filled image
                filled_image = load_image(filled_path)
                if filled_image is None:
                    logging.warning(f"Failed to load Filled image for {bone}. Skipping.")
                    continue

                filled_image_resampled = resample_image1(filled_image, target_spacing=expected_spacing)
                if filled_image_resampled is None:
                    logging.warning(f"Failed to resample Filled image for {bone}. Skipping.")
                    continue

                # Find bounding box
                bounding_box = find_bounding_box(filled_image_resampled)
                if bounding_box is None:
                    logging.warning(f"No bone regions detected in {bone}. Skipping.")
                    continue

                # Update maximum size
                for i in range(3):
                    if bounding_box[3 + i] > max_size[i]:
                        max_size[i] = bounding_box[3 + i]

                # Load, resample, and crop all image stacks
                cropped_images = {}
                for stack_name, stack_dir, pixel_type in zip(
                    ['Raw', 'Cortical', 'Trabecular', 'Filled'],
                    [raw_dir, cortical_dir, trabecular_dir, filled_dir],
                    [sitk.sitkUInt8, sitk.sitkUInt8, sitk.sitkUInt8, sitk.sitkUInt8]
                ):
                    # Try to find the file with matching ID in the stack directory
                    stack_files = os.listdir(stack_dir)
                    matching_files = [f for f in stack_files if os.path.splitext(f)[0] == bone and
                                    (f.lower().endswith('.tiff') or f.lower().endswith('.tif'))]
                    
                    if not matching_files:
                        logging.warning(f"'{stack_name}' file not found for {bone} in {stack_dir}. Skipping this stack.")
                        continue
                    
                    stack_path = os.path.join(stack_dir, matching_files[0])
                    
                    image = load_image(stack_path)
                    if image is None:
                        logging.warning(f"Failed to load {stack_name} image for {bone}. Skipping.")
                        continue

                    image_resampled = resample_image1(image, target_spacing=expected_spacing)
                    if image_resampled is None:
                        logging.warning(f"Failed to resample {stack_name} image for {bone}. Skipping.")
                        continue

                    cropped_image = crop_image(image_resampled, bounding_box)
                    if cropped_image is None:
                        logging.warning(f"Failed to crop {stack_name} image for {bone}. Skipping.")
                        continue

                    cropped_images[stack_name] = cropped_image

                # Store information
                bone_info[bone] = {
                    'bounding_box': bounding_box,
                    'cropped_images': cropped_images
                }

                processed_count += 1
                logging.info(f"Successfully processed {bone} ({processed_count}/{len(bone_names)}): Bounding box size = {bounding_box[3:]}")

                # Clean up to save memory
                del filled_image, filled_image_resampled, cropped_images
                gc.collect()

            except Exception as e:
                logging.error(f"An error occurred while processing {bone}: {e}")
                traceback.print_exc()
                continue  # Skip to the next bone

        logging.info(f"Successfully processed {processed_count} out of {len(bone_names)} Bones")
        logging.info(f"Maximum bounding box size across all Bones: {max_size}")

        return bone_info, max_size

    except Exception as e:
        logging.error(f"An unexpected error occurred in process_Bones: {e}")
        traceback.print_exc()
        return {}, [0, 0, 0]

def pad_bones(bone_info, max_size):
    """ 
    Pads each bone's image stacks to match the maximum bounding box size.
    """
    try:
        padded_bone_info = {}

        for bone, info in bone_info.items():
            try:
                padded_images = {}
                for stack, image in info['cropped_images'].items():
                    padded_image = pad_image(image, max_size)
                    if padded_image is None:
                        logging.error(f"Padding failed for {bone} - {stack}.")
                        continue
                    padded_images[stack] = padded_image
                padded_bone_info[bone] = padded_images
                logging.info(f"Padded images for {bone} to size {max_size}")

                # Clean up to save memory
                del info['cropped_images']
                gc.collect()

            except Exception as e:
                logging.error(f"An error occurred while padding images for {bone}: {e}")
                traceback.print_exc()
                continue  # Skip to the next bone

        return padded_bone_info
    except Exception as e:
        logging.error(f"An unexpected error occurred in pad_Bones: {e}")
        traceback.print_exc()
        return {}


def process_single_bone_aligned(bone, temp_dir, expected_spacing, max_size, output_dir):
    """
    Process a single aligned bone: load -> crop -> pad -> save -> cleanup.
    Memory-efficient approach to avoid RAM exhaustion.
    """
    try:
        # Load filled image first to get bounding box
        filled_dir = os.path.join(temp_dir, 'Filled')
        filled_files = os.listdir(filled_dir)
        matching = [f for f in filled_files if os.path.splitext(f)[0] == bone and
                    (f.lower().endswith('.tiff') or f.lower().endswith('.tif'))]

        if not matching:
            logging.warning(f"Filled file not found for {bone}")
            return False

        filled_path = os.path.join(filled_dir, matching[0])
        filled_image = load_image(filled_path)

        if filled_image is None:
            return False

        # Get bounding box
        bounding_box = find_bounding_box(filled_image)
        if bounding_box is None:
            del filled_image
            return False

        # Process each image type
        for stack_name in ['Raw', 'Cortical', 'Trabecular', 'Filled']:
            stack_dir = os.path.join(temp_dir, stack_name)
            stack_files = os.listdir(stack_dir)
            matching = [f for f in stack_files if os.path.splitext(f)[0] == bone and
                        (f.lower().endswith('.tiff') or f.lower().endswith('.tif'))]

            if not matching:
                logging.warning(f"'{stack_name}' file not found for {bone}. Skipping.")
                continue

            # Load image
            stack_path = os.path.join(stack_dir, matching[0])

            if stack_name == 'Filled':
                image = filled_image  # Reuse
            else:
                image = load_image(stack_path, expected_spacing)
                if image is None:
                    continue

            # Crop
            cropped = crop_image(image, bounding_box)
            if cropped is None:
                if stack_name != 'Filled':
                    del image
                continue

            # Pad
            padded = pad_image(cropped, max_size)
            if padded is None:
                logging.error(f"Padding failed for {bone} - {stack_name}")
                del cropped
                if stack_name != 'Filled':
                    del image
                continue

            # Save immediately
            output_path = os.path.join(output_dir, stack_name, f"{bone}.tiff")
            os.makedirs(os.path.join(output_dir, stack_name), exist_ok=True)
            sitk.WriteImage(padded, output_path)
            logging.info(f"Saved '{stack_name}' for {bone}")

            # Cleanup
            del cropped, padded
            if stack_name != 'Filled':
                del image
            gc.collect()

        # Final cleanup
        del filled_image
        gc.collect()
        return True

    except Exception as e:
        logging.error(f"Error processing aligned bone {bone}: {e}")
        traceback.print_exc()
        return False


def determine_max_size_from_aligned(temp_dir, expected_spacing):
    """
    Determine maximum bounding box from aligned Filled images.
    Only loads Filled images temporarily to save memory.
    """
    try:
        filled_dir = os.path.join(temp_dir, 'Filled')
        filled_files = [f for f in os.listdir(filled_dir)
                        if f.lower().endswith('.tiff') or f.lower().endswith('.tif')]
        bone_names = [os.path.splitext(f)[0] for f in filled_files]

        logging.info(f"Determining max bounding box from {len(bone_names)} aligned Bones")

        max_size = [0, 0, 0]
        valid_bones = []

        for bone, bone_file in zip(bone_names, filled_files):
            filled_path = os.path.join(filled_dir, bone_file)
            filled_image = load_image(filled_path)

            if filled_image is None:
                continue

            bounding_box = find_bounding_box(filled_image)

            if bounding_box is None:
                del filled_image
                continue

            # Update max size
            for i in range(3):
                if bounding_box[3 + i] > max_size[i]:
                    max_size[i] = bounding_box[3 + i]

            valid_bones.append(bone)

            # Immediate cleanup
            del filled_image
            gc.collect()

        logging.info(f"Maximum bounding box: {max_size}")
        return max_size, valid_bones

    except Exception as e:
        logging.error(f"Error determining max size: {e}")
        traceback.print_exc()
        return [0, 0, 0], []

def save_processed_bones(padded_bone_info, output_dir):
    """ 
    Saves the padded image stacks to the specified output directory.
    Modified to use ID.tiff naming convention.
    """
    try:
        # Create output subdirectories
        for stack in ['Raw', 'Cortical', 'Trabecular', 'Filled']:
            stack_output_dir = os.path.join(output_dir, stack)
            os.makedirs(stack_output_dir, exist_ok=True)

        for bone, images in padded_bone_info.items():
            for stack, image in images.items():
                # Save with just ID.tiff format
                stack_output_path = os.path.join(output_dir, stack, f"{bone}.tiff")
                sitk.WriteImage(image, stack_output_path)
                logging.info(f"Saved '{stack}' for {bone} to '{stack_output_path}'")

            # Clean up to save memory
            del images
            gc.collect()

        logging.info(f"All processed Bones saved to '{output_dir}'.")
    except Exception as e:
        logging.error(f"An unexpected error occurred in save_processed_Bones: {e}")
        traceback.print_exc()

def validate_processed_images(output_dir, expected_spacing):
    """ 
    Validates that all processed images have the expected size, spacing, origin, and direction.
    """
    try:
        expected_direction = [1.0, 0.0, 0.0,
                              0.0, 1.0, 0.0,
                              0.0, 0.0, 1.0]

        for stack in ['Raw', 'Cortical', 'Trabecular', 'Filled']:
            stack_output_dir = os.path.join(output_dir, stack)
            for filename in os.listdir(stack_output_dir):
                if filename.endswith('.tiff') or filename.endswith('.tif'):
                    image_path = os.path.join(stack_output_dir, filename)
                    image = sitk.ReadImage(image_path)
                    spacing = image.GetSpacing()
                    direction = image.GetDirection()
                    origin = image.GetOrigin()
                    size = image.GetSize()

                    if spacing != expected_spacing:
                        logging.warning(f"Image {filename} in {stack} has unexpected spacing: {spacing}")
                    if direction != tuple(expected_direction):
                        logging.warning(f"Image {filename} in {stack} has unexpected direction: {direction}")
                    if origin != (0.0, 0.0, 0.0):
                        logging.warning(f"Image {filename} in {stack} has unexpected origin: {origin}")
                    logging.info(f"Validated image {filename} in {stack}: size={size}, spacing={spacing}, origin={origin}")

                    # Clean up
                    del image
                    gc.collect()
    except Exception as e:
        logging.error(f"An unexpected error occurred in validate_processed_images: {e}")
        traceback.print_exc()


def similarity_to_rigid_transform(similarity_transform):
    """
    Extract rigid transform (rotation + translation) from similarity transform.
    Discards the scale component to preserve original specimen sizes.
    """
    try:
        # Get transformation components
        center = similarity_transform.GetCenter()
        translation = similarity_transform.GetTranslation()

        # Get the 3x3 rotation matrix from similarity (includes scale)
        sim_matrix = np.array(similarity_transform.GetMatrix()).reshape(3, 3)

        # Remove scale by normalizing each column (orthonormalization)
        # This extracts just the rotation component
        rotation_matrix = np.zeros((3, 3))
        for i in range(3):
            rotation_matrix[:, i] = sim_matrix[:, i] / np.linalg.norm(sim_matrix[:, i])

        # Create rigid transform
        rigid_transform = sitk.VersorRigid3DTransform()
        rigid_transform.SetCenter(center)
        rigid_transform.SetMatrix(rotation_matrix.flatten().tolist())
        rigid_transform.SetTranslation(translation)

        # Get scale for logging
        params = similarity_transform.GetParameters()
        scale = params[0]
        logging.info(f"Extracted rigid transform (discarded scale: {scale:.4f})")

        return rigid_transform

    except Exception as e:
        logging.error(f"Error converting similarity to rigid transform: {e}")
        traceback.print_exc()
        return None


def _recursive_flatten(transform, flat_list):
    """Recursive helper function to unpack all transforms."""
    # Check if the transform is a composite by looking for the method
    if hasattr(transform, 'GetNumberOfTransforms'):
        for i in range(transform.GetNumberOfTransforms()):
            # Recursively call on each sub-transform
            _recursive_flatten(transform.GetTransform(i), flat_list)
    else:
        # Base case: it's a simple transform, so add it to our list
        flat_list.append(transform)


def flatten_transform(transform):
    """
    Creates a final, flattened composite transform from a potentially nested one.
    """
    flat_transform_list = []
    # Unpack all transforms into a simple Python list
    _recursive_flatten(transform, flat_transform_list)

    # Create the final, single-level composite transform from the flat list
    final_flat_transform = sitk.CompositeTransform(3)
    for tx in flat_transform_list:
        final_flat_transform.AddTransform(tx)

    return final_flat_transform

def calculate_resample_grid(final_transform, reference_image, moving_image):
    """
    Calculates a new image grid that encompasses both the reference and the
    transformed moving image to prevent clipping.
    """
    # Find the 8 corners of the moving image
    moving_size = moving_image.GetSize()
    corners = [
        (0, 0, 0),
        (moving_size[0], 0, 0),
        (0, moving_size[1], 0),
        (0, 0, moving_size[2]),
        (moving_size[0], moving_size[1], 0),
        (moving_size[0], 0, moving_size[2]),
        (0, moving_size[1], moving_size[2]),
        (moving_size[0], moving_size[1], moving_size[2]),
    ]

    # Transform each corner point
    physical_corners = [moving_image.TransformIndexToPhysicalPoint(c) for c in corners]
    transformed_corners = [final_transform.TransformPoint(pc) for pc in physical_corners]

    # Include the corners of the reference image as well
    ref_size = reference_image.GetSize()
    ref_corners_phys = [
        reference_image.TransformIndexToPhysicalPoint((0, 0, 0)),
        reference_image.TransformIndexToPhysicalPoint((ref_size[0], ref_size[1], ref_size[2]))
    ]

    all_corners = transformed_corners + ref_corners_phys

    # Find the min and max coordinates across all corners
    min_x = min(c[0] for c in all_corners)
    min_y = min(c[1] for c in all_corners)
    min_z = min(c[2] for c in all_corners)
    max_x = max(c[0] for c in all_corners)
    max_y = max(c[1] for c in all_corners)
    max_z = max(c[2] for c in all_corners)

    # Define the new grid properties
    output_spacing = reference_image.GetSpacing()
    output_origin = [min_x, min_y, min_z]
    output_size = [
        int((max_x - min_x) / output_spacing[0]),
        int((max_y - min_y) / output_spacing[1]),
        int((max_z - min_z) / output_spacing[2]),
    ]
    output_direction = reference_image.GetDirection()
    return output_size, output_origin, output_spacing, output_direction


def align(input_dir, output_dir, reference_name, scale_factor=5, cores='detect'):
    """
    Main function to execute bone alignment.
    Includes an iterative refinement loop to ensure a minimum Dice score.
    """
    # Configure logging
    log_file = os.path.join(output_dir, f"Alignment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

    start_time = time.time()
    logging.info(f"Starting image alignment to reference: {reference_name}")

    # Set the number of threads for SimpleITK
    if cores == 'detect':
        num_cores = max(1, multiprocessing.cpu_count() - 2)  # Keep 2 cores free, but use at least 1
    else:
        num_cores = int(cores)
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(num_cores)
    logging.info(f"Using {num_cores} CPU threads for SimpleITK processing")

    try:
        # Define and check stack directories
        dir_map = {
            'Filled': os.path.join(input_dir, 'Filled'),
            'Raw': os.path.join(input_dir, 'Raw'),
            'Cortical': os.path.join(input_dir, 'Cortical'),
            'Trabecular': os.path.join(input_dir, 'Trabecular')
        }
        for dir_name, dir_path in dir_map.items():
            if not os.path.exists(dir_path):
                logging.error(f"Required directory '{dir_path}' does not exist.")
                return False
            logging.info(f"Found image directory: {dir_path}")

        # Load reference image
        reference_image_path = os.path.join(dir_map['Filled'], f"{reference_name}.tiff")
        if not os.path.exists(reference_image_path):
            reference_image_path = os.path.join(dir_map['Filled'], f"{reference_name}.tif")

        reference_image = load_image(reference_image_path)
        if reference_image is None:
            logging.error("Failed to load the reference image")
            return False

        reference_image_rescaled = resample_image(reference_image, scale_factor)
        if reference_image_rescaled is None:
            logging.error("Failed to resample reference image")
            return False

        metrics = {"dice": [], "hd": [], "msd": []}

        bone_files = [f for f in os.listdir(dir_map['Filled']) if f.lower().endswith(('.tiff', '.tif'))]
        bone_names = [os.path.splitext(f)[0] for f in bone_files]
        logging.info(f"Found {len(bone_names)} specimens to align.")

        for bone in bone_names:
            try:
                logging.info(f"--- Processing {bone} ---")

                moving_image_path = os.path.join(dir_map['Filled'], f"{bone}.tiff")
                if not os.path.exists(moving_image_path):
                    moving_image_path = os.path.join(dir_map['Filled'], f"{bone}.tif")

                moving_image = load_image(moving_image_path)
                if moving_image is None: continue

                moving_image_rescaled = resample_image(moving_image, scale_factor)

                fixed = sitk.Cast(reference_image_rescaled, sitk.sitkFloat32)
                moving = sitk.Cast(moving_image_rescaled, sitk.sitkFloat32)

                # ============= 1. COARSE ALIGNMENT PROCEDURE =============
                logging.info(f"Performing initial alignment check for {bone}...")

                # --- Guess 1: Standard Centroid Alignment ---
                initial_transform_A = sitk.CenteredTransformInitializer(
                    fixed,
                    moving,
                    sitk.Euler3DTransform(),
                    #sitk.CenteredTransformInitializerFilter.GEOMETRY
                    sitk.CenteredTransformInitializerFilter.MOMENTS
                )
                # --- Guess 2: Flipped Centroid Alignment (180 deg rotation around X-axis) ---
                rotation_center = initial_transform_A.GetCenter()
                x_transform = sitk.Euler3DTransform(rotation_center, math.pi, 0, 0)
                y_transform = sitk.Euler3DTransform(rotation_center, 0, math.pi, 0)
                z_transform = sitk.Euler3DTransform(rotation_center, 0, 0, math.pi)
                xyz_transform = sitk.Euler3DTransform(rotation_center, math.pi, math.pi, math.pi)

                xtrans = sitk.CompositeTransform([initial_transform_A, x_transform])
                ytrans = sitk.CompositeTransform([initial_transform_A, y_transform])
                ztrans = sitk.CompositeTransform([initial_transform_A, z_transform])
                xyztrans = sitk.CompositeTransform([initial_transform_A, xyz_transform])

                # --- Run a quick test on both initializations ---
                metric_values = {}
                for name, init_tx in [("Standard", initial_transform_A), ("X-Rotation", xtrans), ("Y-Rotation", ytrans), ("Z-Rotation", ztrans), ("XYZ-Rotation",xyztrans)]:
                    quick_reg = sitk.ImageRegistrationMethod()
                    quick_reg.SetMetricAsMeanSquares()
                    #quick_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
                    quick_reg.SetInterpolator(sitk.sitkLinear)
                    quick_reg.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0,
                                                                       numberOfIterations=50,
                                                                       minStep=1e-4)
                    quick_reg.SetInitialTransform(init_tx)
                    quick_reg.Execute(fixed, moving)
                    metric_values[name] = quick_reg.GetMetricValue()

                # --- Choose the best initial transform ---
                best_initial_name = min(metric_values, key=metric_values.get)

                if best_initial_name == "Standard":
                    initial_transform = initial_transform_A
                else:
                    if best_initial_name == "X-Rotation":
                        initial_transform = x_transform
                    if best_initial_name == "Y-Rotation":
                        initial_transform = y_transform
                    if best_initial_name == "Z-Rotation":
                        initial_transform = z_transform
                    if best_initial_name == "XYZ-Rotation":
                        initial_transform = xyz_transform
                logging.info(
                    f"Chose '{best_initial_name}' initialization (Metric: {metric_values[best_initial_name]:.4f})"
                )

                # ============= 2. FINE ALIGNMENT: ITERATIVE REFINEMENT LOOP =============
                max_attempts = 3
                dice_threshold = 0.5
                current_transform = initial_transform
                final_transform = initial_transform
                best_dice_so_far = -1.0

                logging.info(f"Starting iterative refinement for {bone} (goal: Dice >= {dice_threshold})...")

                for attempt in range(max_attempts):
                    logging.info(f"--- Attempt {attempt + 1} of {max_attempts} ---")

                    # Use a Mask
                    #fixed_mask = fill(fixed)
                    # fixed_image = sitk.Mask(fixed, fixed_mask)
                    #moving_mask = fill(moving)
                    # moving_image = sitk.Mask(moving_image, moving_mask)
                    #fixed = sitk.Cast(fixed_mask, sitk.sitkFloat32)
                    #moving = sitk.Cast(moving_mask, sitk.sitkFloat32)

                    registration_method = sitk.ImageRegistrationMethod()
                    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
                    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
                    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

                    sampling_percentage = min(0.05 + (attempt * 0.1), 1.0)
                    registration_method.SetMetricAsMeanSquares()
                    #registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
                    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
                    registration_method.SetMetricSamplingPercentage(sampling_percentage)

                    #registration_method.SetMetricFixedMask(fixed)
                    logging.info(f"Using sampling percentage: {sampling_percentage * 100:.1f}%")

                    registration_method.SetInterpolator(sitk.sitkLinear)

                    # *** OPTIMIZER IMPROVEMENT: Auto-estimate learning rate ***
                    registration_method.SetOptimizerAsRegularStepGradientDescent(
                        learningRate=1.0,
                        minStep=1e-4,
                        numberOfIterations=250,
                        relaxationFactor=0.5,
                        estimateLearningRate=registration_method.EachIteration
                    )
                    registration_method.SetOptimizerScalesFromPhysicalShift()
                    registration_method.SetInitialTransform(current_transform, inPlace=True)

                    current_transform = registration_method.Execute(fixed, moving)

                    moving_image_full = load_image(os.path.join(dir_map['Filled'], f"{bone}.tiff"))
                    transformed_image_check = sitk.Resample(
                        moving_image_full, reference_image, current_transform,
                        sitk.sitkLinear, 0.0, moving_image_full.GetPixelID()
                    )
                    dice, _, _ = compute_metrics(transformed_image_check, reference_image)

                    # Prevent the Python NoneType Formatting Crash
                    if dice is None:
                        logging.warning(f"Attempt {attempt + 1} failed: Metrics could not be computed.")
                        continue

                    logging.info(f"Attempt {attempt + 1} Dice: {dice:.4f}")

                    if dice > best_dice_so_far:
                        best_dice_so_far = dice
                        final_transform = sitk.Transform(current_transform)

                    if dice >= dice_threshold:
                        logging.info(f"Success! Dice score of {dice:.4f} meets the threshold.")
                        break
                else:
                    logging.warning(
                        f"Failed to meet Dice threshold for {bone} after {max_attempts} attempts. Best score was {best_dice_so_far:.4f}. Using the best transform found.")

                # ============= 3. FLATTEN, APPLY FINAL TRANSFORM, AND SAVE =============

                final_transform_flat = flatten_transform(final_transform)

                logging.info(f"Applying final transform (Dice: {best_dice_so_far:.4f}) to all image types...")
                transform_dir = os.path.join(output_dir, "Alignment", "Transforms")
                os.makedirs(transform_dir, exist_ok=True)
                sitk.WriteTransform(final_transform_flat, os.path.join(transform_dir, f"{bone}.tfm"))

                for dir_name, dir_path in dir_map.items():
                    full_res_path = os.path.join(dir_path, f"{bone}.tiff")
                    if not os.path.exists(full_res_path):
                        full_res_path = os.path.join(dir_path, f"{bone}.tif")
                    if not os.path.exists(full_res_path): continue

                    moving_image_full = load_image(full_res_path)
                    transformed_image = sitk.Resample(
                        moving_image_full, reference_image, final_transform_flat,
                        sitk.sitkLinear, 0.0, moving_image_full.GetPixelID()
                    )

                    if dir_name == 'Filled':
                        dice, hd, msd = compute_metrics(transformed_image, reference_image)
                        if dice is not None:
                            metrics["dice"].append(dice);
                            metrics["hd"].append(hd);
                            metrics["msd"].append(msd)
                            logging.info(f"Final metrics for {bone}: Dice={dice:.4f}, HD={hd:.4f}, MSD={msd:.4f}")

                    output_subdir = os.path.join(output_dir, dir_name)
                    os.makedirs(output_subdir, exist_ok=True)
                    sitk.WriteImage(transformed_image, os.path.join(output_subdir, f"{bone}.tiff"))

                logging.info(f"Successfully aligned and saved all types for {bone}")
                gc.collect()

            except Exception as e:
                logging.error(f"Error processing {bone}: {e}")
                traceback.print_exc()

        if metrics["dice"]:
            avg_dice = np.mean(metrics["dice"]);
            avg_hd = np.mean(metrics["hd"]);
            avg_msd = np.mean(metrics["msd"])
            logging.info(f"\n--- Alignment Summary ---\n"
                         f"Average Metrics: Dice={avg_dice:.4f}, HD={avg_hd:.4f}, MSD={avg_msd:.4f}\n")
        else:
            logging.warning("No metrics were computed.")

        logging.info(f"Alignment complete. Total time: {(time.time() - start_time):.2f} seconds")
        return
    except Exception as e:
        logging.error(f"A critical error occurred in the alignment script: {e}")
        traceback.print_exc()
        return False

def create_bone_mask(image, threshold=None):
    """Create binary mask of bone from grayscale image"""
    if threshold is None:
        # Auto-threshold using Otsu
        otsu_filter = sitk.OtsuThresholdImageFilter()
        otsu_filter.Execute(image)
        threshold = otsu_filter.GetThreshold()

    mask = sitk.BinaryThreshold(image, lowerThreshold=threshold)

    # Optional: morphological operations to clean up mask
    mask = sitk.BinaryMorphologicalClosing(mask, [3, 3, 3])
    mask = sitk.BinaryFillhole(mask)

    return mask

def compute_center_of_mass(mask):
    """
    Compute the center of mass (centroid) of a binary mask in physical coordinates
    """
    # Get the label statistics to compute center of mass
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(mask)

    # Get centroid in physical coordinates (assumes label 1 for the bone)
    if label_stats.GetNumberOfLabels() > 0:
        centroid = label_stats.GetCentroid(1)
        return centroid
    else:
        # Fallback: compute from array
        mask_array = sitk.GetArrayFromImage(mask)
        coords = np.argwhere(mask_array > 0)

        if len(coords) == 0:
            logging.warning("Empty mask - using image center")
            return mask.TransformContinuousIndexToPhysicalPoint(
                [(sz - 1) / 2.0 for sz in mask.GetSize()]
            )

        # Compute center of mass in index coordinates
        com_index = coords.mean(axis=0)
        # Convert from numpy array order (z,y,x) to ITK order (x,y,z)
        com_index_itk = [float(com_index[2]), float(com_index[1]), float(com_index[0])]

        # Convert to physical coordinates
        com_physical = mask.TransformContinuousIndexToPhysicalPoint(com_index_itk)
        return com_physical

def resample(input_dir, output_dir, expected_spacing, reference_name, cores='detect', padding=10):
    """
    Main function to execute the alignment, trimming, and padding pipeline.
    Uses iterative similarity registration (rigid component only) for alignment.
    """
    # Configure logging
    log_file = os.path.join(output_dir, f"resample_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

    start_time = time.time()
    logging.info(f"Starting image resampling and alignment to reference: {reference_name}")

    # Set the number of threads for SimpleITK
    if cores == 'detect':
        num_cores = multiprocessing.cpu_count() - 4
    else:
        num_cores = cores
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(num_cores)
    logging.info(f"Using {num_cores} CPU threads for SimpleITK processing")

    try:
        # Define stack directories
        filled_dir = os.path.join(input_dir, 'Filled')
        raw_dir = os.path.join(input_dir, 'Raw')
        cortical_dir = os.path.join(input_dir, 'Cortical')
        trabecular_dir = os.path.join(input_dir, 'Trabecular')

        # Ensure all directories exist
        for dir_path in [filled_dir, raw_dir, cortical_dir, trabecular_dir]:
            if not os.path.exists(dir_path):
                logging.error(f"Required directory '{dir_path}' does not exist.")
                return

        # Create temporary directory
        temp_dir = os.path.join(output_dir, 'temp_aligned')
        os.makedirs(temp_dir, exist_ok=True)


        # =========================================================================
        # Resample all images to the same dimensions and spacing
        # =========================================================================
        try:
            # Define your data directories here
            logging.info("Starting the Resampling Procedure.")

            # Step 1: Process Bones to Trim Based on Filled Image
            bone_info, max_size = process_bones(input_dir, expected_spacing, padding_mm=padding)

            if not bone_info:
                logging.error("No Bones were processed. Exiting pipeline.")
                return

            # Step 2: Pad Each bone's Image Stacks to Match Maximum Size
            padded_bone_info = pad_bones(bone_info, max_size)

            if not padded_bone_info:
                logging.error("Padding failed for all Bones. Exiting pipeline.")
                return

            # Step 3: Save the Processed and Padded Image Stacks
            save_processed_bones(padded_bone_info, input_dir)

            # Step 4: Validate the Processed Images
            # validate_processed_images(output_dir, expected_spacing)

            logging.info("Original images were resampled, trimmed, and padded. Continuing....")
            logging.info(f"Processed images are available at '{input_dir}'.")

        except Exception as e:
            logging.error(f"An unexpected error occurred in main: {e}")
            traceback.print_exc()

        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
        logging.info("Cleaned up temporary files")

        end_time = time.time()
        total_time = end_time - start_time
        logging.info(f"Completed in {total_time / 60:.2f} minutes")
        gc.collect()

    except Exception as e:
        logging.error(f"Error in resample: {e}")
        traceback.print_exc()

def resample67(input_dir, output_dir, expected_spacing, cores='detect'):
    """
    Main function to execute the trimming and padding pipeline.
    """
    # Configure logging to file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("resample.log"),
            logging.StreamHandler()
        ]
    )
    
    log_file = os.path.join(output_dir, f"resample_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Start Logging Progress
    start_time = time.time()
    logging.info(f"Starting image resampling of bones")

    # Set the number of threads for SimpleITK
    if cores == 'detect':
        num_cores = multiprocessing.cpu_count() - 4 
    else:
        num_cores = cores
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(num_cores)
    logging.info(f"Using {num_cores} CPU threads for SimpleITK processing")
    
    try:
        # Define your data directories here
        logging.info("Starting the trimming and padding pipeline.")

        # Step 1: Process Bones to Trim Based on Filled Image
        bone_info, max_size = process_bones(input_dir, expected_spacing)

        if not bone_info:
            logging.error("No Bones were processed. Exiting pipeline.")
            return

        # Step 2: Pad Each bone's Image Stacks to Match Maximum Size
        padded_bone_info = pad_bones(bone_info, max_size)

        if not padded_bone_info:
            logging.error("Padding failed for all Bones. Exiting pipeline.")
            return

        # Step 3: Save the Processed and Padded Image Stacks
        save_processed_bones(padded_bone_info, output_dir)

        # Step 4: Validate the Processed Images
        #validate_processed_images(output_dir, expected_spacing)

        logging.info("Trimming and padding pipeline completed successfully.")
        logging.info(f"Processed and padded images are available at '{output_dir}'.")

        # Calculate total runtime
        end_time = time.time()
        total_time = end_time - start_time
        logging.info(f"Bone resampling completed in {total_time/60:.2f} minutes")
        gc.collect()

    except Exception as e:
        logging.error(f"An unexpected error occurred in main: {e}")
        traceback.print_exc()


##################
# cHMA analysis
##################

def create_directories(output_dir):
    """Create the required directory structure for the cHMA workflow."""
    try:
        # Step A directories
        os.makedirs(os.path.join(output_dir, "Similarity_Transform", "Transforms"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "Similarity_Transform", "Filled"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "Similarity_Transform", "Trabecular"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "Similarity_Transform", "Average"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "Similarity_Transform", "Transforms2"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "BSpline_Transform", "Transforms"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "BSpline_Transform", "Transforms2"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "BSpline_Transform", "Filled"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "BSpline_Transform", "Trabecular"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "Canonical_Bone"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "Trabecular"), exist_ok=True)
        
        # Step B directories
        os.makedirs(os.path.join(output_dir, "Isotopological_Meshes"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "HMA_Results"), exist_ok=True)
        
        logging.info(f"Created directory structure in {output_dir}")
    except Exception as e:
        logging.error(f"Error creating directory structure: {e}")
        traceback.print_exc()
        raise

def load_image(file_path):
    """Load an image using SimpleITK and set the expected spacing."""
    try:
        if not os.path.exists(file_path):
            #logging.error(f"File does not exist: {file_path}")
            return None
        image = sitk.ReadImage(file_path)
        spacing = image.GetSpacing()
        expected_spacing = (spacing[1], spacing[1], spacing[1])
        image.SetSpacing(expected_spacing)
        image = cleanup(image)
        return image
    except Exception as e:
        logging.error(f"Error loading image {file_path}: {e}")
        traceback.print_exc()
        return None

def resample_image(image, scale_factor=5):
    """Resample image by the given scale factor to reduce resolution."""
    try:
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        
        # Calculate new spacing and size
        new_spacing = tuple([s * scale_factor for s in original_spacing])
        new_size = [int(round(osz / scale_factor)) for osz in original_size]
        
        # Set up resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(0)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # For binary images
        
        # Execute resampling
        resampled_image = resampler.Execute(image)
        #logging.info(f"Resampled image from size {original_size} to {new_size}, spacing from {original_spacing} to {new_spacing}")
        return resampled_image
    except Exception as e:
        logging.error(f"Error resampling image: {e}")
        traceback.print_exc()
        return None

def cleanup(image):
    """
    Clean an intensity image by creating a binary mask, cleaning it,
    then applying it to the original image to preserve intensities.
    """
    try:
        # Get image stats
        stats = sitk.StatisticsImageFilter()
        stats.Execute(image)
        
        min_value = stats.GetMinimum()
        max_value = stats.GetMaximum()
        mean_value = stats.GetMean()
        
        # For empty images, return as is
        if max_value < 1e-6:
            return image
        elif max_value <= 1:
            max_value = 1

        # Robust thresholding based on mean
        threshold = mean_value * 0.50

        if threshold >= max_value:
            threshold = max_value - 0.001
        
        # Create binary mask from thresholding
        binary = sitk.BinaryThreshold(image, threshold, max_value, 1, 0)

        # Step 2: Clean binary mask to remove small components
        min_comp_size = 0.01
        bone_volume = int(sitk.GetArrayFromImage(binary).sum())
        min_comp_size = max(10, int(bone_volume * min_comp_size))  
        
        # Find connected components
        cc = sitk.ConnectedComponent(binary)
        
        # Keep only significant components
        relabel = sitk.RelabelComponentImageFilter()
        relabel.SetMinimumObjectSize(min_comp_size)  
        cleaned_mask = relabel.Execute(cc)
        
        # Convert back to binary
        cleaned_mask = sitk.BinaryThreshold(cleaned_mask, 1, relabel.GetNumberOfObjects(), 1, 0)
        
        # Check if we removed everything
        mask_stats = sitk.StatisticsImageFilter()
        mask_stats.Execute(cleaned_mask)
        
        if mask_stats.GetSum() < 1:
            logging.warning("Cleaning removed all content. Using original mask.")
            cleaned_mask = binary
        
        # Step 3: Apply cleaned mask to Filled mask to remove background noise
        cleaned_image = sitk.Mask(image, cleaned_mask)
        cleaned_image = sitk.Cast(cleaned_image, sitk.sitkUInt8)
        
        return cleaned_image
        
    except Exception as e:
        logging.error(f"Error in clean_intensity_image: {e}")
        traceback.print_exc()
        return image  # Return original if cleaning fails


def fill(binary_image_sitk, radius_mm=2.5):
    """
    Fills small holes in a binary image using SimpleITK's memory-efficient
    morphological closing operation.

    Args:
        binary_image_sitk (sitk.Image): Input binary image (1s and 0s).
        radius_mm (float): Radius in millimeters for the closing operation.

    Returns:
        sitk.Image: Binary image after morphological closing.
    """
    # Get voxel spacing to convert physical radius to pixel/voxel radius
    spacing = binary_image_sitk.GetSpacing()

    # Convert the radius from millimeters to voxel units for each dimension
    radius_voxels = [int(radius_mm / s + 0.5) for s in spacing]

    # Create the closing filter directly from SimpleITK
    closing_filter = sitk.BinaryMorphologicalClosingImageFilter()

    # Set the size of the kernel using the radius in voxels
    closing_filter.SetKernelRadius(radius_voxels)

    # Set the value in the image that represents the "foreground" to be closed
    closing_filter.SetForegroundValue(1)

    # Execute the highly optimized filter
    filled_image = closing_filter.Execute(binary_image_sitk)

    return filled_image

def prepare_bone_image(image):
    """
    Process bone images by removing background noise and normalizing values.
    """
    try:
        # Get basic statistics
        stats = sitk.StatisticsImageFilter()
        stats.Execute(image)
        
        min_value = stats.GetMinimum()
        max_value = stats.GetMaximum()
        mean_value = stats.GetMean()
        
        # For empty images, return as is
        if max_value < 1e-6:
            return image

        if max_value <= 1:
            max_value = 1

        # Robust thresholding based on mean
        threshold = mean_value * 0.50

        if threshold >= max_value:
            threshold = max_value - 0.001
        
        # Create binary mask from thresholding
        binary_mask = sitk.BinaryThreshold(image, threshold, max_value, 1, 0)
        
        # Ensure the mask isn't empty
        mask_stats = sitk.StatisticsImageFilter()
        mask_stats.Execute(binary_mask)
        
        if mask_stats.GetSum() < 1:
            # Try even lower threshold (0.5% of max)
            threshold = min_value + 0.005 * (max_value - min_value)
            logging.warning(f"First threshold resulted in empty mask, trying: {threshold}")
            binary_mask = sitk.BinaryThreshold(image, threshold, max_value, 1, 0)
            
        # Apply the mask to the original image (keeps original values but zeros out background)
        masked_image = sitk.Mask(image, binary_mask)
        
        # Verify result
        result_stats = sitk.StatisticsImageFilter()
        result_stats.Execute(masked_image)
        
        if result_stats.GetSum() < 1:
            logging.warning("Preparation produced empty image. Using original.")
            return image
            
        return masked_image
        
    except Exception as e:
        logging.error(f"Error in prepare_bone_image: {e}")
        traceback.print_exc()
        return image

def clean_intensity_image(image, min_size_factor=0.01):
    """
    Clean an intensity image by creating a binary mask, cleaning it,
    then applying it to the original image to preserve intensities.
    """
    try:
        # Check if image is empty
        stats = sitk.StatisticsImageFilter()
        stats.Execute(image)
        
        if stats.GetSum() < 1:
            logging.warning("Cannot clean empty image")
            return image
        
        # Step 1: Create binary mask from intensity image
        threshold = stats.GetMinimum() + 0.05 * (stats.GetMaximum() - stats.GetMinimum())
        binary = sitk.BinaryThreshold(image, threshold, stats.GetMaximum(), 1, 0)
        
        # Step 2: Clean binary mask to remove small components
        bone_volume = int(sitk.GetArrayFromImage(binary).sum())
        min_comp_size = max(10, int(bone_volume * min_size_factor))
        
        
        # Find connected components
        cc = sitk.ConnectedComponent(binary)
        
        # Keep only significant components
        relabel = sitk.RelabelComponentImageFilter()
        relabel.SetMinimumObjectSize(min_comp_size)
        cleaned_mask = relabel.Execute(cc)
        
        # Convert back to binary
        cleaned_mask = sitk.BinaryThreshold(cleaned_mask, 1, relabel.GetNumberOfObjects(), 1, 0)
        
        # Check if we removed everything
        mask_stats = sitk.StatisticsImageFilter()
        mask_stats.Execute(cleaned_mask)
        
        if mask_stats.GetSum() < 1:
            logging.warning("Cleaning removed all content. Using original mask.")
            cleaned_mask = binary
        
        # Step 3: Apply cleaned mask to ORIGINAL intensity image
        cleaned_image = sitk.Mask(image, cleaned_mask, outsideValue=0)
        
        # Check result
        result_stats = sitk.StatisticsImageFilter()
        result_stats.Execute(cleaned_image)
        
        # Copy metadata
        cleaned_image.CopyInformation(image)
        
        return cleaned_image
        
    except Exception as e:
        logging.error(f"Error in clean_intensity_image: {e}")
        traceback.print_exc()
        return image  # Return original if cleaning fails

def apply_transform(image, transform, reference_image=None, interpolator=sitk.sitkNearestNeighbor):
    """Apply transform while preserving image content and structure."""
    try:
        # Verify image has content
        image = cleanup(image)
        stats = sitk.StatisticsImageFilter()
        stats.Execute(image)

        if reference_image is not None:
            reference_image = cleanup(reference_image)
        if stats.GetSum() < 1:
            logging.warning("Cannot transform empty image")
            return image
        
        # Apply transform
        if reference_image is None:
            transformed = sitk.Resample(
                image,
                image,
                transform,
                interpolator,
                0.0,
                image.GetPixelID()
            )
        else:
            transformed = sitk.Resample(
                image,
                reference_image,
                transform,
                interpolator,
                0.0,
                image.GetPixelID()
            )
        transformed = cleanup(transformed)
        
        return transformed
        
    except Exception as e:
        logging.error(f"Error applying transform: {e}")
        traceback.print_exc()
        return image  

def similarity_registration(fixed_image, moving_image):
    """Perform similarity registration that preserves bone structure."""
    try:
        # Convert images to float for registration
        fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
        moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
        
        # Log initial statistics to verify content
        fixed_stats = sitk.StatisticsImageFilter()
        moving_stats = sitk.StatisticsImageFilter()
        fixed_stats.Execute(fixed_image)
        moving_stats.Execute(moving_image)
        
        # Initialize transform
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, 
            moving_image,
            sitk.Similarity3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        
        # Configure registration
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsCorrelation()
        #registration_method.SetMetricAsMeanSquares()
        #registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        registration_method.SetOptimizerScalesFromPhysicalShift()
        registration_method.SetInitialTransform(initial_transform, inPlace=True)
        
        # Simplified optimizer settings
        #registration_method.SetOptimizerAsGradientDescent(
        #    learningRate=1.0,
        #    numberOfIterations=100
        #)

        registration_method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0,
            minStep=1e-4,
            numberOfIterations=100,
            relaxationFactor=0.5,
            estimateLearningRate=registration_method.EachIteration
        )
        
        # Execute registration
        final_transform = registration_method.Execute(fixed_image, moving_image)
        
        # Log metric value
        metric_value = registration_method.GetMetricValue()
        logging.info(f"Similarity Registration completed - Metric value: {metric_value:.6f}")
        
        return final_transform
        
    except Exception as e:
        logging.error(f"Error in similarity registration: {e}")
        traceback.print_exc()
        return None
    
def versor_to_quaternion(versor):
    """Convert a versor to a quaternion."""
    try:
        norm = np.linalg.norm(versor)
        if norm == 0:
            return np.array([0, 0, 0, 1])
        angle = norm
        axis = versor / norm
        quaternion = np.concatenate([axis * np.sin(angle / 2), [np.cos(angle / 2)]])
        return quaternion
    except Exception as e:
        logging.error(f"Error converting versor to quaternion: {e}")
        traceback.print_exc()
        return np.array([0, 0, 0, 1])

def average_quaternions(quaternions):
    try:
        M = quaternions.shape[0]
        # Choose the first quaternion as reference
        reference = quaternions[0]
        aligned_quaternions = np.copy(quaternions)
        
        # Align all quaternions to the same hemisphere
        for i in range(1, M):
            # If dot product is negative, flip the sign
            if np.dot(reference, quaternions[i]) < 0:
                aligned_quaternions[i] = -quaternions[i]
        
        # Then proceed with eigendecomposition
        A = np.zeros((4, 4))
        for q in aligned_quaternions:
            A += np.outer(q, q)
        A /= M
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        avg_quaternion = eigenvectors[:, np.argmax(eigenvalues)]
        avg_quaternion /= np.linalg.norm(avg_quaternion)
        return avg_quaternion
    except Exception as e:
        logging.error(f"Error averaging quaternions: {e}")
        traceback.print_exc()
        return np.array([0, 0, 0, 1])

def compute_average_transform(transforms, fixed_center):
    """Compute the average similarity transform from a list of transforms with better quaternion handling."""
    try:
        scales = []
        translations = []
        quaternions = []
        
        for transform in transforms:
            parameters = transform.GetParameters()
            scale = parameters[0]
            rotation = parameters[1:4]  # Versor components
            translation = parameters[4:7]
            
            # Convert versor to quaternion with better handling
            quaternion = np.zeros(4)
            
            # Extract versor (vector part of quaternion)
            versor_norm = np.linalg.norm(rotation)
            if versor_norm > 0:
                # Clamp the value to avoid arcsin domain error
                clamped_norm = min(versor_norm, 1.0)
                rotation_angle = 2 * np.arcsin(clamped_norm)
                axis = rotation / versor_norm
                
                # Convert to quaternion [x,y,z,w] format
                quaternion[0:3] = axis * np.sin(rotation_angle/2)
                quaternion[3] = np.cos(rotation_angle/2)
            else:
                # No rotation
                quaternion = np.array([0, 0, 0, 1])
            
            scales.append(scale)
            translations.append(translation)
            quaternions.append(quaternion)
        
        # Average scale and translation directly
        average_scale = np.mean(scales)
        average_translation = np.mean(translations, axis=0)
        
        # For quaternions, we need special handling
        quaternions = np.array(quaternions)
        
        # Handle potential sign flips by ensuring dot products with first quaternion are positive
        for i in range(1, len(quaternions)):
            if np.dot(quaternions[0], quaternions[i]) < 0:
                quaternions[i] = -quaternions[i]
                
        # Average the quaternions
        average_quaternion = np.mean(quaternions, axis=0)
        average_quaternion = average_quaternion / np.linalg.norm(average_quaternion)
        
        # Extract rotation components from quaternion
        w = average_quaternion[3]
        x, y, z = average_quaternion[0:3]
        
        # Create the rotation matrix
        rotation_matrix = np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
            [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
            [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
        ]).flatten()
        
        # Create new average transform
        average_transform = sitk.Similarity3DTransform()
        average_transform.SetCenter(fixed_center)
        average_transform.SetScale(average_scale)
        average_transform.SetTranslation(average_translation.tolist())
        average_transform.SetMatrix(rotation_matrix.tolist())
        
        return average_transform
    except Exception as e:
        logging.error(f"Error computing average transform: {e}")
        traceback.print_exc()
        return None

def calculate_volume(image, threshold=0):
    """Calculates the volume by counting voxels above a given threshold."""
    # The volume is the number of non-zero voxels.
    voxel_array = sitk.GetArrayViewFromImage(image)
    return np.sum(voxel_array > threshold)

def canonical_trabecular(Canonical_Bone, Averaged_Trabecular, output_path="trabecular.tiff"):
    """
    Generates a canonical trabecular bone network from two specific inputs using
    the pure-Python Slicer Surface Wrap Solidify logic to cleanly isolate the
    medullary cavity without leaking.
    """
    # Load the 3D images (assuming load_image is defined in your script)
    avg_bone_vol = load_image(Canonical_Bone)
    np_view = sitk.GetArrayViewFromImage(avg_bone_vol)
    img_min, img_max = np.min(np_view), np.max(np_view)
    bone = sitk.Cast(avg_bone_vol > float(img_min + 0.5 * (img_max - img_min)), sitk.sitkUInt8)

    messy_trab_vol = load_image(Averaged_Trabecular)
    sharpened = sitk.Cast(messy_trab_vol, sitk.sitkFloat32)
    sharpened = sitk.LaplacianSharpening(sitk.LaplacianSharpening(sharpened))
    triangle = sitk.TriangleThresholdImageFilter()
    triangle.SetInsideValue(0)
    triangle.SetOutsideValue(1)
    messy_trab_vol = triangle.Execute(sharpened)
    np_view = sitk.GetArrayViewFromImage(messy_trab_vol)
    img_min, img_max = np.min(np_view), np.max(np_view)
    messy_trab_vol = sitk.Cast(messy_trab_vol > float(img_min + 0.5 * (img_max - img_min)), sitk.sitkUInt8)

    # ---------------------------------------------------------
    # STEP 1: Process the Canonical Bone Input
    # ---------------------------------------------------------
    solid_condyle = slicer_surface_wrap_solidify(bone, wrap_radius=6)

    # ---------------------------------------------------------
    # STEP 2: Isolate the Pore Network
    # ---------------------------------------------------------
    raw_pores = solid_condyle & ~bone
    pores_array = sitk.GetArrayFromImage(raw_pores).astype(bool)
    cleaned_pores_array = morphology.remove_small_objects(pores_array, min_size=500)
    cleaned_pores = sitk.GetImageFromArray(cleaned_pores_array.astype(np.uint8))
    cleaned_pores.CopyInformation(raw_pores)

    # ---------------------------------------------------------
    # STEP 3: Surface Wrap Solidify the Pores (Medullary Cavity)
    # ---------------------------------------------------------
    medullary_cavity = slicer_surface_wrap_solidify(cleaned_pores, wrap_radius=6)
    median_filter = sitk.MedianImageFilter()
    median_filter.SetRadius([1, 1, 1])
    medullary_cavity = median_filter.Execute(medullary_cavity)

    # ---------------------------------------------------------
    # STEP 4: Conduct Trabecular Segmentation
    # ---------------------------------------------------------
    trabecular = messy_trabecular_vol & medullary_cavity
    trab_array = sitk.GetArrayFromImage(trabecular).astype(bool)
    clean_trab_array = morphology.remove_small_objects(trab_array, min_size=100)
    trabecular_clean = sitk.GetImageFromArray(clean_trab_array.astype(np.uint8))
    trabecular_clean.CopyInformation(trabecular)
    trabecular = trabecular_clean

    # ---------------------------------------------------------
    # STEP 5: Save the Output
    # ---------------------------------------------------------
    trabecular_safe = sitk.Cast(trabecular * 255, sitk.sitkUInt8)
    trabecular_filled_safe = sitk.Cast(medullary_cavity * 255, sitk.sitkUInt8)

    sitk.WriteImage(trabecular_filled_safe, output_path)
    logging.info(f"Saved Solid Canonical Trabecular Bone to {output_path}...")

    canonical_path_dir = os.path.dirname(os.path.abspath(output_path))
    new_path = os.path.join(canonical_path_dir, "Canonical_Trabecular_Image.tiff")
    sitk.WriteImage(trabecular_safe, new_path)
    logging.info(f"Saved Canonical Trabecular Bone to {new_path}...")

    del bone, solid_condyle, raw_pores, pores_array, cleaned_pores_array, cleaned_pores
    del medullary_cavity, trabecular_safe, trabecular_filled_safe
    gc.collect()

    return None

def canonical_trabecular2(Canonical_Bone, Averaged_Trabecular, output_path="trabecular.tiff", erosion_radius=1):
    """
    Generates a canonical trabecular bone network from two inputs using an
    Erosion-Bounded Shrinkwrap to cleanly isolate the medullary cavity and
    prevent the trabecular mask from bleeding into the cortical shell.
    """
    logging.info("Loading Image Volumes")
    # Load the 3D images
    avg_bone_vol = load_image(Canonical_Bone)
    avg_bone_vol = sitk.Cast(avg_bone_vol, sitk.sitkUInt8)

    messy_trab_vol = load_image(Averaged_Trabecular)
    messy_trab_vol = sitk.Cast(messy_trab_vol, sitk.sitkUInt8)

    # ---------------------------------------------------------
    # STEP 1: Process the Canonically Averaged Bone Input
    # ---------------------------------------------------------
    logging.info("Segmenting The Canonical Bone")
    bone = avg_bone_vol > 5

    logging.info("Solidifying Canonical Bone Outer Hull...")
    # Utilize the robust orthogonal 2.5D fill to seal open necks perfectly
    closed_bone = sitk.BinaryMorphologicalClosing(bone, [3, 3, 3], sitk.sitkBall)
    outer_hull = generate_robust_outer_hull(closed_bone)

    # ---------------------------------------------------------
    # STEP 2: Erosion-Bounded Shrinkwrap for Medullary Area
    # ---------------------------------------------------------
    logging.info(f"Applying protective erosion (radius={erosion_radius}) to peel away cortical shell...")
    internal_hull = sitk.BinaryErode(outer_hull, [erosion_radius, erosion_radius, erosion_radius], sitk.sitkBall)

    logging.info("Isolating internal pores...")
    internal_pores = internal_hull & ~bone

    closed_pores = sitk.BinaryMorphologicalClosing(internal_pores, [3, 3, 3], sitk.sitkBall)
    filled_pores = sitk.BinaryFillhole(closed_pores, fullyConnected=True)

    pores_array = sitk.GetArrayFromImage(filled_pores).astype(bool)
    cleaned_pores_array = morphology.remove_small_objects(pores_array, min_size=500)
    cleaned_pores = sitk.GetImageFromArray(cleaned_pores_array.astype(np.uint8))
    cleaned_pores.CopyInformation(filled_pores)

    logging.info("Building the final confined medullary cavity...")
    medullary_base = cleaned_pores > 0
    # Boosted to [5, 5, 5] to safely engulf thick, dense trabecular nodes
    medullary_closed = sitk.BinaryMorphologicalClosing(medullary_base, [5, 5, 5], sitk.sitkBall)
    medullary_cavity = sitk.BinaryFillhole(medullary_closed, fullyConnected=True)

    # CRITICAL: Intersect with internal_hull to guarantee it never touches the outer cortical plate
    medullary_cavity = medullary_cavity & internal_hull

    # ---------------------------------------------------------
    # STEP 3: Conduct Trabecular Segmentation
    # ---------------------------------------------------------
    logging.info("Segmenting the Canonical Trabecular Bone")
    messy_trabecular_mask = messy_trab_vol > 1
    trabecular = messy_trabecular_mask & medullary_cavity

    # ---------------------------------------------------------
    # STEP 4: Save the Output
    # ---------------------------------------------------------
    trabecular = sitk.Cast(trabecular * 255, sitk.sitkUInt8)
    trabecular_filled = sitk.Cast(medullary_cavity * 255, sitk.sitkUInt8)

    sitk.WriteImage(trabecular_filled, output_path)

    canonical_path = os.path.dirname(output_path)
    new_path = os.path.join(canonical_path, "Canonical_Trabecular_Image.tiff")
    sitk.WriteImage(trabecular, new_path)

    logging.info(f"Saved Canonical Trabecular Bone to {new_path}...")
    logging.info(f"Saved isoHMA Function Ready Trabecular Bone to {output_path}...")

    return None

def average(input_dir, output_path, file_extension=".tiff"):
    """
    Loads images, computes the average/median using the average_images
    function, and saves the result.
    """
    try:
        image_files = [f for f in os.listdir(input_dir) if f.endswith(file_extension)]
        if not image_files:
            print(f"Error: No files with extension '{file_extension}' found in '{input_dir}'.")
            return
    except FileNotFoundError:
        print(f"Error: The directory '{input_dir}' was not found.")
        return

    image_paths = [os.path.join(input_dir, f) for f in image_files]
    num_images = len(image_paths)
    print(f"Found {num_images} images to process.")

    # Create a list to hold the SimpleITK image objects
    image_list = []

    # Loop through all images and read them into the list
    for path in image_paths:
        print(f"Loading: {os.path.basename(path)}")
        img = load_image(path)
        img = sitk.Cast(img, sitk.sitkUInt8)
        image_list.append(img)

    # Pass the list of SITK images to the processing function
    averaged_image = average_images(image_list)

    # Check that the processing function returned a valid image before saving
    if averaged_image:
        print(f"Saving final image to: {output_path}")
        sitk.WriteImage(averaged_image, output_path)
        print("Done!")
    else:
        print("Error: Image processing failed and returned an empty result. Nothing was saved.")

def average_images(image_list):
    """
    Automatically tests multiple averaging methods on a list of registered SimpleITK images,
    and selects the optimal result based on volume preservation.

    The function iterates through three predefined averaging pipelines. For each, it
    computes the volume of the resulting averaged image. The "best" method is chosen
    based on the following criteria:
    1. It must produce an image with a volume smaller than the reference (first) image.
    2. Among all methods that meet criterion #1, it should NOT be the one with the
       largest resulting volume (i.e., it picks the second-largest volume). This avoids
       the most aggressive averaging while still ensuring some consolidation of the shape.

    Args:
        image_list (list): A list of registered SimpleITK.Image objects.

    Returns:
        SimpleITK.Image: The optimally averaged image, or None if no suitable
                         average could be created.
    """
    try:
        # 1. Validate Input Images
        if not image_list:
            print("Error: Input image list is empty.")
            return None

        # Filter out invalid or empty images
        valid_images = []
        for img in image_list:
            if img and sitk.GetArrayViewFromImage(img).any():
                valid_images.append(img)

        if not valid_images:
            print("Error: No valid images found in the input list.")
            return None

        # 2. Prepare Data and Reference
        reference_image = valid_images[0]
        original_volume = calculate_volume(reference_image)

        # Convert all images to a stacked NumPy array for efficient processing
        arrays = [sitk.GetArrayFromImage(img) for img in valid_images]
        stacked_arrays = np.stack(arrays, axis=0)
        del arrays, valid_images, image_list  # Free up memory
        gc.collect()

        # 3. Iterate Through Averaging Methods
        results = []
        print("\n--- Testing Averaging Methods ---")

        avg_mean = np.mean(stacked_arrays, axis=0)
        avg_mean = sitk.GetImageFromArray(avg_mean)
        avg_mean.CopyInformation(reference_image)

        avg_median = np.median(stacked_arrays, axis=0)
        avg_median = sitk.GetImageFromArray(avg_median)
        avg_median.CopyInformation(reference_image)

        for method_id in [1, 2, 3, 4, 5, 6, 7]:

            # --- Method-Specific Averaging ---

            if method_id == 1:
                # Mean
                processed_image = avg_mean
            elif method_id == 2:
                # Median
                processed_image = avg_median

            elif method_id == 3:
                # Median + Median Filter
                processed_image = sitk.Median(avg_median)

            elif method_id == 4:
                # Median + Mean Filter
                processed_image = sitk.Mean(avg_median)

            elif method_id == 5:
                # Mean + Mean Filter
                processed_image = sitk.Mean(avg_mean)

            elif method_id == 6:
                # Mean + Median Filter
                processed_image = sitk.Median(avg_mean)

            elif method_id == 7:
                # Median + Mean Filter + Median Filter
                temp_image = sitk.Mean(avg_median)
                processed_image = sitk.Median(temp_image)

            # --- Common Cleaning Pipeline for All Methods ---
            #cleaned_image = prepare_bone_image(processed_image)
            #cleaned_image = clean_intensity_image(cleaned_image)
            final_image = sitk.Cast(processed_image, sitk.sitkUInt8)

            # final_image = fill(cleaned_image)

            # --- Evaluate and Store Result ---
            current_volume = calculate_volume(final_image)
            results.append({
                'method': method_id,
                'volume': current_volume,
                'image': final_image
            })

        # 4. Select the Best Method Based on Your Criteria
        print("\n--- Selecting Optimal Result ---")

        # Sort the valid results by volume in descending order
        results.sort(key=lambda x: x['volume'], reverse=True)

        # Choose the result that is not the largest
        if len(results) > 1:
            best_result = results[1]
            print(f"Selected Method {best_result['method']} (Largest volume).")
        else:
            best_result = results[1]
        del results
        gc.collect()
        return best_result['image']

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        return None

def average_images1(image_list, method=2):
    try:
        if not image_list:
            logging.error("No images provided for averaging.")
            return None
        
        logging.info(f"Averaging {len(image_list)} images.")
        
        # Check that all images have content (with better error handling)
        valid_images = []
        for idx, img in enumerate(image_list):
            if img is None:
                logging.warning(f"Image #{idx} is None - skipping")
                continue
                
            try:
                stats = sitk.StatisticsImageFilter()
                stats.Execute(img)
                
                # For non-binary images, check if there's any meaningful data
                if stats.GetMean() > 0.01:  # Lower threshold for grayscale
                    valid_images.append(img)
                else:
                    logging.warning(f"Image #{idx} has too low mean value ({stats.GetMean():.6f}) - skipping")
            except Exception as e:
                logging.warning(f"Error checking stats for image #{idx}: {e}")
                continue
        
        # More detailed logging
        logging.info(f"Found {len(valid_images)} valid images out of {len(image_list)}")
        
        if not valid_images:
            logging.error("No valid images to average!")
            return None

        image_list = valid_images
        del valid_images
        image_data = image_list[0]

        # Conduct averaging
        
        if method == 1:
            # Method 1 - Median SITK based approach           
            arrays = []
            for img in image_list:
                arr = sitk.GetArrayFromImage(img)
                arrays.append(arr)
            del image_list
                
            # Compute average with more robustness
            avg_array = np.median(np.stack(arrays, axis=0), axis=0)
            del arrays

            # Convert to Simple ITK image
            avg_image = sitk.GetImageFromArray(avg_array)
            avg_image.CopyInformation(image_data)

        elif method == 2:
            # Method 2 - SITK based approach           
            arrays = []
            for img in image_list:
                arr = sitk.GetArrayFromImage(img)
                arrays.append(arr)
            del image_list

            # First Median voxels
            avg_array = np.median(np.stack(arrays, axis=0), axis=0)
            del arrays
            
            # Convert to Simple ITK image
            avg_image = sitk.GetImageFromArray(avg_array)
            avg_image.CopyInformation(image_data)

            # Get Mean Filter
            averager = sitk.MeanImageFilter()
            averager.SetRadius(1) 
            avg_image = averager.Execute(avg_image)

            # Get Median filter
            averager = sitk.MedianImageFilter()
            averager.SetRadius(1)
            avg_image = averager.Execute(avg_image)

            # Double Median filter
            averager = sitk.MedianImageFilter()
            averager.SetRadius(1)
            avg_image = averager.Execute(avg_image)
        
        # clean the resultant image
        avg_image = prepare_bone_image(avg_image)
        avg_image = clean_intensity_image(avg_image)
        avg_mask = fill(avg_image)
        avg_mask = sitk.Mask(avg_image, avg_mask)

        # Cast to consistent type
        avg_image = sitk.Cast(avg_mask, sitk.sitkUInt8)
        gc.collect()
        return avg_image
            
    except Exception as e:
        logging.error(f"Error averaging images: {e}")
        traceback.print_exc()
        return None

def apply_inverse_transform(image, transform):
    """Apply an inverse transform to create the average reference image."""
    try:
        # Create a reference grid with same size and spacing as original
        reference_size = image.GetSize()
        reference_spacing = image.GetSpacing()
        reference_origin = image.GetOrigin()
        reference_direction = image.GetDirection()
        
        # Apply transform
        resampled_image = sitk.Resample(
            image,
            reference_size,
            transform,
            sitk.sitkNearestNeighbor,
            reference_origin,
            reference_spacing,
            reference_direction,
            0.0,
            image.GetPixelID()
        )
        return resampled_image
    except Exception as e:
        logging.error(f"Error applying inverse transform: {e}")
        traceback.print_exc()
        return None

def bspline_registration(fixed_image, moving_image, control_point_grid=[2, 2, 2], scale_factor=5):
    """Perform B-spline registration."""
    try:
        logging.info("Starting B-spline registration.")

        # Convert images to float for registration
        #fixed_mask = fill(fixed_image)
        #fixed_reg = sitk.Mask(fixed_image, fixed_mask)

        #moving_mask = fill(moving_image)
        #moving_reg = sitk.Mask(moving_image, moving_mask)
        
        fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
        moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

        #gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
        #gaus_kernal = control_point_grid[0]
        #gaussian.SetSigma(gaus_kernal)
        #fixed_image = gaussian.Execute(fixed_image)
        #moving_image = gaussian.Execute(moving_image)
        
        # Calculate mesh size based on image physical dimensions
        image_physical_size = [
            size * spacing
            for size, spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())
        ]
        mesh_size = [
            int(image_size / grid_spacing + 0.5)
            for image_size, grid_spacing in zip(image_physical_size, control_point_grid)
        ]

        logging.info(f"Control Points: {mesh_size}")

        # Initialize B-spline transform
        initial_transform = sitk.BSplineTransformInitializer(
            image1=fixed_image,
            transformDomainMeshSize=mesh_size,
            order=3
        )
        
        # Set up registration method
        registration_method = sitk.ImageRegistrationMethod()

        # Add initial transformation
        registration_method.SetInitialTransformAsBSpline(
            initial_transform, 
            inPlace=True
        )
        
        # Use correlation metric for binary images
        registration_method.SetMetricAsCorrelation()
        #registration_method.SetMetricAsMeanSquares()
        #registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.15)
        #registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)

        # Use Nearest Neighbor Interpolation
        registration_method.SetInterpolator(sitk.sitkBSpline)

        #registration_method.SetOptimizerAsLBFGS2(solutionAccuracy=1e-4, numberOfIterations=100, deltaConvergenceTolerance=0.01)

        registration_method.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5,
            numberOfIterations=50,
            maximumNumberOfCorrections=5,
            maximumNumberOfFunctionEvaluations=1000,
            costFunctionConvergenceFactor=1e+7
        )

        # Execute registration
        bspline_transform = registration_method.Execute(fixed_image, moving_image)
        
        logging.info(f"B-spline Registration - Final metric value: {registration_method.GetMetricValue()}")
        
        return bspline_transform
    except Exception as e:
        logging.error(f"Error in B-spline registration: {e}")
        traceback.print_exc()
        return None

def get_bspline_from_composite(transform):
    """Extract BSpline transform from Composite Transform."""
    try:
        if isinstance(transform, sitk.BSplineTransform):
            return transform
        
        if isinstance(transform, sitk.CompositeTransform):
            for i in range(transform.GetNumberOfTransforms()):
                sub_transform = transform.GetNthTransform(i)
                if isinstance(sub_transform, sitk.BSplineTransform):
                    return sub_transform
        
        logging.error("No BSplineTransform found in the provided transform.")
        return None
    except Exception as e:
        logging.error(f"Error extracting BSpline from composite transform: {e}")
        traceback.print_exc()
        return None
        
def average_displacement_fields(displacement_fields):
    """Average displacement fields using memory mapping."""
    try:
        import tempfile
        import os
        
        # Get reference field information
        reference_field = displacement_fields[0]
        ref_array = sitk.GetArrayFromImage(reference_field)
        shape = ref_array.shape
        
        # Create a temporary memory-mapped array for accumulation
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()
        
        # Use float32 for memory efficiency during calculation
        result_array = np.memmap(temp_file.name, dtype=np.float32, mode='w+', shape=shape)
        result_array[:] = 0  # Initialize with zeros
        
        # Process each displacement field
        for i, df in enumerate(displacement_fields):
            logging.info(f"Processing displacement field {i+1}/{len(displacement_fields)}")
            arr = sitk.GetArrayFromImage(df)
            result_array += arr
            del arr
            gc.collect()
        
        # Divide by the number of fields
        result_array /= len(displacement_fields)
        
        # Convert back to SimpleITK image - CRITICAL: Convert to float64 before creating the SimpleITK image
        # Make a copy of the memmap as float64 array
        float64_array = np.array(result_array, dtype=np.float64)
        avg_field = sitk.GetImageFromArray(float64_array)
        avg_field.CopyInformation(reference_field)
        
        # Clean up
        del result_array
        os.unlink(temp_file.name)
        gc.collect()
        
        return avg_field
    except Exception as e:
        logging.error(f"Error averaging displacement fields: {e}")
        traceback.print_exc()
        return None

def bspline_inversion(bspline_transform, reference_image):
    """Invert a B-spline transform using displacement field approach."""
    try:
        logging.info("Inverting B-spline transform via displacement field conversion.")
        
        # Convert B-spline to displacement field
        displacement_field = sitk.TransformToDisplacementField(
            bspline_transform,
            sitk.sitkVectorFloat64,
            reference_image.GetSize(),
            reference_image.GetOrigin(),
            reference_image.GetSpacing(),
            reference_image.GetDirection()
        )
        
        # Invert the displacement field
        inverted_field = sitk.InvertDisplacementField(
            displacement_field,
            maximumNumberOfIterations=100,
            meanErrorToleranceThreshold=0.0001,
            maxErrorToleranceThreshold=0.01,
            enforceBoundaryCondition=True
        )
        
        # Create a displacement field transform from the inverted field
        inverted_transform = sitk.DisplacementFieldTransform(inverted_field)
        
        logging.info("B-spline transform inversion completed.")
        return inverted_transform
    except Exception as e:
        logging.error(f"Error inverting B-spline transform: {e}")
        traceback.print_exc()
        return None

def invert_displacement_field(displacement_field):
    """Invert a displacement field with more robust error handling."""
    try:
        # Check if displacement field is valid
        vector_mag_filter = sitk.VectorMagnitudeImageFilter()
        mag_image = vector_mag_filter.Execute(displacement_field)
        stats = sitk.StatisticsImageFilter()
        stats.Execute(mag_image)
        
        logging.info(f"Displacement field magnitude - Min: {stats.GetMinimum()}, Max: {stats.GetMaximum()}, Mean: {stats.GetMean()}")
        
        # Use a multi-stage approach with relaxed parameters
        try:
            inverted_field = sitk.InvertDisplacementField(
                displacement_field,
                maximumNumberOfIterations=100,
                meanErrorToleranceThreshold=0.0001,
                maxErrorToleranceThreshold=0.01,
                enforceBoundaryCondition=True
            )
        except Exception as e:
            logging.warning(f"First inversion attempt failed: {e}. Trying with more relaxed parameters...")
            
            # Even more relaxed parameters for difficult cases
            # Apply smoothing before inversion to stabilize
            displacement_components = [sitk.VectorIndexSelectionCast(displacement_field, i) for i in range(3)]
            smoothed_components = [sitk.DiscreteGaussian(comp, 1.0) for comp in displacement_components]
            
            # IMPORTANT: Use float64 components
            vector_components = [sitk.Cast(comp, sitk.sitkFloat64) for comp in smoothed_components]
            smoothed_field = sitk.Compose(vector_components)
            
            inverted_field = sitk.InvertDisplacementField(
                smoothed_field,
                maximumNumberOfIterations=150,
                meanErrorToleranceThreshold=0.01,
                maxErrorToleranceThreshold=0.05,
                enforceBoundaryCondition=False
            )
        
        # Check that inversion worked
        mag_image = vector_mag_filter.Execute(inverted_field)
        stats.Execute(mag_image)
        logging.info(f"Inverted field magnitude - Min: {stats.GetMinimum()}, Max: {stats.GetMaximum()}, Mean: {stats.GetMean()}")
        
        # IMPORTANT: Ensure inverted field uses float64 vectors
        components = [sitk.VectorIndexSelectionCast(inverted_field, i) for i in range(3)]
        float64_components = [sitk.Cast(comp, sitk.sitkFloat64) for comp in components]
        inverted_field_float64 = sitk.Compose(float64_components)
        
        # Create transform
        inverted_transform = sitk.DisplacementFieldTransform(inverted_field_float64)
        logging.info("Displacement Field inversion completed successfully.")
        return inverted_transform
    except Exception as e:
        logging.error(f"Error inverting displacement field: {e}")
        traceback.print_exc()
        # Return identity transform as a fallback
        logging.warning("Returning identity transform as fallback")
        return sitk.Transform(3, sitk.sitkIdentity)


def compute_metrics(fixed_image, moving_image, scale_factor=5):
    """Compute metrics between two images with robust error handling."""
    try:
        # Downscale
        rfix = resample_image(fixed_image, scale_factor)
        rmove = resample_image(moving_image, scale_factor)

        # Binarize to 0 and 1! This fixes the 0.4548 Dice bug and makes fill() work correctly
        rfix_bin = sitk.BinaryThreshold(rfix, lowerThreshold=1, insideValue=1, outsideValue=0)
        rmove_bin = sitk.BinaryThreshold(rmove, lowerThreshold=1, insideValue=1, outsideValue=0)

        # Fill holes and cast
        fixed_mask = sitk.Cast(fill(rfix_bin), sitk.sitkUInt8)
        moving_mask = sitk.Cast(fill(rmove_bin), sitk.sitkUInt8)

        # PREVENT HAUSDORFF CRASH: Check if either image is completely blank
        stats = sitk.StatisticsImageFilter()
        stats.Execute(fixed_mask)
        fixed_sum = stats.GetSum()

        stats.Execute(moving_mask)
        moving_sum = stats.GetSum()

        if fixed_sum == 0 or moving_sum == 0:
            logging.warning("One of the images is completely empty (pushed out of bounds).")
            return None, None, None

        # Compute Dice coefficient
        dice_filter = sitk.LabelOverlapMeasuresImageFilter()
        dice_filter.Execute(fixed_mask, moving_mask)
        dice = dice_filter.GetDiceCoefficient()  # Defaults to checking label 1
        dice = abs(dice)

        # Compute Hausdorff Distance and Mean Surface Distance
        hausdorff = sitk.HausdorffDistanceImageFilter()
        hausdorff.Execute(fixed_mask, moving_mask)
        hd = hausdorff.GetHausdorffDistance()
        msd = hausdorff.GetAverageHausdorffDistance()

        return dice, hd, msd

    except Exception as e:
        logging.error(f"Error computing metrics: {e}")
        traceback.print_exc()
        return None, None, None

def cHMA(input_dir, output_dir, reference_name="reference", scale_factor=10, max_iterations=5, cp=[2,2,2], cores='detect'):
    """
    Perform Canonical Holistic Morphometric Analysis (cHMA) on trabecular bone images.
    Modified to use ID.tiff naming convention.
    
    Parameters:
    input_dir (str): Directory containing filled and trabecular binary images
    output_dir (str): Directory to save output files
    reference_name (str): Name of the reference bone (default: "reference")
    scale_factor (int): Factor by which to reduce image resolution for faster processing (default: 3)
    max_iterations (int): Maximum number of iterations for canonical bone creation (default: 5)
    cp (list): Control point grid spacing in mm for B-spline registration [x, y, z] (default: [1, 1, 1])
    cores (int): How many cores should be used. Default is max number of cores - 3.
    
    Returns:
    bool: True if successful, False otherwise
    """
    # Configure logging
    log_file = os.path.join(output_dir, f"cHMA_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Start Logging Progress
    start_time = time.time()
    logging.info(f"Starting cHMA analysis - Input: {input_dir}, Output: {output_dir}")

    # Set the number of threads for SimpleITK
    if cores == 'detect':
        num_cores = multiprocessing.cpu_count() - 4  # Leave 4 cores free for system operations
    else:
        num_cores = cores
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(num_cores)
    logging.info(f"Using {num_cores} CPU threads for SimpleITK processing")
    
    # Create directory structure
    create_directories(output_dir)

    # Start cHMA Analysis
    try:
        
        # =====================================================================
        # Step A: Create a canonical bone
        # =====================================================================
        logging.info("=== Step A: Creating a canonical bone ===")

        # =====================================================================
        # A1: Get a list of bone names from filled images
        # =====================================================================
        
        filled_dir = os.path.join(input_dir, "Filled")
        filled_files = [f for f in os.listdir(filled_dir) if f.endswith(".tiff") or f.endswith(".tif")]
        # Extract bone names (just the ID without extension)
        bone_names = [os.path.splitext(f)[0] for f in filled_files]
        
        logging.info(f"Found {len(bone_names)} Bones: {', '.join(bone_names)}")
        
        # Ensure reference bone is in the list
        if f"{reference_name}.tiff" not in filled_files and f"{reference_name}.tif" not in filled_files:
            logging.error(f"Reference bone {reference_name} not found in input directory")
            return False

        # =====================================================================
        # A2: Initial similarity transformation
        # =====================================================================
    
        logging.info(f"A2. Performing initial similarity transformation using reference: {reference_name}")
        
        # Load reference image
        reference_image_path = os.path.join(filled_dir, f"{reference_name}.tiff")
        if not os.path.exists(reference_image_path):
            reference_image_path = os.path.join(filled_dir, f"{reference_name}.tif")
        reference_image = load_image(reference_image_path)
        
        if reference_image is None:
            logging.error("Failed to load the reference image")
            return False
        
        # Resample reference image to lower resolution for registration
        reference_image_rescaled = resample_image(reference_image, scale_factor)
        if reference_image_rescaled is None:
            logging.error("Failed to resample reference image")
            return False


        reference_avg_path = os.path.join(output_dir, "Similarity_Transform", "Average", "reference.tiff")
        reference_avg = load_image(reference_avg_path)
        
        if reference_avg is None:

            similarity_transforms = {}
            
            # Apply similarity transformation to each bone
            for bone in bone_names:
                try:
                    # Load original image
                    moving_image_path = os.path.join(filled_dir, f"{bone}.tiff")
                    if not os.path.exists(moving_image_path):
                        moving_image_path = os.path.join(filled_dir, f"{bone}.tif")
                    moving_image = load_image(moving_image_path)
                    
                    if moving_image is None:
                        logging.warning(f"Failed to load {bone}. Skipping.")
                        continue
                    
                    # Resample moving image for registration
                    moving_image_rescaled = resample_image(moving_image, scale_factor)
                    if moving_image_rescaled is None:
                        logging.warning(f"Failed to resample {bone}. Skipping.")
                        continue
    
                    logging.info(f"Starting Similarity Registration for {bone}.")
                    
                    # Register rescaled moving image to rescaled reference
                    transform = similarity_registration(reference_image_rescaled, moving_image_rescaled)
                    
                    if transform is None:
                        logging.warning(f"Similarity registration failed for {bone}. Skipping.")
                        continue
                    
                    # Save transform
                    transform_path = os.path.join(output_dir, "Similarity_Transform", "Transforms", f"{bone}.tfm")
                    sitk.WriteTransform(transform, transform_path)
                    
                    # Apply transform to ORIGINAL (non-rescaled) image
                    transformed_image = apply_transform(moving_image, transform, reference_image)
                    
                    transformed_image_path = os.path.join(output_dir, "Similarity_Transform", "Filled", f"{bone}.tiff")
                    sitk.WriteImage(transformed_image, transformed_image_path)
                    
                    # Store transform for averaging
                    similarity_transforms[bone] = transform
                    
                    logging.info(f"Similarity registered {bone} to reference")
                    
                    # Clean up
                    del moving_image, moving_image_rescaled, transformed_image
                    gc.collect()
                    
                except Exception as e:
                    logging.error(f"Error processing {bone} for similarity transform: {e}")
                    traceback.print_exc()
                    continue
            
            # =====================================================================
            # A3: Average similarity transformations
            # =====================================================================
            
            logging.info("A3. Averaging similarity transforms")
            
            if not similarity_transforms:
                logging.error("No valid similarity transforms to average")
                return False
    
            fixed_center = reference_image.TransformContinuousIndexToPhysicalPoint(
                np.array(reference_image.GetSize()) / 2.0
            )
            
            average_similarity_transform = compute_average_transform(
                list(similarity_transforms.values()), 
                fixed_center
            )
            
            if average_similarity_transform is None:
                logging.error("Failed to compute average similarity transform")
                return False
            
            # Save average transform
            avg_transform_path = os.path.join(output_dir, "Similarity_Transform", "Transforms", "average.tfm")
            sitk.WriteTransform(average_similarity_transform, avg_transform_path)
            
            # =====================================================================
            # A4: Apply inverse average transform to create similarity-transformed average image
            # =====================================================================
            
            logging.info("A4. Creating similarity-transformed average reference image")
            
            try:
                inverse_avg_transform = average_similarity_transform.GetInverse()
                
                # Apply inverse transform to ORIGINAL reference image
                reference_avg = apply_transform(reference_image, inverse_avg_transform, interpolator=sitk.sitkNearestNeighbor)
                reference_avg = prepare_bone_image(reference_avg)
                reference_avg = clean_intensity_image(reference_avg)
                
                if reference_avg is None:
                    logging.error("Failed to apply inverse transform to reference image. Using original reference.")
                    reference_avg = reference_image
    
                # Now Make sure the reference image is aligned with the similarity transform domain space
                ref_image_sim = similarity_registration(reference_image, reference_avg)
    
                reference_avg = apply_transform(reference_avg, ref_image_sim)
                
                # Update reference image for next steps
                reference_image = reference_avg
    
                reference_avg_path = os.path.join(output_dir, "Similarity_Transform", "Average", "reference.tiff")
                sitk.WriteImage(reference_avg, reference_avg_path)
    
                logging.info("Succesfully Created the Reference Image!")
                
            except Exception as e:
                logging.error(f"Error applying inverse transform: {e}")
                reference_avg = reference_image  # Fallback to original reference

        else: 
            reference_image = reference_avg
            logging.info("Succesfully Created the Reference Image!")

        # =====================================================================
        # A5: Iterative Similarity + B-spline registration to create canonical bone
        # =====================================================================
        
        logging.info("A5. Starting iterative Canonical Bone Creation")
        
        # Initialize canonical image
        canonical_image = reference_image
        prev_dice = 1
        
        # Store convergence metrics
        convergence = {
            "iteration": [],
            "dice": [],
            "hd": [],
            "msd": []
        }
        
        for iteration in range(1, max_iterations + 1):
            logging.info(f"Starting iteration {iteration} of {max_iterations}")
            
            # Resample canonical image for faster processing
            canonical_image_rescaled = resample_image(canonical_image, scale_factor)
            
            if canonical_image_rescaled is None:
                logging.error(f"Failed to resample canonical image in iteration {iteration}")
                break

            # Store bone specific metrics
            metrics = {
                "dice": [],
                "hd": [],
                "msd": []
            }
            
            # Initialize lists to store transforms and transformed images
            similarity_transforms2 = {}
            bspline_transforms = {}
            transformed_images = []
            
            # bone Iteration
            for bone in bone_names:
                try:
                    # Load original filled image
                    original_image_path = os.path.join(output_dir, "Similarity_Transform", "Filled", f"{bone}.tiff")
                    original_image = load_image(original_image_path)
                    
                    if original_image is None:
                        logging.warning(f"Failed to load {bone}. Skipping.")
                        continue
                    
                    # Resample for faster processing
                    original_image_rescaled = resample_image(original_image, scale_factor)
                    
                    if original_image_rescaled is None:
                        logging.warning(f"Failed to resample {bone}. Skipping.")
                        continue
                    #======================================================
                    # Apply similarity transformation using RESCALED images
                    similarity_transform = similarity_registration(
                        canonical_image_rescaled, 
                        original_image_rescaled
                    )
                    
                    if similarity_transform is None:
                        logging.warning(f"Similarity registration failed for {bone} in iteration {iteration}. Skipping.")
                        continue

                    #======================================================
                    # Save similarity transform
                    transform_path = os.path.join(
                        output_dir, "Similarity_Transform", "Transforms2", f"{bone}_iter{iteration}.tfm"
                    )
                    sitk.WriteTransform(similarity_transform, transform_path)
                    
                    # Apply similarity transform to ORIGINAL image for B-spline registration
                    similarity_image = apply_transform(
                        original_image, 
                        similarity_transform, 
                        canonical_image
                    )

                    similarity_image_rescaled = resample_image(similarity_image, scale_factor)
                    
                    if similarity_image_rescaled is None:
                        logging.warning(f"Failed to apply similarity transform for {bone} in iteration {iteration}. Skipping.")
                        continue
                        
                    #======================================================
                    # Apply B-spline transformation using RESCALED images

                    bspline_transform = bspline_registration(
                        canonical_image_rescaled, 
                        similarity_image_rescaled,
                        cp,
                        scale_factor
                    )
                    
                    if bspline_transform is None:
                        logging.warning(f"B-spline registration failed for {bone} in iteration {iteration}. Skipping.")
                        continue
                        
                    #======================================================
                    # Save B-spline transform
                    
                    bspline_path = os.path.join(
                        output_dir, "BSpline_Transform", "Transforms", f"{bone}_iter{iteration}.tfm"
                    )
                    sitk.WriteTransform(bspline_transform, bspline_path)
                    
                    #========================================================
                    # Apply B-Spline Transform 
                    
                    bspline_full_res = apply_transform(similarity_image, 
                                                       bspline_transform, 
                                                       canonical_image, 
                                                       interpolator=sitk.sitkNearestNeighbor
                    )
                    
                    if bspline_full_res is None:
                        logging.warning(f"Failed to apply B-spline transform to full-res image for {bone}. Skipping.")
                        continue

                    logging.info(f"Successfully B-Spline Transformed, {bone}, for iteration: {iteration}.")
                    
                    # Save transformed image
                    transformed_path = os.path.join(output_dir, "BSpline_Transform", "Filled", f"{bone}_iter{iteration}.tiff")
                    sitk.WriteImage(bspline_full_res, transformed_path)

                    #========================================================
                    # Add to lists for averaging
                    similarity_transforms2[bone] = similarity_transform
                    bspline_transforms[bone] = bspline_transform

                    #========================================================
                    # Compute metrics
                    dice, hd, msd = compute_metrics(canonical_image, bspline_full_res)
                    if dice is not None:
                        logging.info(f"Metrics for {bone}: Dice={dice:.4f}, HD={hd:.4f}, MSD={msd:.4f}")
                        metrics["dice"].append(dice)
                        metrics["hd"].append(hd)
                        metrics["msd"].append(msd)
                    
                    logging.info(f"Successfully processed {bone} in iteration {iteration}")
                    
                    # Clean up
                    del original_image, original_image_rescaled, similarity_image_rescaled
                    del similarity_image, bspline_full_res
                    gc.collect()
                    
                except Exception as e:
                    logging.error(f"Error processing {bone} in iteration {iteration}: {e}")
                    traceback.print_exc()
                    continue

            # =====================================================================
            # A6: Create the Canonical Bone
            # =====================================================================

            logging.info(f"A6. Creating Canonical Bone for Iteration {iteration}")

            # ==================================================
            # Average BSpline Transforms through displacement field conversion
            gc.collect()
            logging.info("Converting BSpline Transforms to Displacement Fields")

            try:
                # Accumulate displacement fields one by one to save RAM
                sum_displacement_field = None
                num_valid_fields = 0

                for bone, bspline_transform in bspline_transforms.items():
                    try:
                        current_field = sitk.TransformToDisplacementField(
                            bspline_transform,
                            sitk.sitkVectorFloat64,
                            canonical_image.GetSize(),
                            canonical_image.GetOrigin(),
                            canonical_image.GetSpacing(),
                            canonical_image.GetDirection()
                        )

                        # Add to running total
                        if sum_displacement_field is None:
                            sum_displacement_field = current_field
                        else:
                            sum_displacement_field = sitk.Add(sum_displacement_field, current_field)

                        num_valid_fields += 1

                        # CRITICAL: Delete the current field and force garbage collection
                        del current_field
                        gc.collect()

                    except Exception as e:
                        logging.warning(f"Error converting B-spline to displacement field for {bone}: {e}")
                        continue

                logging.info("Creating averaged inverted transform")

                if num_valid_fields > 0:
                    # ============================
                    # Average the accumulated displacement field
                    # FIX: Use Numpy to divide to bypass SimpleITK's VectorFloat64 limitation
                    sum_arr = sitk.GetArrayFromImage(sum_displacement_field)
                    avg_arr = sum_arr / float(num_valid_fields)

                    # Convert back to an image, explicitly defining it as a Vector
                    avg_displacement_field = sitk.GetImageFromArray(avg_arr, isVector=True)
                    avg_displacement_field.CopyInformation(sum_displacement_field)

                    # Clean up the massive arrays to free up memory
                    del sum_displacement_field, sum_arr, avg_arr
                    gc.collect()

                    # ============================
                    # Invert with better parameters
                    inverted_transform = invert_displacement_field(avg_displacement_field)
                    logging.info("Successfully created averaged inverted transform")

                    # Clean up the average field
                    del avg_displacement_field
                    gc.collect()

                    # ============================
                    # Apply transform with extra safeguards
                    logging.info("Creating canonical image")
                    inverted_image = apply_transform(canonical_image,
                                                     inverted_transform,
                                                     interpolator=sitk.sitkNearestNeighbor)
                    # ============================
                    # Verify inverted image has content
                    stats = sitk.StatisticsImageFilter()
                    stats.Execute(inverted_image)
                    if stats.GetSum() < 1:
                        logging.warning("Inverted image is empty! Using previous canonical image instead.")
                        new_canonical_image = canonical_image  # FIX: Updated from averaged_image
                    else:
                        new_canonical_image = inverted_image
                        logging.info("Successfully created the canonical image")
                else:
                    logging.error("No valid displacement fields for inversion. Stopping analysis.")
                    break

            except Exception as e:
                logging.warning(f"Error in displacement field processing: {e}")
                logging.warning("Using previous canonical image as fallback")
                new_canonical_image = canonical_image  # FIX: Updated from averaged_image

            logging.info(f"Successfully created canonical bone image for iteration {iteration}")

            # =====================================================================
            # A7: Calculate Convergence Metrics
            # =====================================================================

            logging.info(f"A7. Calculating Iteration {iteration} Metrics")

            # Calculate averages if we have values
            avg_dice = np.mean(metrics["dice"])
            avg_hd = np.mean(metrics["hd"])
            avg_msd = np.mean(metrics["msd"])
            
            logging.info(f"Iteration {iteration} average metrics - Dice: {avg_dice:.4f}, HD: {avg_hd:.4f}, MSD: {avg_msd:.4f}")

            if iteration == 1:
                convergence["iteration"].append(iteration)
                convergence["dice"].append(avg_dice)
                convergence["hd"].append(avg_hd)
                convergence["msd"].append(avg_msd)
                
                dice = avg_dice
                hd = avg_hd
                msd = avg_msd
                prev_dice = avg_dice
                previous_canonical_image = canonical_image
            else:
                # Normal metrics calculation between consecutive canonical images
                dice, hd, msd = compute_metrics(canonical_image, new_canonical_image)
                
                # Store metrics
                convergence["iteration"].append(iteration)
                convergence["dice"].append(dice)
                convergence["hd"].append(hd)
                convergence["msd"].append(msd)

            logging.info(f"Iteration {iteration} metrics - Dice: {dice:.4f}, HD: {hd:.4f}, MSD: {msd:.4f}")
                
            # Check convergence criteria
            if dice is not None:
                if dice <= prev_dice - 0.05: 
                    logging.info(f"Dice score decreased significantly ({prev_dice:.4f} to {dice:.4f}). Stopping iterations.")
                    # Use the previous canonical image instead of the new one that caused problems
                    canonical_image = previous_canonical_image
                    break
                    
                if dice > 0.95: 
                    logging.info(f"Dice score above threshold ({dice:.4f} > 0.95). Stopping iterations.")
                    canonical_image = new_canonical_image
                    break
                    
                prev_dice = dice
            else:
                # If dice is None (metrics calculation failed)
                logging.warning("Metrics calculation failed. Stopping iterations.")
                # Use previous iteration's canonical image
                canonical_image = previous_canonical_image
                break
            
            # Save previous canonical image before updating
            previous_canonical_image = canonical_image 
            
            # Update canonical image for next iteration
            canonical_image = new_canonical_image
            canonical_image = sitk.Cast(new_canonical_image, sitk.sitkUInt8)
            
            # Save canonical image for this iteration
            canonical_path = os.path.join(output_dir, "Canonical_Bone", f"canonical_iter{iteration}.tiff")
            sitk.WriteImage(canonical_image, canonical_path)
        
        # Save final canonical image
        final_canonical_path = os.path.join(output_dir, "Canonical_Bone", "canonical.tiff")
        sitk.WriteImage(canonical_image, final_canonical_path)
        
        # Save convergence metrics
        metrics_df = pd.DataFrame(convergence)
        metrics_path = os.path.join(output_dir, "Canonical_Bone", "convergence_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        
        logging.info(f"Canonical bone creation completed after {len(convergence['iteration'])} iterations")
        
        # =====================================================================
        # Step B: Create an averaged trabecular image for meshing
        # =====================================================================
        logging.info("=== Step B: Creating canonical trabecular image ===")
        
        # B1. Transform trabecular images to canonical space
        logging.info("B1. Transforming trabecular images to canonical space")

        filled_dir = os.path.join(input_dir, "Filled")
        filled_files = [f for f in os.listdir(filled_dir) if f.endswith(".tiff") or f.endswith(".tif")]
        bone_names = [os.path.splitext(f)[0] for f in filled_files]
        
        reference_image_path = os.path.join(filled_dir, f"{reference_name}.tiff")
        if not os.path.exists(reference_image_path):
            reference_image_path = os.path.join(filled_dir, f"{reference_name}.tif")
        reference_image = load_image(reference_image_path)
                
        trabecular_dir = os.path.join(input_dir, "Trabecular")
        transformed_trabecular_images = []
        
        for bone in bone_names:
            # Load trabecular image
            trabecular_path = os.path.join(trabecular_dir, f"{bone}.tiff")
            if not os.path.exists(trabecular_path):
                trabecular_path = os.path.join(trabecular_dir, f"{bone}.tif")
            trabecular_image = load_image(trabecular_path)
        
            # Load initial similarity transform to align bones
            first_sim_path = os.path.join(
                output_dir, "Similarity_Transform", "Transforms",
                f"{bone}.tfm"
            )
            
            # Load final transforms from last iteration
            last_iteration = len(convergence["iteration"])
            similarity_path = os.path.join(
                output_dir, "Similarity_Transform", "Transforms2", 
                f"{bone}_iter{last_iteration}.tfm"
            )
            bspline_path = os.path.join(
                output_dir, "BSpline_Transform", "Transforms", 
                f"{bone}_iter{last_iteration}.tfm"
            )
        
            # Read transform paths
            first_sim_transform = sitk.ReadTransform(first_sim_path)
            similarity_transform = sitk.ReadTransform(similarity_path)
            bspline_transform = sitk.ReadTransform(bspline_path)
        
            #=========================================================
            # Apply transforms
            
            # First similarity
            trabecular_first = apply_transform(trabecular_image, first_sim_transform, reference_image)
            
            # Second similarity
            trabecular_similarity = apply_transform(trabecular_first, similarity_transform, canonical_image)
        
            tspath = os.path.join(output_dir, "Similarity_Transform", "Trabecular", f"{bone}.tiff")
            sitk.WriteImage(trabecular_similarity, tspath)
            
            # Then B-spline
            trabecular_bspline = apply_transform(trabecular_similarity, bspline_transform, canonical_image)
        
            # Save transformed trabecular image
            transformed_path = os.path.join(output_dir, "BSpline_Transform", "Trabecular", f"{bone}.tiff")
            sitk.WriteImage(trabecular_bspline, transformed_path)
            
            # Add to list for averaging
            transformed_trabecular_images.append(trabecular_bspline)
            
            # Clean up
            del trabecular_image, trabecular_similarity, trabecular_bspline
            gc.collect()
        
        del reference_image
        
        # B2. Average transformed trabecular images with proper binary handling
        if not transformed_trabecular_images:
            logging.error("No valid transformed trabecular images to average")
            return False
            
        # Use majority voting for binary images
        averaged_trabecular = None
        averaged_trabecular = average_images(transformed_trabecular_images)
        
        del transformed_trabecular_images
        
        if averaged_trabecular is None:
            logging.error("Failed to average transformed trabecular images")
            return False
            
        # Save canonical trabecular image
        trabecular_canonical_path = os.path.join(output_dir, "Canonical_Bone", "trabecular.tiff")
        sitk.WriteImage(averaged_trabecular, trabecular_canonical_path)

        final_canonical_path = os.path.join(output_dir, "Canonical_Bone", "canonical.tiff")
        averaged_trabecular = canonical_trabecular(final_canonical_path, trabecular_canonical_path, trabecular_canonical_path)
        
        # Calculate total runtime
        end_time = time.time()
        total_time = end_time - start_time
        logging.info(f"cHMA analysis completed in {total_time/60:.2f} minutes")
        logging.info(f"cHMA Analysis Took {len(convergence["iteration"])} Iterations to reach convergence.")
        gc.collect()
        
        return True
        
    except Exception as e:
        logging.error(f"Unexpected error in cHMA function: {e}")
        traceback.print_exc()
        return False


####################
# Holistic Morphometric Analysis - Creation of the Canonical and Isotopological Meshes with Scalar Values
####################

"""
Holistic Morphometric Analysis with Isotopological Meshes

This module implements the cHMA methodology for trabecular bone analysis after 
canonical bone generation as described in Bachmann et al. 2022.

The workflow consists of two main steps:
1. Step A: Create or load a tetrahedral canonical mesh from the canonical trabecular image
2. Step B: Create isotopological meshes from canonical mesh
3. Step C: Perform HMA analysis on the isotopological meshes
"""


def create_directories1(output_dir):
    """Create the required directory structure for the cHMA workflow."""
    try:
        os.makedirs(os.path.join(output_dir, "Canonical_Bone"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "Trabecular"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "Isotopological_Meshes"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "HMA_Results"), exist_ok=True)

        logging.info(f"Created directory structure in {output_dir}")
    except Exception as e:
        logging.error(f"Error creating directory structure: {e}")
        traceback.print_exc()
        raise


def create_bone_presence_mask(trabecular_image, mesh):
    """Generate a mask showing where bone is present vs. absent"""
    try:
        # Get image properties
        spacing = trabecular_image.GetSpacing()
        origin = trabecular_image.GetOrigin()
        size = trabecular_image.GetSize()

        # Get binary image as numpy array (Z,Y,X order)
        array = sitk.GetArrayFromImage(trabecular_image)

        # Create dilated version to include nearby regions
        from scipy import ndimage
        struct = ndimage.generate_binary_structure(3, 1)
        dilated = ndimage.binary_dilation(array, structure=struct, iterations=2)

        # Create mask matching the mesh dimensions
        mask = np.zeros(len(mesh['vertices']), dtype=bool)

        # Process in batches for efficiency
        batch_size = 10000
        total_batches = (len(mesh['vertices']) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(mesh['vertices']))

            # For each vertex in batch
            for i in range(start_idx, end_idx):
                vertex = mesh['vertices'][i]

                try:
                    # Convert physical coordinates to image indices
                    idx = trabecular_image.TransformPhysicalPointToIndex(vertex)

                    # Check if within image bounds
                    if (0 <= idx[0] < size[0] and
                            0 <= idx[1] < size[1] and
                            0 <= idx[2] < size[2]):
                        # Z,Y,X order in numpy array
                        mask[i] = dilated[idx[2], idx[1], idx[0]]
                except Exception as e:
                    # Skip vertices that can't be mapped to image indices
                    continue

        return mask

    except Exception as e:
        logging.error(f"Error creating bone presence mask: {e}")
        traceback.print_exc()
        # Return all True as fallback
        return np.ones(len(mesh['vertices']), dtype=bool)


def calculate_confidence_scores(mesh, trabecular_image, grid_points, grid_values):
    """Calculate confidence scores for each vertex based on distance to valid samples"""
    try:
        logging.info("Calculating confidence scores for mesh vertices...")
        confidence = np.zeros(len(mesh['vertices']))

        # Create KD-tree for grid points
        from scipy.spatial import cKDTree
        tree = cKDTree(grid_points)

        # Process in batches
        batch_size = 10000
        total_batches = (len(mesh['vertices']) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(mesh['vertices']))

            # Get vertices in this batch
            batch_vertices = mesh['vertices'][start_idx:end_idx]

            # Find nearest grid points
            k = 8  # Number of nearest neighbors to consider
            distances, indices = tree.query(batch_vertices, k=k)

            # For each vertex
            for i in range(end_idx - start_idx):
                # Check BV/TV values at nearest grid points
                nearby_bvtv = grid_values['bv_tv'][indices[i]]

                # Calculate distance-weighted average of BV/TV
                weights = 1.0 / (distances[i] + 1e-10)
                weights = weights / np.sum(weights)

                # Count valid sampling points (those with meaningful BV/TV)
                valid_points = np.sum(nearby_bvtv > 0.01)

                # Calculate confidence based on:
                # 1. Number of valid sampling points
                # 2. Inverse distance to those points
                # 3. Mean BV/TV value (higher BV/TV = more confidence)
                if valid_points > 0:
                    mean_bvtv = np.sum(weights * nearby_bvtv)
                    confidence[start_idx + i] = (valid_points / k) * (1.0 - np.mean(distances[i]) / 10.0) * min(1.0,
                                                                                                                mean_bvtv * 5)
                else:
                    confidence[start_idx + i] = 0.0

        # Normalize confidence to [0, 1]
        max_conf = np.max(confidence)
        if max_conf > 0:
            confidence = confidence / max_conf

        logging.info(f"Calculated confidence scores: mean = {np.mean(confidence):.4f}")
        return confidence

    except Exception as e:
        logging.error(f"Error calculating confidence scores: {e}")
        traceback.print_exc()
        # Return all ones as fallback
        return np.ones(len(mesh['vertices']))


def apply_transform(image, transform, reference_image=None, interpolator=sitk.sitkNearestNeighbor):
    """Apply transform while preserving image content and structure."""
    try:
        # Verify image has content
        image = cleanup(image)
        stats = sitk.StatisticsImageFilter()
        stats.Execute(image)

        if reference_image is not None:
            reference_image = cleanup(reference_image)
        if stats.GetSum() < 1:
            logging.warning("Cannot transform empty image")
            return image

        # Apply transform
        if reference_image is None:
            transformed = sitk.Resample(
                image,
                image,
                transform,
                interpolator,
                0.0,
                image.GetPixelID()
            )
        else:
            transformed = sitk.Resample(
                image,
                reference_image,
                transform,
                interpolator,
                0.0,
                image.GetPixelID()
            )
        transformed = cleanup(transformed)

        return transformed

    except Exception as e:
        logging.error(f"Error applying transform: {e}")
        traceback.print_exc()
        return image


def find_transforms_for_bone(input_dir, bone_name, iteration):
    """
    Finds the B-spline transform for a given bone with fixed variable tracking.
    """
    try:
        possible_patterns = [
            f"{bone_name}",
            f"{bone_name.split('_')[0]}",
            f"{bone_name.replace('.tiff', '')}"
        ]

        bspline_tfm_path = None

        # Try to find B-spline transform
        bspline_dir = os.path.join(input_dir, "BSpline_Transform", "Transforms")
        for pattern in possible_patterns:
            for suffix in [f"_iter{iteration}.tfm", ".tfm"]:
                transform_path = os.path.join(bspline_dir, f"{pattern}{suffix}")
                if os.path.exists(transform_path):
                    bspline_tfm_path = transform_path
                    break
            if bspline_tfm_path:
                break

        if bspline_tfm_path:
            return sitk.ReadTransform(bspline_tfm_path)
        else:
            logging.error(f"Could not find B-spline transform for {bone_name}")
            return None

    except Exception as e:
        logging.error(f"Error finding transforms for {bone_name}: {e}")
        traceback.print_exc()
        return None

def find_transforms_for_bone2(input_dir, bone_name, iteration):
    """
    Find transform files for a given bone with flexible naming pattern matching.

    Parameters:
    -----------
    output_dir : str
        Base output directory
    bone_name : str
        Name of the bone to find transforms for
    iteration : int
        Number of the last iteration

    Returns:
    --------
    tuple or None
        (similarity_transform, bspline_transform) or None if not found
    """
    try:
        # Possible name patterns for the bone in transform files
        possible_patterns = [
            f"{bone_name}",  # Basic pattern
            f"{bone_name.split('_')[0]}",  # First part before underscore
            f"{bone_name.replace('.tiff', '')}",  # Remove .tiff
            f"{bone_name.replace('_trabecular_resampled', '')}"  # Remove _trabecular_resampled
        ]

        similarity_transform = None
        bspline_transform = None

        # Find the initial similarity transform
        sim_dir = os.path.join(input_dir, "Similarity_Transform", "Transforms")
        for pattern in possible_patterns:
            for suffix in ["", ".tfm", f"_iter{iteration}", f"_iter{iteration}.tfm"]:
                transform_path = os.path.join(sim_dir, f"{pattern}{suffix}")
                if os.path.exists(transform_path):
                    sim_tfm = transform_path
                    break
            if sim_tfm:
                break

        # Try to find the iterated similarity
        sim_dir_iter = os.path.join(input_dir, "Similarity_Transform", "Transforms2")
        for pattern in possible_patterns:
            for suffix in ["", ".tfm", f"_iter{iteration}", f"_iter{iteration}.tfm"]:
                transform_path = os.path.join(sim_dir_iter, f"{pattern}{suffix}")
                if os.path.exists(transform_path):
                    sim_iter_tfm = transform_path
                    break
            if similarity_transform:
                break

        # Try to find B-spline transform
        bspline_dir = os.path.join(input_dir, "BSpline_Transform", "Transforms")
        for pattern in possible_patterns:
            for suffix in ["", ".tfm", f"_iter{iteration}", f"_iter{iteration}.tfm"]:
                transform_path = os.path.join(bspline_dir, f"{pattern}{suffix}")
                if os.path.exists(transform_path):
                    bspline_tfm = transform_path
                    break
            if bspline_transform:
                break

        sim_tfm = sitk.ReadTransform(sim_tfm)
        sim_iter_tfm = sitk.ReadTransform(sim_iter_tfm)
        bspline_tfm = sitk.ReadTransform(bspline_tfm)

        return bspline_tfm

    except Exception as e:
        logging.error(f"Error finding transforms for {bone_name}: {e}")
        traceback.print_exc()
        return None


def load_tetgen(node_path, ele_path):
    """
    Load a tetrahedral mesh directly from TetGen .node and .ele files.

    Parameters:
    -----------
    node_path : str
        Path to the .node file containing vertex coordinates
    ele_path : str
        Path to the .ele file containing tetrahedral elements

    Returns:
    --------
    dict or None
        Mesh dictionary containing vertices and tetrahedra
    """
    try:
        logging.info(f"Loading TetGen mesh from {node_path} and {ele_path}")

        # Check if files exist
        if not os.path.exists(node_path) or not os.path.exists(ele_path):
            logging.error("Node or element file not found")
            return None

        # Read vertices from .node file
        with open(node_path, 'r') as f:
            lines = f.readlines()

        # Parse header
        header = lines[0].strip().split()
        num_points = int(header[0])
        dimension = int(header[1])
        num_attributes = int(header[2])
        has_boundary = int(header[3])

        # Read vertices
        vertices = []
        for i in range(1, num_points + 1):
            line = lines[i].strip().split()
            # Format: index x y z [attributes] [boundary marker]
            vertices.append([float(line[1]), float(line[2]), float(line[3])])

        # Read elements from .ele file
        with open(ele_path, 'r') as f:
            lines = f.readlines()

        # Parse header
        header = lines[0].strip().split()
        num_tets = int(header[0])
        nodes_per_tet = int(header[1])
        has_region = int(header[2]) if len(header) > 2 else 0

        # Read tetrahedra
        tetrahedra = []
        for i in range(1, num_tets + 1):
            line = lines[i].strip().split()
            # Format: index n1 n2 n3 n4 [region]
            # Note: TetGen is 1-indexed, need to convert to 0-indexed
            tetrahedra.append([
                int(line[1]) - 1,
                int(line[2]) - 1,
                int(line[3]) - 1,
                int(line[4]) - 1
            ])

        logging.info(f"Loaded TetGen mesh with {len(vertices)} vertices and {len(tetrahedra)} tetrahedra")

        # Create mesh dictionary
        mesh_dict = {
            "vertices": np.array(vertices),
            "tetrahedra": np.array(tetrahedra),
            "surface_triangles": np.array([])  # Surface triangles would require reading .face file
        }

        return mesh_dict

    except Exception as e:
        logging.error(f"Error loading TetGen mesh: {e}")
        traceback.print_exc()
        return None


def load_netgen(filepath):
    """
    Load a tetrahedral mesh from a Netgen .vol file with format-specific parsing.

    Parameters:
    -----------
    filepath : str
        Path to the .vol file

    Returns:
    --------
    dict or None
        Mesh dictionary containing vertices and tetrahedra
    """
    try:
        logging.info(f"Loading Netgen .vol mesh from {filepath}")

        # Check if file exists
        if not os.path.exists(filepath):
            logging.error(f"File {filepath} not found")
            return None

        # Parse the VOL file
        vertices = []
        tetrahedra = []

        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Process the file line by line
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Look for the points section
            if line == "points":
                i += 1  # Move to the next line (number of points)
                if i < len(lines):
                    try:
                        num_points = int(lines[i].strip())
                        logging.info(f"Found points section with {num_points} vertices")
                        i += 1  # Move to the first point

                        # Read all points
                        for j in range(num_points):
                            if i + j < len(lines):
                                try:
                                    # Strip out any text and get the coordinates
                                    coords_line = lines[i + j].strip()
                                    # Split by whitespace and convert to float
                                    parts = [float(x) for x in coords_line.split()]
                                    if len(parts) >= 3:
                                        vertices.append(parts[:3])
                                except Exception as e:
                                    logging.warning(f"Error parsing vertex at line {i + j}: {e}")

                        i += num_points  # Skip past all the vertices
                        continue
                    except ValueError:
                        logging.warning(f"Could not parse number of points: {lines[i].strip()}")

            # Look for the volumeelements section
            elif line == "volumeelements":
                i += 1  # Move to the next line (number of elements)
                if i < len(lines):
                    try:
                        num_elements = int(lines[i].strip())
                        logging.info(f"Found volumeelements section with {num_elements} elements")
                        i += 1  # Move to the first volume element

                        # Read all volume elements
                        for j in range(num_elements):
                            if i + j < len(lines):
                                try:
                                    # Parse the element line
                                    parts = [int(x) for x in lines[i + j].strip().split()]

                                    # Netgen volumeelements format is usually:
                                    # element_type material domain [vertices]
                                    # element_type 1 is typically a tetrahedron with 4 vertices
                                    if len(parts) >= 5 and parts[0] == 1:
                                        # Get the last 4 elements as vertex indices (1-indexed)
                                        tet_indices = parts[-4:]
                                        # Convert to 0-indexed for our mesh format
                                        tetrahedra.append([idx - 1 for idx in tet_indices])
                                except Exception as e:
                                    logging.warning(f"Error parsing tetrahedron at line {i + j}: {e}")

                        i += num_elements  # Skip past all the elements
                        continue
                    except ValueError:
                        logging.warning(f"Could not parse number of volume elements: {lines[i].strip()}")

            i += 1  # Move to the next line

        # Verify we have data
        if not vertices:
            logging.error("No vertices found in the file")
            return None

        if not tetrahedra:
            logging.error("No tetrahedra found in the file")
            return None

        logging.info(f"Successfully loaded Netgen mesh with {len(vertices)} vertices and {len(tetrahedra)} tetrahedra")

        mesh_dict = {
            "vertices": np.array(vertices),
            "tetrahedra": np.array(tetrahedra),
            "surface_triangles": np.array([])  # Surface triangles could be extracted later if needed
        }

        return mesh_dict

    except Exception as e:
        logging.error(f"Error loading Netgen .vol mesh: {e}")
        traceback.print_exc()
        return None


def load_gmsh(filepath):
    """
    Load a tetrahedral mesh from a Gmsh .msh file.

    Parameters:
    -----------
    filepath : str
        Path to the .msh file

    Returns:
    --------
    dict or None
        Mesh dictionary containing vertices and tetrahedra
    """
    try:
        logging.info(f"Loading Gmsh .msh mesh from {filepath}")

        # Check if file exists
        if not os.path.exists(filepath):
            logging.error(f"File {filepath} not found")
            return None

        # Parse the MSH file
        vertices = []
        tetrahedra = []

        with open(filepath, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Read nodes
            if line == "$Nodes":
                i += 1
                num_vertices = int(lines[i].strip())
                i += 1

                for j in range(num_vertices):
                    if i + j < len(lines):
                        parts = lines[i + j].strip().split()
                        if len(parts) >= 4:
                            # Format: node_idx x y z
                            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

                i += num_vertices

            # Read elements
            elif line == "$Elements":
                i += 1
                num_elements = int(lines[i].strip())
                i += 1

                for j in range(num_elements):
                    if i + j < len(lines):
                        parts = lines[i + j].strip().split()
                        if len(parts) > 4:
                            element_type = int(parts[1])
                            # Element type 4 is a tetrahedron
                            if element_type == 4:
                                # Convert 1-indexed to 0-indexed
                                # Format varies based on MSH version, but last 4 values are always vertices
                                tet_indices = [int(parts[-4]) - 1, int(parts[-3]) - 1,
                                               int(parts[-2]) - 1, int(parts[-1]) - 1]
                                tetrahedra.append(tet_indices)

                i += num_elements

            else:
                i += 1

        logging.info(f"Loaded Gmsh .msh mesh with {len(vertices)} vertices and {len(tetrahedra)} tetrahedra")

        mesh_dict = {
            "vertices": np.array(vertices),
            "tetrahedra": np.array(tetrahedra),
            "surface_triangles": np.array([])  # Surface triangles would need to be extracted
        }

        return mesh_dict

    except Exception as e:
        logging.error(f"Error loading Gmsh .msh mesh: {e}")
        traceback.print_exc()
        return None


def load_vtk(filepath):
    """
    Load an existing VTK mesh file with better handling for quadratic tetrahedra.

    Parameters:
    -----------
    filepath : str
        Path to the VTK file

    Returns:
    --------
    dict or None
        Mesh dictionary containing vertices, tetrahedra, and surface triangles
    """
    try:
        logging.info(f"Attempting to load existing mesh from {filepath}")

        # Check if file exists
        if not os.path.exists(filepath):
            logging.info(f"No existing mesh found at {filepath}")
            return None

        # Try with generic VTK file reader first
        reader = vtk.vtkGenericDataObjectReader()
        reader.SetFileName(filepath)
        reader.Update()

        # Check if reader was successful
        if not reader.IsFileValid(filepath):
            logging.warning(f"File {filepath} is not a valid VTK file")
            return None

        # Determine what type of data we have
        data = reader.GetOutput()
        data_type = data.GetDataObjectType()

        # Use appropriate reader based on data type
        if data_type == vtk.VTK_POLY_DATA:
            logging.info(f"Detected PolyData in {filepath}")
            reader = vtk.vtkPolyDataReader()
        elif data_type == vtk.VTK_UNSTRUCTURED_GRID:
            logging.info(f"Detected UnstructuredGrid in {filepath}")
            reader = vtk.vtkUnstructuredGridReader()
        else:
            logging.warning(f"Unsupported VTK data type: {data_type}")
            return None

        reader.SetFileName(filepath)
        reader.Update()
        mesh = reader.GetOutput()

        # Log cell information
        cell_types = {}
        for i in range(mesh.GetNumberOfCells()):
            cell_type = mesh.GetCellType(i)
            if cell_type in cell_types:
                cell_types[cell_type] += 1
            else:
                cell_types[cell_type] = 1

        logging.info(f"Cell types found: {cell_types}")

        # Extract points
        points = []
        for i in range(mesh.GetNumberOfPoints()):
            points.append(list(mesh.GetPoint(i)))
        points = np.array(points)

        # Extract tetrahedra
        tetrahedra = []
        triangles = []

        # Handle different cell types
        for i in range(mesh.GetNumberOfCells()):
            cell = mesh.GetCell(i)
            cell_type = cell.GetCellType()

            if cell_type == vtk.VTK_TETRA:
                # Linear tetrahedron
                tetrahedra.append([cell.GetPointId(j) for j in range(4)])
            elif cell_type == 24:  # VTK_QUADRATIC_TETRA
                # Quadratic tetrahedron - extract just the corner vertices (first 4 points)
                tetrahedra.append([cell.GetPointId(j) for j in range(4)])
            elif cell_type == vtk.VTK_TRIANGLE:
                triangles.append([cell.GetPointId(j) for j in range(3)])

        # Log what we found
        logging.info(f"Found {len(points)} vertices, {len(tetrahedra)} tetrahedra, {len(triangles)} triangles")

        # Check if we have tetrahedra
        if len(tetrahedra) == 0:
            logging.warning("No tetrahedra found. This mesh may not be suitable for analysis.")

            # Try to convert from surface if we have triangles
            if len(triangles) > 0:
                logging.info("Attempting to generate tetrahedra from surface triangles...")

                # Create a clean polydata surface
                surface = vtk.vtkPolyData()
                vtkPoints = vtk.vtkPoints()
                for point in points:
                    vtkPoints.InsertNextPoint(point)
                surface.SetPoints(vtkPoints)

                # Add triangles
                vtkCells = vtk.vtkCellArray()
                for tri in triangles:
                    vtkTriangle = vtk.vtkTriangle()
                    for j in range(3):
                        vtkTriangle.GetPointIds().SetId(j, tri[j])
                    vtkCells.InsertNextCell(vtkTriangle)
                surface.SetPolys(vtkCells)

                # Generate tetrahedral mesh using Delaunay3D
                delaunay = vtk.vtkDelaunay3D()
                delaunay.SetInputData(surface)
                delaunay.SetTolerance(0.001)
                delaunay.SetOffset(5.0)  # Larger offset for better results
                delaunay.Update()

                # Extract tetrahedra
                tetMesh = delaunay.GetOutput()
                tetrahedra = []
                for i in range(tetMesh.GetNumberOfCells()):
                    cell = tetMesh.GetCell(i)
                    if cell.GetCellType() == vtk.VTK_TETRA:
                        tetrahedra.append([cell.GetPointId(j) for j in range(4)])

                logging.info(f"Generated {len(tetrahedra)} tetrahedra from surface")

        # Create the mesh dictionary
        mesh_dict = {
            "vertices": points,
            "tetrahedra": np.array(tetrahedra),
            "surface_triangles": np.array(triangles)
        }

        # Verify mesh quality
        if len(tetrahedra) > 0:
            from collections import Counter
            # Check for duplicate tetrahedra
            tet_counter = Counter([tuple(sorted(tet)) for tet in tetrahedra])
            duplicates = [item for item, count in tet_counter.items() if count > 1]
            if duplicates:
                logging.warning(f"Found {len(duplicates)} duplicate tetrahedra")

            # Check for inverted elements
            negative_volume = 0
            for tet in tetrahedra:
                v0 = points[tet[0]]
                v1 = points[tet[1]]
                v2 = points[tet[2]]
                v3 = points[tet[3]]

                e1 = np.array(v1) - np.array(v0)
                e2 = np.array(v2) - np.array(v0)
                e3 = np.array(v3) - np.array(v0)

                volume = np.dot(np.cross(e1, e2), e3) / 6.0
                if volume <= 0:
                    negative_volume += 1

            if negative_volume > 0:
                logging.warning(f"Found {negative_volume} tetrahedra with negative or zero volume")

            logging.info(f"Loaded mesh with {len(points)} vertices and {len(tetrahedra)} tetrahedra")

        return mesh_dict

    except Exception as e:
        logging.error(f"Error loading VTK mesh: {e}")
        traceback.print_exc()
        return None


def create_solid_tetrahedral_mesh2(image, method="tetgen", edge_length=1.0, output_dir=None):
    import pymesh

    spacing = np.array(image.GetSpacing())
    origin = np.array(image.GetOrigin())
    size = np.array(image.GetSize())

    image_array = sitk.GetArrayFromImage(image)

    # Generate mesh with TetGen or another tool
    # Example using pymesh
    mesh = pymesh.generate_tet_mesh(image_array)

    # Mesh vertices and tetrahedra
    vertices = mesh.vertices
    tetrahedra = mesh.elements
    # Convert voxel coordinates to physical coordinates
    vertices = vertices * spacing + origin

    return {
        "vertices": vertices,
        "tetrahedra": tetrahedra
    }


def create_solid_tetrahedral_mesh(image, method="tetgen", edge_length=1.0, output_dir=None):
    """
    Create a solid tetrahedral mesh from a binary trabecular mask using various meshing methods.

    Parameters:
    -----------
    image : sitk.Image
        Input trabecular bone mask (binary image)
    method : str
        Meshing method to use: "tetgen", "netgen", or "gmsh"
    edge_length : float
        Target edge length for mesh in mm (smaller = finer mesh)
    output_dir : str
        Directory to save intermediate results

    Returns:
    --------
    dict or None
        Dictionary containing mesh data or None if meshing fails
    """
    try:

        # Get image properties
        spacing = image.GetSpacing()
        origin = image.GetOrigin()

        # Convert to numpy array for processing
        np_array = sitk.GetArrayFromImage(image)

        np_array = ndimage.binary_fill_holes(np_array)

        # smoothed = ndimage.gaussian_filter(filled_array.astype(float), sigma=0.5)

        np_array = np.transpose(np_array, (2, 1, 0))  # Fix axis orientation

        # Extract isosurface using marching cubes
        from skimage import measure
        verts, faces, normals, _ = measure.marching_cubes(
            volume=np_array,
            level=0.5,
            spacing=spacing,
            allow_degenerate=False
        )

        # Adjust vertices for origin offset
        verts = verts + np.array(origin)

        # Create tetrahedral mesh using selected method
        if method == "tetgen":
            import tetgen

            # Create TetGen object
            tgen = tetgen.TetGen(verts, faces)

            # Parameters for solid mesh based on Bachmann et al. (2022)
            nodes, elements = tgen.tetrahedralize(
                quality=True,
                plc=True,  # Preserve input geometry
                minratio=1.2,  # Quality constraint
                mindihedral=25.0,  # Minimum angle
                maxvolume=edge_length ** 3,  # Control element size
                steinerleft=100000,  # Allow sufficient refinement points
                nobisect=False  # Allow boundary refinement
            )

            mesh_dict = {
                "vertices": nodes,
                "tetrahedra": elements,
                "surface_triangles": faces
            }

        elif method == "netgen":
            # Try to use NGLib if available
            try:
                import nglib

                # Initialize NGLib
                nglib.Init()

                # Create a temporary STL file for NetGen
                import tempfile
                import stl
                from stl import mesh as stl_mesh

                with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
                    stl_file = tmp.name

                # Save surface as STL
                surface = stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype))
                for i, f in enumerate(faces):
                    for j in range(3):
                        surface.vectors[i][j] = verts[f[j], :]
                surface.save(stl_file)

                # Initialize NetGen mesh
                mesh = nglib.Mesh()

                # Load STL file
                mesh.Import(stl_file)

                # Set meshing parameters for solid volume
                mp = nglib.MeshingParameters()
                mp.maxh = edge_length  # Max element size
                mp.minh = edge_length / 5  # Min element size
                mp.fineness = 0.5  # Medium mesh refinement
                mp.secondorder = 0  # Linear elements

                # Generate volume mesh
                mesh.GenerateVolumeMesh(mp)

                # Extract mesh data
                vertices = []
                tetrahedra = []

                # Extract vertices
                for i in range(1, mesh.GetNP() + 1):
                    p = mesh.GetPoint(i)
                    vertices.append([p.p[0], p.p[1], p.p[2]])

                # Extract tetrahedra
                for i in range(1, mesh.GetNE() + 1):
                    el = mesh.GetElement(i)
                    if el.GetType() == 1:  # Tetrahedral elements
                        tetrahedra.append([el.GetVertex(j) - 1 for j in range(4)])  # 0-indexed

                mesh_dict = {
                    "vertices": np.array(vertices),
                    "tetrahedra": np.array(tetrahedra),
                    "surface_triangles": faces
                }

                # Clean up temporary file
                os.unlink(stl_file)

            except ImportError:
                logging.warning("NGLib not available. Falling back to TetGen.")

                # Fall back to TetGen
                import tetgen
                tgen = tetgen.TetGen(verts, faces)
                nodes, elements = tgen.tetrahedralize(
                    quality=True,
                    plc=True,  # Preserve input geometry
                    minratio=1.2,  # Allows slightly varying edge lengths for better volume fitting
                    mindihedral=25.0,  # Standard high-quality angle (prevents over-constraining)
                    maxvolume=edge_length ** 3,  # Control element size
                    steinerleft=100000,  # Allow sufficient refinement points
                    nobisect=False  # Allow boundary refinement
                )

                mesh_dict = {
                    "vertices": nodes,
                    "tetrahedra": elements,
                    "surface_triangles": faces
                }

        elif method == "gmsh":
            try:
                import gmsh

                # Initialize GMsh
                gmsh.initialize()
                gmsh.model.add("solid_mesh")

                # Create temporary STL file
                import tempfile
                import stl
                from stl import mesh as stl_mesh

                with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
                    stl_file = tmp.name

                # Save surface as STL
                surface = stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype))
                for i, f in enumerate(faces):
                    for j in range(3):
                        surface.vectors[i][j] = verts[f[j], :]
                surface.save(stl_file)

                # Load STL file into GMsh
                gmsh.merge(stl_file)

                # Create volume from surface
                surface_loop_tags = gmsh.model.getEntities(2)  # Get all surfaces
                if surface_loop_tags:
                    # Create volume from surfaces
                    loop_tag = gmsh.model.geo.addSurfaceLoop([tag[1] for tag in surface_loop_tags])
                    volume_tag = gmsh.model.geo.addVolume([loop_tag])
                    gmsh.model.geo.synchronize()

                    # Set meshing parameters
                    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", edge_length)
                    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", edge_length / 5)
                    gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay algorithm

                    # Generate 3D mesh
                    gmsh.model.mesh.generate(3)

                    # Get nodes
                    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
                    vertices = np.array(nodeCoords).reshape(-1, 3)

                    # Get tetrahedra (element type 4)
                    elementTypes, elementTags, elementNodeTags = gmsh.model.mesh.getElements(3)  # 3D elements

                    if 4 in elementTypes:  # Check if tetrahedra exist
                        tet_idx = elementTypes.index(4)
                        tet_tags = elementNodeTags[tet_idx]
                        tetrahedra = np.array(tet_tags).reshape(-1, 4) - 1  # Convert to 0-indexed

                        mesh_dict = {
                            "vertices": vertices,
                            "tetrahedra": tetrahedra,
                            "surface_triangles": faces
                        }
                    else:
                        raise ValueError("No tetrahedral elements created by GMsh")
                else:
                    raise ValueError("No surfaces found in STL file")

                # Clean up
                gmsh.finalize()
                os.unlink(stl_file)

            except (ImportError, ValueError) as e:
                logging.warning(f"GMsh failed: {e}. Falling back to TetGen.")

                # Fall back to TetGen
                import tetgen
                tgen = tetgen.TetGen(verts, faces)
                nodes, elements = tgen.tetrahedralize(
                    quality=True,
                    plc=True,  # Preserve input geometry
                    minratio=1.2,  # Allows slightly varying edge lengths for better volume fitting
                    mindihedral=25.0,  # Standard high-quality angle (prevents over-constraining)
                    maxvolume=edge_length ** 3,  # Control element size
                    steinerleft=100000,  # Allow sufficient refinement points
                    nobisect=False  # Allow boundary refinement
                )

                mesh_dict = {
                    "vertices": nodes,
                    "tetrahedra": elements,
                    "surface_triangles": faces
                }

        else:
            raise ValueError(f"Unknown meshing method: {method}")

        # Save mesh if output directory provided
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # Save as VTK
            vtk_path = os.path.join(output_dir, "trabecular_solid_mesh.vtk")
            save_mesh_as_vtk(mesh_dict, vtk_path)

            # Also save in native format of chosen method
            if method == "tetgen":
                # Save TetGen files (.node, .ele)
                base_path = os.path.join(output_dir, "trabecular")
                save_tetgen_files(mesh_dict, base_path)
            elif method == "netgen" and "nglib" in sys.modules:
                # Save NetGen .vol file
                vol_path = os.path.join(output_dir, "trabecular.vol")
                save_netgen_vol(mesh_dict, vol_path)
            elif method == "gmsh" and "gmsh" in sys.modules:
                # Save GMsh .msh file
                msh_path = os.path.join(output_dir, "trabecular.msh")
                save_gmsh_msh(mesh_dict, msh_path)

        # Evaluate mesh quality
        tet_collapse, volume_skew = evaluate_mesh_quality(mesh_dict)
        logging.info(f"Mesh quality - Tet collapse: {tet_collapse:.4f}, Volume skew: {volume_skew:.4f}")

        return mesh_dict

    except Exception as e:
        logging.error(f"Error creating solid tetrahedral mesh: {e}")
        traceback.print_exc()
        return None


def save_tetgen_files(mesh, base_path):
    """Save mesh in TetGen format (.node and .ele files)"""
    try:
        # Save .node file (vertices)
        with open(f"{base_path}.node", 'w') as f:
            # Header: num_points dimension num_attributes boundary_marker
            f.write(f"{len(mesh['vertices'])} 3 0 0\n")
            # Write vertices: index x y z
            for i, v in enumerate(mesh['vertices']):
                f.write(f"{i + 1} {v[0]} {v[1]} {v[2]}\n")

        # Save .ele file (tetrahedra)
        with open(f"{base_path}.ele", 'w') as f:
            # Header: num_tets nodes_per_tet attribute
            f.write(f"{len(mesh['tetrahedra'])} 4 0\n")
            # Write tetrahedra: index v1 v2 v3 v4
            for i, t in enumerate(mesh['tetrahedra']):
                # Add 1 to convert to 1-indexed format
                f.write(f"{i + 1} {t[0] + 1} {t[1] + 1} {t[2] + 1} {t[3] + 1}\n")

        logging.info(f"Saved TetGen files: {base_path}.node and {base_path}.ele")
        return True
    except Exception as e:
        logging.error(f"Error saving TetGen files: {e}")
        return False


def save_netgen_vol(mesh, filepath):
    """Save mesh in NetGen .vol format"""
    try:
        with open(filepath, 'w') as f:
            # Write header
            f.write("mesh3d\n")
            f.write("dimension\n3\n")

            # Write points section
            f.write("points\n")
            f.write(f"{len(mesh['vertices'])}\n")
            for v in mesh['vertices']:
                f.write(f"{v[0]:20.16f} {v[1]:20.16f} {v[2]:20.16f}\n")

            # Write volumeelements section
            f.write("volumeelements\n")
            f.write(f"{len(mesh['tetrahedra'])}\n")
            for t in mesh['tetrahedra']:
                # Format: type mat domain vol v1 v2 v3 v4
                # Add 1 to convert to 1-indexed
                f.write(f"1 1 1 0 {t[0] + 1} {t[1] + 1} {t[2] + 1} {t[3] + 1}\n")

        logging.info(f"Saved NetGen .vol file: {filepath}")
        return True
    except Exception as e:
        logging.error(f"Error saving NetGen .vol file: {e}")
        return False


def save_gmsh_msh(mesh, filepath):
    """Save mesh in GMsh .msh format"""
    try:
        with open(filepath, 'w') as f:
            # Write MSH format v2.2 header
            f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")

            # Write nodes
            f.write("$Nodes\n")
            f.write(f"{len(mesh['vertices'])}\n")
            for i, v in enumerate(mesh['vertices']):
                f.write(f"{i + 1} {v[0]} {v[1]} {v[2]}\n")
            f.write("$EndNodes\n")

            # Write elements (tetrahedra)
            f.write("$Elements\n")
            f.write(f"{len(mesh['tetrahedra'])}\n")
            for i, t in enumerate(mesh['tetrahedra']):
                # Format: elem_num elem_type tags v1 v2 v3 v4
                # elem_type 4 = tetrahedral
                # Add 1 to convert to 1-indexed
                f.write(f"{i + 1} 4 2 1 1 {t[0] + 1} {t[1] + 1} {t[2] + 1} {t[3] + 1}\n")
            f.write("$EndElements\n")

        logging.info(f"Saved GMsh .msh file: {filepath}")
        return True
    except Exception as e:
        logging.error(f"Error saving GMsh .msh file: {e}")
        return False


def save_mesh_as_vtk(mesh, filename, binary=True):
    """
    Save a mesh to a legacy VTK file format with better error handling.

    Parameters:
    -----------
    mesh : dict
        Mesh dictionary containing vertices and tetrahedra
    filename : str
        Path to save the VTK file
    binary : bool
        Whether to save in binary format (faster, smaller file)
    """
    try:
        logging.info(f"Saving mesh to {filename}...")

        if not mesh or "vertices" not in mesh or "tetrahedra" not in mesh:
            logging.error("Invalid mesh data - missing vertices or tetrahedra")
            return False

        # Create VTK data structures
        vtk_mesh = vtk.vtkUnstructuredGrid()

        # Add points
        points = vtk.vtkPoints()
        for vertex in mesh["vertices"]:
            points.InsertNextPoint(vertex)
        vtk_mesh.SetPoints(points)

        # Add tetrahedra
        for tet in mesh["tetrahedra"]:
            tetra = vtk.vtkTetra()
            for i in range(4):
                tetra.GetPointIds().SetId(i, tet[i])
            vtk_mesh.InsertNextCell(tetra.GetCellType(), tetra.GetPointIds())

        # Ensure directory exists
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Save the file
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(filename)
        writer.SetInputData(vtk_mesh)
        if binary:
            writer.SetFileTypeToBinary()
        else:
            writer.SetFileTypeToASCII()
        writer.Write()

        logging.info(
            f"Mesh with {len(mesh['vertices'])} vertices and {len(mesh['tetrahedra'])} tetrahedra saved to {filename}")
        return True
    except Exception as e:
        logging.error(f"Error saving mesh to VTK: {e}")
        traceback.print_exc()
        return False


def save_mesh_with_scalars(mesh, scalars, filename):
    """
    Save a mesh with scalar values to a VTK file.

    Parameters:
    -----------
    mesh : dict
        Mesh dictionary containing vertices and tetrahedra
    scalars : numpy.ndarray
        Scalar values at each vertex
    filename : str
        Path to save the VTK file
    """
    try:
        logging.info(f"Saving mesh with scalars to {filename}...")

        # Create a VTK mesh object
        vtk_mesh = vtk.vtkUnstructuredGrid()

        # Add points
        points = vtk.vtkPoints()
        for vertex in mesh["vertices"]:
            points.InsertNextPoint(vertex)
        vtk_mesh.SetPoints(points)

        # Add tetrahedra
        for tet in mesh["tetrahedra"]:
            tetra = vtk.vtkTetra()
            for i in range(4):
                tetra.GetPointIds().SetId(i, tet[i])
            vtk_mesh.InsertNextCell(tetra.GetCellType(), tetra.GetPointIds())

        # Add scalar values as point data
        vtk_scalars = numpy_support.numpy_to_vtk(np.ascontiguousarray(scalars, dtype=np.float64))
        vtk_scalars.SetName("Scalar_Value")
        vtk_mesh.GetPointData().AddArray(vtk_scalars)
        vtk_mesh.GetPointData().SetActiveScalars("Scalar_Value")

        # Write the mesh to a VTK file
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(filename)
        writer.SetInputData(vtk_mesh)
        writer.Write()

        logging.info(f"Mesh with {len(scalars)} scalar values saved to {filename}")
        return True
    except Exception as e:
        logging.error(f"Error saving mesh with scalars: {e}")
        traceback.print_exc()
        return False


def evaluate_mesh_quality(mesh, max_tetrahedra_to_check=10000):
    """
    Evaluate tetrahedral mesh quality with tet_collapse and volume_skew metrics
    as specified in Bachmann et al. (2022).

    Parameters:
    -----------
    mesh : dict
        Mesh dictionary containing vertices and tetrahedra
    max_tetrahedra_to_check : int
        Maximum number of tetrahedra to check for very large meshes

    Returns:
    --------
    tuple
        (tet_collapse, volume_skew) metrics
    """
    try:
        # logging.info("Evaluating mesh quality...")

        if not mesh or "vertices" not in mesh or "tetrahedra" not in mesh:
            logging.error("Invalid mesh - missing vertices or tetrahedra")
            return 0, 1  # Worst quality values

        vertices = mesh["vertices"]
        tetrahedra = mesh["tetrahedra"]

        if len(tetrahedra) == 0:
            logging.warning("No tetrahedra found in mesh!")
            return 0, 1  # Worst quality values

        # For large meshes, check a subset
        if len(tetrahedra) > max_tetrahedra_to_check:
            indices = np.linspace(0, len(tetrahedra) - 1, max_tetrahedra_to_check).astype(int)
            tetrahedra_subset = tetrahedra[indices]
        else:
            tetrahedra_subset = tetrahedra

        # Calculate tet_collapse metric (ranges from 0 for collapsed to 1 for optimal)
        # Formula based on MSC.Software - ratio of height to face area
        tet_collapse_values = []

        # Calculate volume_skew metric (ranges from 0 for equilateral to 1 for degenerated)
        # Formula: deviation from equilateral tetrahedron volume
        volume_skew_values = []

        # Ideal volume ratio for equilateral tetrahedron
        equilateral_volume_ratio = np.sqrt(2) / 12

        for tet in tetrahedra_subset:
            try:
                # Get vertices
                v0, v1, v2, v3 = vertices[tet[0]], vertices[tet[1]], vertices[tet[2]], vertices[tet[3]]

                # Calculate edge vectors
                e01 = v1 - v0
                e02 = v2 - v0
                e03 = v3 - v0
                e12 = v2 - v1
                e13 = v3 - v1
                e23 = v3 - v2

                # Calculate edge lengths
                edges = [
                    np.linalg.norm(e01),
                    np.linalg.norm(e02),
                    np.linalg.norm(e03),
                    np.linalg.norm(e12),
                    np.linalg.norm(e13),
                    np.linalg.norm(e23)
                ]

                # Calculate volume
                volume = np.abs(np.dot(np.cross(e01, e02), e03)) / 6

                if volume > 0:
                    # Calculate face areas for tet_collapse
                    # Face areas
                    a0 = 0.5 * np.linalg.norm(np.cross(e12, e13))  # Face opposite v0
                    a1 = 0.5 * np.linalg.norm(np.cross(e02, e03))  # Face opposite v1
                    a2 = 0.5 * np.linalg.norm(np.cross(e01, e03))  # Face opposite v2
                    a3 = 0.5 * np.linalg.norm(np.cross(e01, e02))  # Face opposite v3

                    # Heights from each vertex to opposite face
                    h0 = 3 * volume / a0 if a0 > 0 else 0
                    h1 = 3 * volume / a1 if a1 > 0 else 0
                    h2 = 3 * volume / a2 if a2 > 0 else 0
                    h3 = 3 * volume / a3 if a3 > 0 else 0

                    # Calculate min height / max face area ratio
                    heights = [h0, h1, h2, h3]
                    areas = [a0, a1, a2, a3]

                    # Avoid division by zero
                    min_height = min(heights) if any(h > 0 for h in heights) else 0
                    max_area = max(areas) if any(a > 0 for a in areas) else 1

                    # Normalize to [0,1] range where 1 is optimal
                    # For regular tetrahedron, height/area = √6/2
                    ideal_ratio = np.sqrt(6) / 2
                    actual_ratio = min_height / max_area if max_area > 0 else 0

                    # Normalize to [0,1] (0 = collapsed, 1 = optimal)
                    tet_collapse = min(actual_ratio / ideal_ratio, 1.0) if ideal_ratio > 0 else 0
                    tet_collapse_values.append(tet_collapse)

                    # Calculate volume_skew
                    # Maximum edge length
                    max_edge = max(edges)

                    # Volume of equilateral tetrahedron with same max edge length
                    equilateral_volume = equilateral_volume_ratio * (max_edge ** 3)

                    # Deviation from equilateral (0 = equilateral, 1 = degenerated)
                    volume_skew = 1.0 - min(volume / equilateral_volume, 1.0) if equilateral_volume > 0 else 1.0
                    volume_skew_values.append(volume_skew)

            except Exception as e:
                logging.warning(f"Error evaluating tetrahedron: {e}")
                continue

        # Average the metrics
        if tet_collapse_values:
            avg_tet_collapse = np.mean(tet_collapse_values)
        else:
            avg_tet_collapse = 0

        if volume_skew_values:
            avg_volume_skew = np.mean(volume_skew_values)
        else:
            avg_volume_skew = 1

        # logging.info(f"Mesh quality metrics:")
        # logging.info(f"  Tet collapse: {avg_tet_collapse:.4f} (higher is better, 1.0 is optimal)")
        # logging.info(f"  Volume skew: {avg_volume_skew:.4f} (lower is better, 0.0 is optimal)")

        return avg_tet_collapse, avg_volume_skew

    except Exception as e:
        logging.error(f"Error evaluating mesh quality: {e}")
        import traceback
        traceback.print_exc()
        return 0, 1  # Return worst quality values


def similarity_registration(fixed_image, moving_image):
    """Perform similarity registration that preserves bone structure."""
    try:
        # Convert images to float for registration
        fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
        moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

        # Log initial statistics to verify content
        fixed_stats = sitk.StatisticsImageFilter()
        moving_stats = sitk.StatisticsImageFilter()
        fixed_stats.Execute(fixed_image)
        moving_stats.Execute(moving_image)

        logging.info(f"Registration input - Fixed: Min={fixed_stats.GetMinimum()}, "
                     f"Max={fixed_stats.GetMaximum()}, Mean={fixed_stats.GetMean():.4f}")
        logging.info(f"Registration input - Moving: Min={moving_stats.GetMinimum()}, "
                     f"Max={moving_stats.GetMaximum()}, Mean={moving_stats.GetMean():.4f}")

        # Initialize transform
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image,
            moving_image,
            sitk.Similarity3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

        # Configure registration
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsCorrelation()
        #registration_method.SetMetricAsMeanSquares()
        #registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetInterpolator(sitk.sitkNearestNeighbor)
        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        # Simplified optimizer settings
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=0.1,
            numberOfIterations=100
        )

        # Execute registration
        final_transform = registration_method.Execute(fixed_image, moving_image)

        # Log metric value
        metric_value = registration_method.GetMetricValue()
        logging.info(f"Registration completed - Metric value: {metric_value:.6f}")

        return final_transform

    except Exception as e:
        logging.error(f"Error in similarity registration: {e}")
        traceback.print_exc()
        return None


def create_confined_background_grid(image, grid_spacing=2.5):
    """Create background grid confined to the bone region only."""
    # Extract bone content boundaries
    array = sitk.GetArrayFromImage(image)
    if np.sum(array) > 0:
        indices = np.nonzero(array)
        z_min, y_min, x_min = np.min(indices, axis=1)
        z_max, y_max, x_max = np.max(indices, axis=1)

        # Add small margin around the bone
        margin = int(grid_spacing / image.GetSpacing()[0])
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        z_min = max(0, z_min - margin)
        x_max = min(image.GetSize()[0] - 1, x_max + margin)
        y_max = min(image.GetSize()[1] - 1, y_max + margin)
        z_max = min(image.GetSize()[2] - 1, z_max + margin)

        # Convert to physical coordinates
        origin = image.GetOrigin()
        spacing = image.GetSpacing()

        # Calculate grid points
        grid_points = []
        for z in np.arange(z_min, z_max + 1, grid_spacing / spacing[2]):
            for y in np.arange(y_min, y_max + 1, grid_spacing / spacing[1]):
                for x in np.arange(x_min, x_max + 1, grid_spacing / spacing[0]):
                    # Convert to physical coordinates
                    px = origin[0] + x * spacing[0]
                    py = origin[1] + y * spacing[1]
                    pz = origin[2] + z * spacing[2]
                    grid_points.append([px, py, pz])

        return np.array(grid_points)
    else:
        return None


def align_mesh_to_image(mesh_dict, image, output_dir=None):
    """
    Enhanced mesh alignment function that works with Dragonfly-exported meshes.

    This function handles the scale and coordinate system differences between
    a mesh exported from Dragonfly and the SimpleITK image space.
    """
    try:
        logging.info("Starting advanced mesh-to-image alignment...")

        # Get image properties
        size = image.GetSize()
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()

        # Calculate image physical boundaries
        physical_dims = [size[i] * spacing[i] for i in range(3)]
        image_center = [origin[i] + physical_dims[i] / 2 for i in range(3)]

        # Extract non-zero region from image (bone content)
        array = sitk.GetArrayFromImage(image)
        if np.sum(array) > 0:
            indices = np.nonzero(array)
            z_min, y_min, x_min = np.min(indices, axis=1)
            z_max, y_max, x_max = np.max(indices, axis=1)

            # Convert to physical coordinates (SimpleITK uses x,y,z ordering)
            content_min = [
                origin[0] + x_min * spacing[0],
                origin[1] + y_min * spacing[1],
                origin[2] + z_min * spacing[2]
            ]
            content_max = [
                origin[0] + x_max * spacing[0],
                origin[1] + y_max * spacing[1],
                origin[2] + z_max * spacing[2]
            ]
            content_center = [(content_min[i] + content_max[i]) / 2 for i in range(3)]
            content_dims = [content_max[i] - content_min[i] for i in range(3)]

        else:
            logging.warning("Image appears empty. Using full image bounds.")
            content_center = image_center
            content_dims = physical_dims

        # Get mesh properties
        mesh_vertices = mesh_dict['vertices']
        mesh_min = np.min(mesh_vertices, axis=0)
        mesh_max = np.max(mesh_vertices, axis=0)
        mesh_center = (mesh_min + mesh_max) / 2
        mesh_dims = mesh_max - mesh_min

        # Calculate scaling factors based on physical dimensions
        # Use minimum scaling to preserve aspect ratio
        scale_factors = [content_dims[i] / mesh_dims[i] for i in range(3)]
        # logging.info(f"Scaling factors: {scale_factors}")

        uniform_scale = min(scale_factors) * 0.98  # Slight reduction to ensure fit
        # logging.info(f"Using uniform scale factor: {uniform_scale}")

        # Transform vertices: scale and translate
        transformed_vertices = np.copy(mesh_vertices)

        # Center the mesh at origin
        transformed_vertices -= mesh_center

        # Scale uniformly
        transformed_vertices *= uniform_scale

        # Translate to content center
        transformed_vertices += content_center

        transformed_mesh = {
            'vertices': transformed_vertices,
            'tetrahedra': mesh_dict['tetrahedra'].copy()
        }

        # Save visualization if requested
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # Save meshes before and after transformation
            # save_mesh_as_vtk(mesh_dict, os.path.join(output_dir, 'original_mesh.vtk'))
            # save_mesh_as_vtk(transformed_mesh, os.path.join(output_dir, 'aligned_mesh.vtk'))

            # Visualize alignment
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')

            # Sample points for visualization (to avoid plotting all vertices)
            sample_size = min(1000, len(mesh_vertices))
            indices = np.random.choice(len(mesh_vertices), sample_size, replace=False)

            # Plot original and transformed vertices
            ax.scatter(
                mesh_vertices[indices, 0],
                mesh_vertices[indices, 1],
                mesh_vertices[indices, 2],
                c='red', alpha=0.5, label='Original Mesh'
            )
            ax.scatter(
                transformed_vertices[indices, 0],
                transformed_vertices[indices, 1],
                transformed_vertices[indices, 2],
                c='blue', alpha=0.5, label='Aligned Mesh'
            )

            # Plot content bounds as box
            def plot_box(min_coords, max_coords, color='k'):
                x_min, y_min, z_min = min_coords
                x_max, y_max, z_max = max_coords

                # Draw box edges
                for x, y, z in [(x_min, y_min, z_min), (x_max, y_min, z_min),
                                (x_max, y_max, z_min), (x_min, y_max, z_min)]:
                    ax.plot([x, x], [y, y], [z_min, z_max], color)

                for z in [z_min, z_max]:
                    ax.plot([x_min, x_max, x_max, x_min, x_min],
                            [y_min, y_min, y_max, y_max, y_min],
                            [z, z, z, z, z], color)

            plot_box(content_min, content_max)

            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.set_title('Mesh Alignment with Content Bounds')
            ax.legend()

            plt.savefig(os.path.join(output_dir, 'mesh_alignment_visualization.png'), dpi=600)
            plt.close()

        logging.info(f"Mesh alignment completed successfully")
        return transformed_mesh

    except Exception as e:
        logging.error(f"Error in mesh alignment: {e}")
        traceback.print_exc()
        return mesh_dict  # Return original as fallback


def displace(mesh, displacement_field, sigma=1.0):
    # Get vector displacement array [Z, Y, X, 3]
    disp_array = sitk.GetArrayFromImage(displacement_field)

    # Smooth each component
    smoothed_components = []
    for i in range(3):  # x, y, z
        comp_img = sitk.GetImageFromArray(disp_array[..., i])
        comp_img.CopyInformation(displacement_field)
        smoothed = sitk.SmoothingRecursiveGaussian(comp_img, sigma)
        smoothed_components.append(smoothed)

    # Create scalar interpolators for each component
    interpolators = [sitk.LinearInterpolateImageFunction() for _ in range(3)]
    for i in range(3):
        interpolators[i].SetInputImage(smoothed_components[i])

    # Apply to mesh
    morphed_vertices = []
    for v in mesh['vertices']:
        displacement = [float(interpolators[i].Evaluate(v)) for i in range(3)]
        morphed = np.array(v) + np.array(displacement)
        morphed_vertices.append(morphed)

    return {
        'vertices': np.array(morphed_vertices),
        'tetrahedra': mesh['tetrahedra'].copy()
    }


def center_mesh_to_image(mesh, image):
    """
    Pure translation alignment. Shifts the mesh so its geometric center
    perfectly matches the center of the target trabecular image mask.
    Does not scale or distort the B-spline morphed tetrahedra.
    """



    # 1. Find the physical center of the actual bone in the image
    array = sitk.GetArrayFromImage(image)
    if np.sum(array) == 0:
        logging.warning("Image is empty. Skipping centering.")
        return mesh

    indices = np.nonzero(array)
    z_min, y_min, x_min = np.min(indices, axis=1)
    z_max, y_max, x_max = np.max(indices, axis=1)

    spacing = image.GetSpacing()
    origin = image.GetOrigin()

    content_center = [
        origin[0] + ((x_min + x_max) / 2.0) * spacing[0],
        origin[1] + ((y_min + y_max) / 2.0) * spacing[1],
        origin[2] + ((z_min + z_max) / 2.0) * spacing[2]
    ]

    # 2. Find the geometric center of the morphed mesh
    mesh_vertices = mesh['vertices']
    mesh_min = np.min(mesh_vertices, axis=0)
    mesh_max = np.max(mesh_vertices, axis=0)
    mesh_center = (mesh_min + mesh_max) / 2.0

    # 3. Calculate the required slide (translation vector)
    translation_vector = np.array(content_center) - np.array(mesh_center)
    shifted_vertices = mesh_vertices + translation_vector

    centered_mesh = {
        'vertices': shifted_vertices,
        'tetrahedra': mesh['tetrahedra'].copy()
    }

    if 'surface_triangles' in mesh:
        centered_mesh['surface_triangles'] = mesh['surface_triangles'].copy()

    return centered_mesh

def isomorph_Maybe(mesh, canonical_image, individual_image, transforms_dir, bone_name, iteration, debug_dir=None):
    """
    Morphs the canonical mesh using Normalized Convolution to smooth the
    B-spline deformation field. This prevents tetrahedra from folding while
    guaranteeing the boundary vectors do not shrink or pull off-center.
    """
    logging.info(f"Morphing canonical mesh to {bone_name} space...")

    bspline_transform = find_transforms_for_bone(
        transforms_dir,
        bone_name,
        iteration
    )

    if bspline_transform is None:
        logging.error("Morphing failed: Transform is missing.")
        return None

    # 1. Convert raw B-spline to a continuous displacement field
    displacement_field = sitk.TransformToDisplacementField(
        bspline_transform,
        sitk.sitkVectorFloat64,
        canonical_image.GetSize(),
        canonical_image.GetOrigin(),
        canonical_image.GetSpacing(),
        canonical_image.GetDirection()
    )

    # 2. Create the confidence mask (1 inside the bone, 0 outside)
    # We dilate it slightly (by 2 voxels) so the mesh boundaries are fully engulfed
    mask = sitk.Cast(canonical_image > 0, sitk.sitkUInt8)
    mask = sitk.BinaryDilate(mask, [2, 2, 2], sitk.sitkBall)
    mask_float = sitk.Cast(mask, sitk.sitkFloat64)

    # 3. Apply Normalized Convolution Smoothing
    sigma = 1.5
    smoothed_mask = sitk.SmoothingRecursiveGaussian(mask_float, sigma)

    # --- THE FIX: Force the Gaussian output to 64-bit float ---
    smoothed_mask = sitk.Cast(smoothed_mask, sitk.sitkFloat64)
    # ----------------------------------------------------------

    # Create a safe denominator to prevent division by zero in empty space
    condition_float = sitk.Cast(smoothed_mask > 0.001, sitk.sitkFloat64)
    not_condition_float = sitk.Cast(smoothed_mask <= 0.001, sitk.sitkFloat64)
    safe_mask = condition_float * smoothed_mask + not_condition_float

    smoothed_components = []
    for i in range(3):
        comp = sitk.VectorIndexSelectionCast(displacement_field, i)

        # Mask the component to ignore the zero-displacement background
        masked_comp = comp * mask_float

        # Smooth the masked component
        smoothed_comp = sitk.SmoothingRecursiveGaussian(masked_comp, sigma)

        # Force the smoothed component to 64-bit float before division
        smoothed_comp = sitk.Cast(smoothed_comp, sitk.sitkFloat64)

        # Normalize to restore true magnitude at the boundaries
        corrected_comp = smoothed_comp / safe_mask
        smoothed_components.append(corrected_comp)

    # Compose the corrected displacement field
    smoothed_field = sitk.Compose(smoothed_components)
    smooth_transform = sitk.DisplacementFieldTransform(smoothed_field)

    # 4. Apply the perfect, relaxed morph to the mesh vertices
    logging.info("Warping mesh vertices...")
    morphed_vertices = []
    for vertex in mesh['vertices']:
        transformed_pt = smooth_transform.TransformPoint(vertex.tolist())
        morphed_vertices.append(transformed_pt)

    morphed_mesh = {
        'vertices': np.array(morphed_vertices),
        'tetrahedra': mesh['tetrahedra'].copy()
    }

    # Preserve surface triangles if they exist
    if 'surface_triangles' in mesh:
        morphed_mesh['surface_triangles'] = mesh['surface_triangles'].copy()

    logging.info("Mesh morphing completed successfully.")
    return morphed_mesh

def isomorph(mesh, canonical_image, individual_image, transforms_dir, bone_name, iteration, debug_dir=None):
    """
    Morphs the canonical mesh to the individual trabecular space.
    Uses smoothed displacement fields to prevent mesh tearing and non-diffeomorphic folding.
    """
    logging.info(f"Morphing canonical mesh to {bone_name} space...")

    bspline_transform = find_transforms_for_bone(
        transforms_dir,
        bone_name,
        iteration
    )

    if bspline_transform is None:
        logging.error("Morphing failed: Transform is missing.")
        return None

    # Convert the raw B-spline to a Continuous Displacement Field
    displacement_field = sitk.TransformToDisplacementField(
        bspline_transform,
        sitk.sitkVectorFloat64,
        canonical_image.GetSize(),
        canonical_image.GetOrigin(),
        canonical_image.GetSpacing(),
        canonical_image.GetDirection()
    )

    # Gaussian Smoothing
    displacement_components = [sitk.VectorIndexSelectionCast(displacement_field, i) for i in range(3)]

    smoothed_components = []
    for comp in displacement_components:
        # A sigma of 1.5 is the sweet spot for relaxing trabecular deformations
        smoothed = sitk.SmoothingRecursiveGaussian(comp, 1.5)
        smoothed_components.append(sitk.Cast(smoothed, sitk.sitkFloat64))

    smoothed_field = sitk.Compose(smoothed_components)

    # Create the smooth Forward transform. (DO NOT INVERT)
    smooth_transform = sitk.DisplacementFieldTransform(smoothed_field)

    # Apply the relaxed, accurate morph to the mesh vertices
    morphed_vertices = []
    for vertex in mesh['vertices']:
        transformed_pt = smooth_transform.TransformPoint(vertex.tolist())
        morphed_vertices.append(transformed_pt)

    morphed_mesh = {
        'vertices': np.array(morphed_vertices),
        'tetrahedra': mesh['tetrahedra'].copy()
    }

    # Preserve surface triangles if they exist
    if 'surface_triangles' in mesh:
        morphed_mesh['surface_triangles'] = mesh['surface_triangles'].copy()

    morphed_mesh = center_mesh_to_image(morphed_mesh, individual_image)

    logging.info(f"Isotopological Mesh Created Successfully for {bone_name}.")
    return morphed_mesh

def isomorph1(mesh, canonical_image, individual_image, transforms_dir, bone_name, iteration, debug_dir=None):
    """
    Morphs the canonical mesh to the individual trabecular space by directly
    applying the ITK B-spline transform without backwards inversion.
    """
    logging.info(f"Morphing canonical mesh to {bone_name} space...")

    bspline_transform = find_transforms_for_bone(
        transforms_dir,
        bone_name,
        iteration
    )

    if bspline_transform is None:
        logging.error("Morphing failed: Transformation file is missing.")
        return None

    morphed_vertices = []

    for vertex in mesh['vertices']:
        # TransformPoint expects physical coordinates (which the vertices already are)
        transformed_pt = bspline_transform.TransformPoint(vertex.tolist())
        morphed_vertices.append(transformed_pt)

    morphed_mesh = {
        'vertices': np.array(morphed_vertices),
        'tetrahedra': mesh['tetrahedra'].copy()
    }

    # (Optional) Keep any surface triangles if they exist in the mesh dictionary
    if 'surface_triangles' in mesh:
        morphed_mesh['surface_triangles'] = mesh['surface_triangles'].copy()

    logging.info("Isotopological Mesh Creation Completed Successfully.")
    return morphed_mesh


def isomorph2(mesh, canonical_image, individual_image, transforms_dir, bone_name, iteration, debug_dir=None):
    """Enhanced function to morph canonical mesh to individual trabecular space."""

    bspline_transform = find_transforms_for_bone(
        transforms_dir,
        bone_name,
        iteration
    )

    displacement_field = sitk.TransformToDisplacementField(
        bspline_transform,
        sitk.sitkVectorFloat64,
        canonical_image.GetSize(),
        canonical_image.GetOrigin(),
        canonical_image.GetSpacing(),
        canonical_image.GetDirection()
    )

    displacement_components = [sitk.VectorIndexSelectionCast(displacement_field, i) for i in range(3)]
    smoothed_components = [sitk.DiscreteGaussian(comp, 1) for comp in displacement_components]
    vector_components = [sitk.Cast(comp, sitk.sitkFloat64) for comp in smoothed_components]
    smoothed_field = sitk.Compose(vector_components)
    # bspline_transform = sitk.DisplacementFieldTransform(smoothed_field)

    inverted_field = sitk.InvertDisplacementField(
        smoothed_field,
        maximumNumberOfIterations=150,
        meanErrorToleranceThreshold=0.01,
        maxErrorToleranceThreshold=0.05,
        enforceBoundaryCondition=True
    )

    bspline_transform = sitk.DisplacementFieldTransform(inverted_field)

    morphed_vertices = []
    for vertex in mesh['vertices']:
        transformed = bspline_transform.TransformPoint(vertex.tolist())
        morphed_vertices.append(transformed)

    morphed_mesh = {
        'vertices': np.array(morphed_vertices),
        'tetrahedra': mesh['tetrahedra'].copy()
    }

    # morphed_mesh = align_mesh_to_image(morphed_mesh, individual_image, output_dir = "B:/")

    return morphed_mesh


def hma(trabecular_image, mesh, grid_spacing=2.5, sphere_diameter=5.0, output_dir=None):
    """
    Enhanced HMA implementation that ensures proper alignment between mesh and image.
    """
    try:
        logging.info("Starting improved holistic morphometric analysis...")

        # First, verify the mesh and image alignment
        logging.info("Verifying mesh alignment with image...")
        mesh_vertices = mesh['vertices']

        # Get image properties
        spacing = trabecular_image.GetSpacing()
        size = trabecular_image.GetSize()
        origin = trabecular_image.GetOrigin()

        # Calculate image physical boundaries
        image_bounds = [
            origin[0], origin[0] + size[0] * spacing[0],
            origin[1], origin[1] + size[1] * spacing[1],
            origin[2], origin[2] + size[2] * spacing[2]
        ]

        # Get mesh boundaries
        mesh_min = np.min(mesh_vertices, axis=0)
        mesh_max = np.max(mesh_vertices, axis=0)

        # Check if mesh is contained within image bounds
        mesh_contained = (
                mesh_min[0] >= image_bounds[0] and mesh_max[0] <= image_bounds[1] and
                mesh_min[1] >= image_bounds[2] and mesh_max[1] <= image_bounds[3] and
                mesh_min[2] >= image_bounds[4] and mesh_max[2] <= image_bounds[5]
        )

        if not mesh_contained:
            logging.warning("Mesh extends outside image bounds! This may affect results.")
            logging.warning(f"Image bounds: {image_bounds}")
            logging.warning(f"Mesh bounds: {mesh_min.tolist()} to {mesh_max.tolist()}")

        # Create background grid with appropriate dimensions
        logging.info(f"Creating background grid with {grid_spacing}mm spacing...")

        # Calculate number of grid points
        n_grid_x = max(2, int((image_bounds[1] - image_bounds[0]) / grid_spacing) + 1)
        n_grid_y = max(2, int((image_bounds[3] - image_bounds[2]) / grid_spacing) + 1)
        n_grid_z = max(2, int((image_bounds[5] - image_bounds[4]) / grid_spacing) + 1)

        logging.info(f"Grid dimensions: {n_grid_x} x {n_grid_y} x {n_grid_z} points")

        # Create grid points
        grid_points = []
        for k in range(n_grid_z):
            z = image_bounds[4] + k * grid_spacing
            for j in range(n_grid_y):
                y = image_bounds[2] + j * grid_spacing
                for i in range(n_grid_x):
                    x = image_bounds[0] + i * grid_spacing
                    grid_points.append([x, y, z])

        grid_points = np.array(grid_points)

        # Convert image to numpy array
        np_array = sitk.GetArrayFromImage(trabecular_image)

        # Calculate BV/TV at each grid point
        logging.info("Calculating BV/TV at grid points...")
        bv_tv_values = np.zeros(len(grid_points))
        fabric_tensors = np.zeros((len(grid_points), 3, 3))

        # Calculate sphere radius in voxels
        radius_voxels = [sphere_diameter / (2.0 * spacing[i]) for i in range(3)]

        # Process in batches
        batch_size = 100
        num_batches = (len(grid_points) + batch_size - 1) // batch_size

        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, len(grid_points))

            for i in range(start_idx, end_idx):
                point = grid_points[i]

                # Convert to voxel coordinates
                idx_x = int((point[0] - origin[0]) / spacing[0])
                idx_y = int((point[1] - origin[1]) / spacing[1])
                idx_z = int((point[2] - origin[2]) / spacing[2])

                # Check if within image bounds
                if (0 <= idx_x < size[0] and 0 <= idx_y < size[1] and 0 <= idx_z < size[2]):
                    # Define sampling sphere
                    x_min = max(0, int(idx_x - radius_voxels[0]))
                    x_max = min(size[0] - 1, int(idx_x + radius_voxels[0]))
                    y_min = max(0, int(idx_y - radius_voxels[1]))
                    y_max = min(size[1] - 1, int(idx_y + radius_voxels[1]))
                    z_min = max(0, int(idx_z - radius_voxels[2]))
                    z_max = min(size[2] - 1, int(idx_z + radius_voxels[2]))

                    # Skip if sphere is outside image
                    if x_min >= x_max or y_min >= y_max or z_min >= z_max:
                        continue

                    # Extract region (numpy array is Z,Y,X)
                    region = np_array[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1]

                    # Calculate BV/TV
                    total_voxels = region.size
                    bone_voxels = np.sum(region > 0)

                    if total_voxels > 0:
                        bv_tv = bone_voxels / total_voxels
                    else:
                        bv_tv = 0

                    bv_tv_values[i] = bv_tv

                    # Calculate fabric tensor if enough bone
                    if bone_voxels > 10 and region.shape[0] > 2 and region.shape[1] > 2 and region.shape[2] > 2:
                        # Calculate structure tensor using image gradients
                        gz, gy, gx = np.gradient(region.astype(float))

                        fabric = np.zeros((3, 3))
                        fabric[0, 0] = np.mean(gx * gx)  # xx
                        fabric[0, 1] = fabric[1, 0] = np.mean(gx * gy)  # xy/yx
                        fabric[0, 2] = fabric[2, 0] = np.mean(gx * gz)  # xz/zx
                        fabric[1, 1] = np.mean(gy * gy)  # yy
                        fabric[1, 2] = fabric[2, 1] = np.mean(gy * gz)  # yz/zy
                        fabric[2, 2] = np.mean(gz * gz)  # zz

                        # Ensure positive definiteness
                        evals, evecs = np.linalg.eigh(fabric)
                        if np.any(evals < 0):
                            evals[evals < 1e-6] = 1e-6
                            # evals = np.abs(evals)
                            fabric = evecs @ np.diag(evals) @ evecs.T

                        fabric_tensors[i] = fabric
                    else:
                        # Default to isotropic tensor for regions with little/no bone
                        fabric_tensors[i] = np.eye(3)

        # Create KD-tree for interpolation
        logging.info("Interpolating values to mesh vertices...")
        from scipy.spatial import cKDTree
        tree = cKDTree(grid_points)

        # Interpolate to mesh vertices
        vertex_bv_tv = np.zeros(len(mesh_vertices))
        vertex_fabric = np.zeros((len(mesh_vertices), 3, 3))

        # Process in batches
        vertex_batch_size = 10000
        num_batches = (len(mesh_vertices) + vertex_batch_size - 1) // vertex_batch_size

        for batch in range(num_batches):
            start_idx = batch * vertex_batch_size
            end_idx = min((batch + 1) * vertex_batch_size, len(mesh_vertices))

            batch_vertices = mesh_vertices[start_idx:end_idx]

            # Find nearest neighbors for interpolation (k=8 for trilinear)
            k = 8
            distances, indices = tree.query(batch_vertices, k=k)

            # Calculate interpolation weights (inverse distance)
            weights = 1.0 / (distances + 1e-10)
            weights = weights / np.sum(weights, axis=1, keepdims=True)

            # Interpolate BV/TV and fabric tensor
            for i in range(end_idx - start_idx):
                # BV/TV interpolation
                vertex_bv_tv[start_idx + i] = np.sum(weights[i] * bv_tv_values[indices[i]])

                # Fabric tensor interpolation
                for j in range(k):
                    vertex_fabric[start_idx + i] += weights[i, j] * fabric_tensors[indices[i, j]]

        # Calculate element values from vertex values
        logging.info("Computing element-wise values...")
        element_bv_tv = np.zeros(len(mesh['tetrahedra']))
        element_fabric = np.zeros((len(mesh['tetrahedra']), 3, 3))

        for i, tet in enumerate(mesh['tetrahedra']):
            # Calculate element centroid
            centroid = np.mean(mesh_vertices[tet], axis=0)

            # Find nearest grid point
            _, idx = tree.query(centroid, k=1)

            # Assign values
            element_bv_tv[i] = bv_tv_values[idx]
            element_fabric[i] = fabric_tensors[idx]

        # Calculate mean BV/TV (exclude empty regions)
        valid_bv_tv = vertex_bv_tv[vertex_bv_tv > 0.01]
        if len(valid_bv_tv) > 0:
            mean_bv_tv = np.mean(valid_bv_tv)
        else:
            mean_bv_tv = 0.001  # Fallback

        # Calculate relative BV/TV
        vertex_r_bv_tv = vertex_bv_tv / mean_bv_tv if mean_bv_tv > 0 else np.zeros_like(vertex_bv_tv)
        element_r_bv_tv = element_bv_tv / mean_bv_tv if mean_bv_tv > 0 else np.zeros_like(element_bv_tv)

        # Calculate degree of anisotropy
        vertex_da = np.ones(len(mesh_vertices))  # Default to isotropic
        element_da = np.ones(len(mesh['tetrahedra']))

        for i in range(len(mesh_vertices)):
            try:
                evals, _ = np.linalg.eigh(vertex_fabric[i])
                evals[evals < 1e-6] = 1e-6
                # evals = np.abs(evals)  # Ensure positive
                evals = np.sort(evals)  # Sort ascending

                if evals[0] > 1e-6:
                    vertex_da[i] = evals[2] / evals[0]  # DA = λ₁/λ₃
            except:
                pass  # Keep default value

        for i in range(len(mesh['tetrahedra'])):
            try:
                evals, _ = np.linalg.eigh(element_fabric[i])
                evals[evals < 1e-6] = 1e-6
                # evals = np.abs(evals)
                evals = np.sort(evals)

                if evals[0] > 1e-6:
                    element_da[i] = evals[2] / evals[0]
            except:
                pass

        # Calculate mean DA
        valid_da = vertex_da[(vertex_da > 1.0) & (vertex_da < 10.0)]
        mean_da = np.mean(valid_da) if len(valid_da) > 0 else 1.0

        # Calculate relative DA
        vertex_r_da = vertex_da / mean_da if mean_da > 0 else np.ones_like(vertex_da)
        element_r_da = element_da / mean_da if mean_da > 0 else np.ones_like(element_da)

        # 11. Visualize results if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # Visualize background grid with BV/TV values
            if len(grid_points) > 0:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D

                # Plot with randomly sampled points to avoid overplotting
                max_points = 5000
                if len(grid_points) > max_points:
                    idx = np.random.choice(len(grid_points), max_points, replace=False)
                    sample_points = grid_points[idx]
                    sample_values = bv_tv_values[idx]
                else:
                    sample_points = grid_points
                    sample_values = bv_tv_values

                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')

                sc = ax.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2],
                                c=sample_values, cmap='viridis', alpha=0.5, s=10)

                plt.colorbar(sc, ax=ax, label='BV/TV')
                ax.set_xlabel('X (mm)')
                ax.set_ylabel('Y (mm)')
                ax.set_zlabel('Z (mm)')
                ax.set_title('Background Grid BV/TV Values')

                plt.savefig(os.path.join(output_dir, 'background_grid_bvtv.png'), dpi=600)
                plt.close()

                # Visualize mesh vertices with BV/TV values
                if len(mesh_vertices) > max_points:
                    idx = np.random.choice(len(mesh_vertices), max_points, replace=False)
                    sample_vertices = mesh_vertices[idx]
                    sample_bvtv = vertex_bv_tv[idx]
                else:
                    sample_vertices = mesh_vertices
                    sample_bvtv = vertex_bv_tv

                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')

                sc = ax.scatter(sample_vertices[:, 0], sample_vertices[:, 1], sample_vertices[:, 2],
                                c=sample_bvtv, cmap='viridis', alpha=0.5, s=10)

                plt.colorbar(sc, ax=ax, label='BV/TV')
                ax.set_xlabel('X (mm)')
                ax.set_ylabel('Y (mm)')
                ax.set_zlabel('Z (mm)')
                ax.set_title('Mesh Vertices BV/TV Values')

                plt.savefig(os.path.join(output_dir, 'mesh_vertices_bvtv.png'), dpi=600)
                plt.close()

                # Also save cross-sections of the original image for reference
                array = sitk.GetArrayFromImage(trabecular_image)
                middle_z = array.shape[0] // 2

                plt.figure(figsize=(10, 8))
                plt.imshow(array[middle_z, :, :], cmap='gray')
                plt.title(f'Binary Image (Z-slice {middle_z})')
                plt.colorbar(label='Intensity')
                plt.savefig(os.path.join(output_dir, 'trabecular_image_z_slice.png'), dpi=300)
                plt.close()

                middle_y = array.shape[1] // 2
                plt.figure(figsize=(10, 8))
                plt.imshow(array[:, middle_y, :], cmap='gray')
                plt.title(f'Binary Image (Y-slice {middle_y})')
                plt.colorbar(label='Intensity')
                plt.savefig(os.path.join(output_dir, 'trabecular_image_y_slice.png'), dpi=300)
                plt.close()

        # 12. Create result dictionary
        result = {
            'vertex_bv_tv': vertex_bv_tv,
            'vertex_r_bv_tv': vertex_r_bv_tv,
            'vertex_da': vertex_da,
            'vertex_r_da': vertex_r_da,
            'vertex_fabric_tensors': vertex_fabric,
            'element_bv_tv': element_bv_tv,
            'element_r_bv_tv': element_r_bv_tv,
            'element_da': element_da,
            'element_r_da': element_r_da,
            'element_fabric_tensors': element_fabric,
            'mean_bv_tv': mean_bv_tv,
            'mean_da': mean_da
        }

        logging.info(f"HMA completed successfully. Mean BV/TV: {mean_bv_tv:.6f}, Mean DA: {mean_da:.6f}")

        return result

    except Exception as e:
        logging.error(f"Error in holistic morphometric analysis: {e}")
        traceback.print_exc()
        return None


def improved_hma(trabecular_image, mesh, grid_spacing=1.5, sphere_diameter=3.0, output_dir=None):
    """
    Improved holistic morphometric analysis following Bachmann et al. 2022.

    Parameters:
    -----------
    trabecular_image : sitk.Image
        Binary image of trabecular bone
    mesh : dict
        Mesh dictionary with vertices and tetrahedra
    grid_spacing : float
        Grid spacing in mm (default: 1.5 mm)
    sphere_diameter : float
        Diameter of sampling sphere in mm (default: 3 mm)
    output_dir : str, optional
        Directory to save debug visualizations

    Returns:
    --------
    dict
        Dictionary with HMA results
    """
    try:
        logging.info("Starting improved holistic morphometric analysis...")

        # Step 1: Create background grid confined to bone region
        grid_points = create_confined_background_grid(
            trabecular_image,
            grid_spacing=grid_spacing
        )

        if grid_points is None or len(grid_points) == 0:
            logging.error("Failed to create background grid")
            return None

        # Step 2: Calculate BV/TV and anisotropy at grid points
        grid_values = calculate_bvtv_and_anisotropy_at_grid_points(
            trabecular_image,
            grid_points,
            sphere_diameter=sphere_diameter
        )

        if grid_values is None:
            logging.error("Failed to calculate BV/TV and anisotropy at grid points")
            return None

        # Step 3: Create bone presence mask
        bone_mask = create_bone_presence_mask(trabecular_image, mesh)

        # Step 4: Calculate confidence scores
        confidence_scores = calculate_confidence_scores(
            mesh,
            trabecular_image,
            grid_points,
            grid_values
        )

        # Step 5: Interpolate values to mesh vertices
        mesh_values = interpolate_values_to_mesh(
            grid_points,
            grid_values,
            mesh['vertices']
        )

        if mesh_values is None:
            logging.error("Failed to interpolate values to mesh vertices")
            return None

        # Step 6: Calculate element values from vertex values
        element_values = calculate_element_values(mesh, mesh_values)

        # Step 7: Create final result dictionary
        result = {
            # Vertex values
            'vertex_bv_tv': mesh_values['vertex_bv_tv'],
            'vertex_r_bv_tv': mesh_values['vertex_r_bv_tv'],
            'vertex_da': mesh_values['vertex_anisotropy'],
            'vertex_r_da': mesh_values['vertex_r_anisotropy'],
            'vertex_fabric': mesh_values['vertex_fabric'],

            # Element values
            'element_bv_tv': element_values['element_bv_tv'],
            'element_r_bv_tv': element_values['element_r_bv_tv'],
            'element_da': element_values['element_anisotropy'],
            'element_r_da': element_values['element_r_anisotropy'],
            'element_fabric': element_values['element_fabric'],

            # Mean values
            'mean_bv_tv': mesh_values['mean_bv_tv'],
            'mean_da': mesh_values['mean_anisotropy'],

            # Confidence values
            'confidence': confidence_scores,
            'bone': bone_mask
        }

        # Step 8: Visualize results if output directory is provided
        if output_dir:
            # Create debug visualizations
            create_debug_visualizations(
                trabecular_image,
                mesh,
                grid_points,
                grid_values,
                mesh_values,
                confidence_scores,
                bone_mask,
                output_dir
            )

        logging.info("Holistic morphometric analysis completed successfully.")
        return result

    except Exception as e:
        logging.error(f"Error in holistic morphometric analysis: {e}")
        traceback.print_exc()
        return None


def calculate_bvtv_and_anisotropy_at_grid_points(trabecular_image, grid_points, sphere_diameter=5.0):
    """
    Calculate BV/TV and anisotropy at each grid point according to Bachmann et al. 2022.

    Parameters:
    -----------
    trabecular_image : sitk.Image
        Binary image of trabecular bone
    grid_points : numpy.ndarray
        Array of physical coordinates for grid points
    sphere_diameter : float
        Diameter of sampling sphere in mm (default: 5.0 mm)

    Returns:
    --------
    dict
        Dictionary with BV/TV, anisotropy and fabric tensors at each grid point
    """
    try:
        logging.info(f"Calculating BV/TV and anisotropy at {len(grid_points)} grid points...")

        # Get image properties
        spacing = trabecular_image.GetSpacing()
        size = trabecular_image.GetSize()
        origin = trabecular_image.GetOrigin()

        # Convert binary image to numpy array
        np_array = sitk.GetArrayFromImage(trabecular_image)

        # Calculate sphere radius in voxels for each dimension
        radius_voxels = [sphere_diameter / (2.0 * spacing[i]) for i in range(3)]

        # Initialize results
        bv_tv_values = np.zeros(len(grid_points))
        fabric_tensors = np.zeros((len(grid_points), 3, 3))

        # Process in batches to save memory
        batch_size = 100
        total_batches = (len(grid_points) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(grid_points))

            for i in range(start_idx, end_idx):
                point = grid_points[i]

                # Convert physical point to voxel coordinates
                idx_x = int((point[0] - origin[0]) / spacing[0])
                idx_y = int((point[1] - origin[1]) / spacing[1])
                idx_z = int((point[2] - origin[2]) / spacing[2])

                # Check if point is inside image domain
                if (0 <= idx_x < size[0] and 0 <= idx_y < size[1] and 0 <= idx_z < size[2]):
                    # Define sampling sphere
                    x_min = max(0, int(idx_x - radius_voxels[0]))
                    x_max = min(size[0] - 1, int(idx_x + radius_voxels[0]))
                    y_min = max(0, int(idx_y - radius_voxels[1]))
                    y_max = min(size[1] - 1, int(idx_y + radius_voxels[1]))
                    z_min = max(0, int(idx_z - radius_voxels[2]))
                    z_max = min(size[2] - 1, int(idx_z + radius_voxels[2]))

                    # Skip if sphere is too small or outside image
                    if x_min >= x_max or y_min >= y_max or z_min >= z_max:
                        continue

                    # Extract sphere region (numpy array is z,y,x)
                    region = np_array[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1]

                    # Calculate BV/TV
                    total_voxels = region.size
                    bone_voxels = np.sum(region > 0)

                    if total_voxels > 0:
                        bv_tv = bone_voxels / total_voxels
                    else:
                        bv_tv = 0

                    bv_tv_values[i] = bv_tv

                    # Calculate fabric tensor using Mean Intercept Length (MIL) method
                    # as described in Bachmann et al. 2022 and Gross et al. 2014
                    if bone_voxels > 10 and region.shape[0] > 2 and region.shape[1] > 2 and region.shape[2] > 2:
                        try:
                            # Calculate structure tensor using gradient approach
                            # This is an approximation of the MIL
                            gz, gy, gx = np.gradient(region.astype(float))

                            # Create fabric tensor (structure tensor)
                            M = np.zeros((3, 3))
                            M[0, 0] = np.mean(gx * gx)  # xx
                            M[0, 1] = M[1, 0] = np.mean(gx * gy)  # xy/yx
                            M[0, 2] = M[2, 0] = np.mean(gx * gz)  # xz/zx
                            M[1, 1] = np.mean(gy * gy)  # yy
                            M[1, 2] = M[2, 1] = np.mean(gy * gz)  # yz/zy
                            M[2, 2] = np.mean(gz * gz)  # zz

                            # Ensure matrix is positive definite
                            eigenvalues, eigenvectors = np.linalg.eigh(M)
                            if np.any(eigenvalues < 0):
                                eigenvalues = np.abs(eigenvalues)
                                M = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

                            fabric_tensors[i] = M
                        except Exception as e:
                            logging.warning(f"Error calculating fabric tensor at point {i}: {e}")
                            fabric_tensors[i] = np.eye(3)  # Default to isotropic
                    else:
                        # Not enough bone or region too small
                        fabric_tensors[i] = np.eye(3)  # Default to isotropic tensor

        # Calculate degree of anisotropy from fabric tensors
        anisotropy_values = np.ones(len(grid_points))  # Default to isotropic

        for i in range(len(grid_points)):
            try:
                eigenvalues, _ = np.linalg.eigh(fabric_tensors[i])
                eigenvalues = np.sort(eigenvalues)  # Sort in ascending order

                if eigenvalues[0] > 1e-6:  # Avoid division by zero
                    # DA = λ₁/λ₃ (largest divided by smallest eigenvalue)
                    anisotropy_values[i] = eigenvalues[2] / eigenvalues[0]
            except Exception as e:
                logging.warning(f"Error calculating anisotropy at point {i}: {e}")
                # Keep default value of 1.0 (isotropic)

        # Calculate mean values (excluding zeros and outliers)
        valid_bv_tv = bv_tv_values[bv_tv_values > 0.01]
        mean_bv_tv = np.mean(valid_bv_tv) if len(valid_bv_tv) > 0 else 0.001

        valid_anisotropy = anisotropy_values[(anisotropy_values > 1.0) & (anisotropy_values < 10.0)]
        mean_anisotropy = np.mean(valid_anisotropy) if len(valid_anisotropy) > 0 else 1.0

        # Calculate relative values
        r_bv_tv_values = bv_tv_values / mean_bv_tv if mean_bv_tv > 0 else np.zeros_like(bv_tv_values)
        r_anisotropy_values = anisotropy_values / mean_anisotropy if mean_anisotropy > 0 else np.ones_like(
            anisotropy_values)

        logging.info(f"BV/TV calculation completed. Mean BV/TV: {mean_bv_tv:.4f}, Mean DA: {mean_anisotropy:.4f}")

        return {
            'grid_points': grid_points,
            'bv_tv': bv_tv_values,
            'r_bv_tv': r_bv_tv_values,
            'anisotropy': anisotropy_values,
            'r_anisotropy': r_anisotropy_values,
            'fabric_tensors': fabric_tensors,
            'mean_bv_tv': mean_bv_tv,
            'mean_anisotropy': mean_anisotropy
        }

    except Exception as e:
        logging.error(f"Error calculating BV/TV and anisotropy: {e}")
        traceback.print_exc()
        return None


def create_confined_background_grid(trabecular_image, grid_spacing=2.5, margin_factor=1.1):
    """
    Create background grid confined to the bone region with margin.

    Parameters:
    -----------
    trabecular_image : sitk.Image
        Binary image of trabecular bone
    grid_spacing : float
        Grid spacing in mm (default: 2.5 mm)
    margin_factor : float
        Factor to add margin around bone (default: 1.2 = 20% extra space)

    Returns:
    --------
    numpy.ndarray
        Array of grid point coordinates in physical space
    """
    try:
        logging.info("Creating confined background grid...")

        # Get image properties
        spacing = trabecular_image.GetSpacing()
        size = trabecular_image.GetSize()
        origin = trabecular_image.GetOrigin()

        # Convert binary image to numpy array
        array = sitk.GetArrayFromImage(trabecular_image)

        # Extract bone content boundaries
        if np.sum(array) > 0:
            # Find indices of non-zero voxels
            indices = np.nonzero(array)

            # Get min and max indices (z, y, x in numpy array)
            z_min, y_min, x_min = np.min(indices, axis=1)
            z_max, y_max, x_max = np.max(indices, axis=1)

            # Convert to physical coordinates
            min_physical = [
                origin[0] + x_min * spacing[0],
                origin[1] + y_min * spacing[1],
                origin[2] + z_min * spacing[2]
            ]

            max_physical = [
                origin[0] + x_max * spacing[0],
                origin[1] + y_max * spacing[1],
                origin[2] + z_max * spacing[2]
            ]

            # Calculate center and dimensions
            center = [(min_physical[i] + max_physical[i]) / 2 for i in range(3)]
            dimensions = [max_physical[i] - min_physical[i] for i in range(3)]

            # Add margin
            dimensions = [d * margin_factor for d in dimensions]

            # Recalculate min and max with margin
            min_physical = [center[i] - dimensions[i] / 2 for i in range(3)]
            max_physical = [center[i] + dimensions[i] / 2 for i in range(3)]

            # Create grid points
            grid_points = []

            # Calculate number of points in each dimension
            n_points = [int(dimensions[i] / grid_spacing) + 1 for i in range(3)]

            # Ensure at least 2 points in each dimension
            n_points = [max(2, n) for n in n_points]

            logging.info(f"Creating grid with {n_points[0]}x{n_points[1]}x{n_points[2]} points")

            # Create evenly spaced grid
            for k in range(n_points[2]):
                z = min_physical[2] + k * (dimensions[2] / (n_points[2] - 1))
                for j in range(n_points[1]):
                    y = min_physical[1] + j * (dimensions[1] / (n_points[1] - 1))
                    for i in range(n_points[0]):
                        x = min_physical[0] + i * (dimensions[0] / (n_points[0] - 1))
                        grid_points.append([x, y, z])

            logging.info(f"Created background grid with {len(grid_points)} points")
            return np.array(grid_points)
        else:
            logging.warning("Image appears to be empty. Creating default grid.")
            # Create default grid based on full image dimensions
            physical_size = [size[i] * spacing[i] for i in range(3)]
            n_grid_x = max(2, int(physical_size[0] / grid_spacing) + 1)
            n_grid_y = max(2, int(physical_size[1] / grid_spacing) + 1)
            n_grid_z = max(2, int(physical_size[2] / grid_spacing) + 1)

            grid_points = []
            for k in range(n_grid_z):
                z = origin[2] + k * grid_spacing
                for j in range(n_grid_y):
                    y = origin[1] + j * grid_spacing
                    for i in range(n_grid_x):
                        x = origin[0] + i * grid_spacing
                        grid_points.append([x, y, z])

            logging.info(f"Created default grid with {len(grid_points)} points")
            return np.array(grid_points)

    except Exception as e:
        logging.error(f"Error creating confined background grid: {e}")
        traceback.print_exc()
        return None


def interpolate_values_to_mesh(grid_points, grid_values, mesh_vertices, k=8):
    """
    Interpolate values from grid points to mesh vertices.

    Parameters:
    -----------
    grid_points : numpy.ndarray
        Array of grid point coordinates in physical space
    grid_values : dict
        Dictionary with values at grid points
    mesh_vertices : numpy.ndarray
        Array of mesh vertex coordinates
    k : int
        Number of nearest neighbors for interpolation (default: 8)

    Returns:
    --------
    dict
        Dictionary with interpolated values at mesh vertices
    """
    try:
        logging.info(
            f"Interpolating values from {len(grid_points)} grid points to {len(mesh_vertices)} mesh vertices...")

        # Create KD-tree for efficient nearest neighbor search
        from scipy.spatial import cKDTree
        tree = cKDTree(grid_points)

        # Initialize result arrays
        vertex_bv_tv = np.zeros(len(mesh_vertices))
        vertex_r_bv_tv = np.zeros(len(mesh_vertices))
        vertex_anisotropy = np.ones(len(mesh_vertices))
        vertex_r_anisotropy = np.ones(len(mesh_vertices))
        vertex_fabric = np.zeros((len(mesh_vertices), 3, 3))

        # Process in batches to save memory
        batch_size = 10000
        total_batches = (len(mesh_vertices) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(mesh_vertices))

            batch_vertices = mesh_vertices[start_idx:end_idx]

            # Find k nearest neighbors for each vertex
            distances, indices = tree.query(batch_vertices, k=k)

            # Handle case where some points might be far from any grid point
            valid_mask = distances < np.inf
            if not np.all(valid_mask):
                logging.warning(f"Some vertices have no valid grid points nearby. Using nearest point.")
                # Fall back to nearest point for these cases
                for i in range(len(batch_vertices)):
                    if not np.any(valid_mask[i]):
                        _, idx = tree.query(batch_vertices[i], k=1)
                        indices[i, 0] = idx
                        distances[i, 0] = 1.0
                        valid_mask[i, 0] = True

            # Calculate interpolation weights using inverse distance weighting
            weights = np.zeros_like(distances)
            for i in range(len(batch_vertices)):
                valid_indices = np.where(valid_mask[i])[0]
                if len(valid_indices) > 0:
                    # Inverse distance weights
                    inv_dist = 1.0 / (distances[i, valid_indices] + 1e-10)
                    weights[i, valid_indices] = inv_dist / np.sum(inv_dist)

            # Interpolate BV/TV
            for i in range(end_idx - start_idx):
                # BV/TV and relative BV/TV
                vertex_bv_tv[start_idx + i] = np.sum(weights[i] * grid_values['bv_tv'][indices[i]])
                vertex_r_bv_tv[start_idx + i] = np.sum(weights[i] * grid_values['r_bv_tv'][indices[i]])

                # Anisotropy and relative anisotropy
                vertex_anisotropy[start_idx + i] = np.sum(weights[i] * grid_values['anisotropy'][indices[i]])
                vertex_r_anisotropy[start_idx + i] = np.sum(weights[i] * grid_values['r_anisotropy'][indices[i]])

                # Fabric tensor (weighted average)
                fabric = np.zeros((3, 3))
                for j in range(k):
                    if weights[i, j] > 0:
                        fabric += weights[i, j] * grid_values['fabric_tensors'][indices[i, j]]
                vertex_fabric[start_idx + i] = fabric

        logging.info("Interpolation completed successfully.")

        return {
            'vertex_bv_tv': vertex_bv_tv,
            'vertex_r_bv_tv': vertex_r_bv_tv,
            'vertex_anisotropy': vertex_anisotropy,
            'vertex_r_anisotropy': vertex_r_anisotropy,
            'vertex_fabric': vertex_fabric,
            'mean_bv_tv': grid_values['mean_bv_tv'],
            'mean_anisotropy': grid_values['mean_anisotropy']
        }

    except Exception as e:
        logging.error(f"Error interpolating values to mesh: {e}")
        traceback.print_exc()
        return None


def create_debug_visualizations(trabecular_image, mesh, grid_points, grid_values, mesh_values,
                                confidence_scores, bone_mask, output_dir):
    """
    Create debug visualizations for HMA with confidence scores.
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Sample points for visualization (to avoid plotting all points)
        max_points = 2000
        if len(mesh['vertices']) > max_points:
            sample_indices = np.random.choice(len(mesh['vertices']), max_points, replace=False)
        else:
            sample_indices = np.arange(len(mesh['vertices']))

        # 1. Confidence scores visualization
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(
            mesh['vertices'][sample_indices, 0],
            mesh['vertices'][sample_indices, 1],
            mesh['vertices'][sample_indices, 2],
            c=confidence_scores[sample_indices],
            cmap='viridis',
            s=10,
            alpha=0.8
        )

        plt.colorbar(scatter, ax=ax, label='Confidence Score')
        ax.set_title('Confidence Scores')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')

        plt.savefig(os.path.join(output_dir, 'confidence_scores.png'), dpi=600)
        plt.close()

        # 2. Bone mask visualization
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Points with bone
        bone_indices = sample_indices[bone_mask[sample_indices]]
        if len(bone_indices) > 0:
            ax.scatter(
                mesh['vertices'][bone_indices, 0],
                mesh['vertices'][bone_indices, 1],
                mesh['vertices'][bone_indices, 2],
                c='blue',
                s=10,
                alpha=0.8,
                label='Bone Present'
            )

        # Points without bone
        no_bone_indices = sample_indices[~bone_mask[sample_indices]]
        if len(no_bone_indices) > 0:
            ax.scatter(
                mesh['vertices'][no_bone_indices, 0],
                mesh['vertices'][no_bone_indices, 1],
                mesh['vertices'][no_bone_indices, 2],
                c='red',
                s=10,
                alpha=0.8,
                label='No Bone'
            )

        ax.set_title('Bone Presence Mask')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.legend()

        plt.savefig(os.path.join(output_dir, 'bone_mask.png'), dpi=600)
        plt.close()

        # 3. Combined visualization
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Color by confidence, but use different marker for bone presence
        bone_indices = sample_indices[bone_mask[sample_indices]]
        no_bone_indices = sample_indices[~bone_mask[sample_indices]]

        if len(bone_indices) > 0:
            scatter1 = ax.scatter(
                mesh['vertices'][bone_indices, 0],
                mesh['vertices'][bone_indices, 1],
                mesh['vertices'][bone_indices, 2],
                c=confidence_scores[bone_indices],
                cmap='viridis',
                s=15,
                marker='o',
                alpha=0.8
            )

        if len(no_bone_indices) > 0:
            scatter2 = ax.scatter(
                mesh['vertices'][no_bone_indices, 0],
                mesh['vertices'][no_bone_indices, 1],
                mesh['vertices'][no_bone_indices, 2],
                c=confidence_scores[no_bone_indices],
                cmap='viridis',
                s=15,
                marker='x',
                alpha=0.8
            )

        plt.colorbar(scatter1, ax=ax, label='Confidence Score')
        ax.set_title('Confidence Scores with Bone Presence (o = bone, x = no bone)')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')

        plt.savefig(os.path.join(output_dir, 'combined_visualization.png'), dpi=600)
        plt.close()

        logging.info(f"Created confidence visualizations in {output_dir}")

    except Exception as e:
        logging.error(f"Error creating debug visualizations: {e}")
        traceback.print_exc()


def visualize_mesh_image_alignment(mesh, image, output_path):
    """
    Visualize alignment between mesh and image.

    Parameters:
    -----------
    mesh : dict
        Mesh dictionary with vertices and tetrahedra
    image : sitk.Image
        Binary image of trabecular bone
    output_path : str
        Path to save visualization
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Get image properties
        spacing = image.GetSpacing()
        size = image.GetSize()
        origin = image.GetOrigin()

        # Convert binary image to numpy array
        array = sitk.GetArrayFromImage(image)

        # Extract mesh boundaries
        vertices = mesh['vertices']
        mesh_min = np.min(vertices, axis=0)
        mesh_max = np.max(vertices, axis=0)

        # Extract image boundaries
        if np.sum(array) > 0:
            # Find indices of non-zero voxels
            indices = np.nonzero(array)

            # Get min and max indices (z, y, x in numpy array)
            z_min, y_min, x_min = np.min(indices, axis=1)
            z_max, y_max, x_max = np.max(indices, axis=1)

            # Convert to physical coordinates
            image_min = [
                origin[0] + x_min * spacing[0],
                origin[1] + y_min * spacing[1],
                origin[2] + z_min * spacing[2]
            ]

            image_max = [
                origin[0] + x_max * spacing[0],
                origin[1] + y_max * spacing[1],
                origin[2] + z_max * spacing[2]
            ]
        else:
            # If image is empty, use full image domain
            image_min = [origin[i] for i in range(3)]
            image_max = [origin[i] + size[i] * spacing[i] for i in range(3)]

        # Create figure with 4 subplots
        fig = plt.figure(figsize=(20, 15))

        # 3D visualization of mesh and image
        ax1 = fig.add_subplot(221, projection='3d')

        # Sample mesh vertices for visualization
        sample_size = min(1000, len(vertices))
        sample_indices = np.random.choice(len(vertices), sample_size, replace=False)
        sample_vertices = vertices[sample_indices]

        # Plot mesh vertices
        ax1.scatter(
            sample_vertices[:, 0],
            sample_vertices[:, 1],
            sample_vertices[:, 2],
            c='blue',
            alpha=0.5,
            label='Mesh Vertices'
        )

        # Plot image boundaries as box
        def plot_box(ax, min_coords, max_coords, color='r', label=None):
            x_min, y_min, z_min = min_coords
            x_max, y_max, z_max = max_coords

            # Create array of vertices
            xs = [x_min, x_max, x_max, x_min, x_min, x_max, x_max, x_min]
            ys = [y_min, y_min, y_max, y_max, y_min, y_min, y_max, y_max]
            zs = [z_min, z_min, z_min, z_min, z_max, z_max, z_max, z_max]

            # List of sides' polygons
            verts = [
                [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6],
                [3, 0, 4, 7], [0, 1, 2, 3], [4, 5, 6, 7]
            ]

            # Plot sides
            for v in verts:
                ax.plot3D(
                    [xs[v[0]], xs[v[1]], xs[v[2]], xs[v[3]], xs[v[0]]],
                    [ys[v[0]], ys[v[1]], ys[v[2]], ys[v[3]], ys[v[0]]],
                    [zs[v[0]], zs[v[1]], zs[v[2]], zs[v[3]], zs[v[0]]],
                    color=color,
                    alpha=0.7
                )

            # Add label to only one edge
            if label:
                ax.plot3D([x_min, x_max], [y_min, y_min], [z_min, z_min], color=color, label=label)

        # Plot image and mesh boundaries
        plot_box(ax1, image_min, image_max, color='r', label='Image Content')
        plot_box(ax1, mesh_min, mesh_max, color='g', label='Mesh Boundaries')

        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_zlabel('Z (mm)')
        ax1.set_title('3D Visualization of Mesh and Image Alignment')
        ax1.legend()

        # Create 2D cross-sections
        # X-Y plane (mid Z)
        ax2 = fig.add_subplot(222)
        z_mid = int((z_min + z_max) / 2)
        if 0 <= z_mid < array.shape[0]:
            ax2.imshow(array[z_mid], cmap='gray', origin='lower')

            # Project mesh vertices near z_mid to this slice
            z_slice = sample_vertices[(sample_vertices[:, 2] >= origin[2] + (z_mid - 1) * spacing[2]) &
                                      (sample_vertices[:, 2] <= origin[2] + (z_mid + 1) * spacing[2])]

            if len(z_slice) > 0:
                # Convert to pixel coordinates
                px = (z_slice[:, 0] - origin[0]) / spacing[0]
                py = (z_slice[:, 1] - origin[1]) / spacing[1]

                ax2.scatter(px, py, c='r', s=5, alpha=0.5)

        ax2.set_title(f'X-Y Plane (Z={z_mid})')

        # X-Z plane (mid Y)
        ax3 = fig.add_subplot(223)
        y_mid = int((y_min + y_max) / 2)
        if 0 <= y_mid < array.shape[1]:
            ax3.imshow(array[:, y_mid, :], cmap='gray', origin='lower')

            # Project mesh vertices near y_mid to this slice
            y_slice = sample_vertices[(sample_vertices[:, 1] >= origin[1] + (y_mid - 1) * spacing[1]) &
                                      (sample_vertices[:, 1] <= origin[1] + (y_mid + 1) * spacing[1])]

            if len(y_slice) > 0:
                # Convert to pixel coordinates
                px = (y_slice[:, 0] - origin[0]) / spacing[0]
                pz = (y_slice[:, 2] - origin[2]) / spacing[2]

                ax3.scatter(px, pz, c='r', s=5, alpha=0.5)

        ax3.set_title(f'X-Z Plane (Y={y_mid})')

        # Y-Z plane (mid X)
        ax4 = fig.add_subplot(224)
        x_mid = int((x_min + x_max) / 2)
        if 0 <= x_mid < array.shape[2]:
            ax4.imshow(array[:, :, x_mid], cmap='gray', origin='lower')

            # Project mesh vertices near x_mid to this slice
            x_slice = sample_vertices[(sample_vertices[:, 0] >= origin[0] + (x_mid - 1) * spacing[0]) &
                                      (sample_vertices[:, 0] <= origin[0] + (x_mid + 1) * spacing[0])]

            if len(x_slice) > 0:
                # Convert to pixel coordinates
                py = (x_slice[:, 1] - origin[1]) / spacing[1]
                pz = (x_slice[:, 2] - origin[2]) / spacing[2]

                ax4.scatter(py, pz, c='r', s=5, alpha=0.5)

        ax4.set_title(f'Y-Z Plane (X={x_mid})')

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        logging.info(f"Saved alignment visualization to {output_path}")

    except Exception as e:
        logging.error(f"Error creating alignment visualization: {e}")
        traceback.print_exc()


def calculate_element_values(mesh, vertex_values):
    """
    Calculate element values from vertex values.

    Parameters:
    -----------
    mesh : dict
        Mesh dictionary with vertices and tetrahedra
    vertex_values : dict
        Dictionary with values at mesh vertices

    Returns:
    --------
    dict
        Dictionary with values at mesh elements
    """
    try:
        logging.info(f"Calculating element values for {len(mesh['tetrahedra'])} tetrahedra...")

        # Initialize result arrays
        element_bv_tv = np.zeros(len(mesh['tetrahedra']))
        element_r_bv_tv = np.zeros(len(mesh['tetrahedra']))
        element_anisotropy = np.ones(len(mesh['tetrahedra']))
        element_r_anisotropy = np.ones(len(mesh['tetrahedra']))
        element_fabric = np.zeros((len(mesh['tetrahedra']), 3, 3))

        # For each tetrahedron, average the values of its vertices
        for i, tet in enumerate(mesh['tetrahedra']):
            # BV/TV and relative BV/TV
            element_bv_tv[i] = np.mean(vertex_values['vertex_bv_tv'][tet])
            element_r_bv_tv[i] = np.mean(vertex_values['vertex_r_bv_tv'][tet])

            # Anisotropy and relative anisotropy
            element_anisotropy[i] = np.mean(vertex_values['vertex_anisotropy'][tet])
            element_r_anisotropy[i] = np.mean(vertex_values['vertex_r_anisotropy'][tet])

            # Fabric tensor (average of vertex tensors)
            fabric = np.zeros((3, 3))
            for v in tet:
                fabric += vertex_values['vertex_fabric'][v]
            element_fabric[i] = fabric / len(tet)

        return {
            'element_bv_tv': element_bv_tv,
            'element_r_bv_tv': element_r_bv_tv,
            'element_anisotropy': element_anisotropy,
            'element_r_anisotropy': element_r_anisotropy,
            'element_fabric': element_fabric
        }

    except Exception as e:
        logging.error(f"Error calculating element values: {e}")
        traceback.print_exc()
        return None


def save_mesh_with_scalars_csv(mesh, scalars, base_filename):
    """
    Save mesh vertices and scalar values to CSV files for easier import into R.

    Parameters:
    -----------
    mesh : dict
        Mesh dictionary containing vertices and tetrahedra
    scalars : numpy.ndarray
        Scalar values at each vertex
    base_filename : str
        Base path for the CSV files (without extension)
    """
    try:
        # Save vertices
        vertices_file = f"{base_filename}_vertices.csv"
        np.savetxt(vertices_file, mesh['vertices'], delimiter=',',
                   header='x,y,z', comments='')

        # Save scalar values
        scalars_file = f"{base_filename}_scalars.csv"
        np.savetxt(scalars_file, scalars, delimiter=',',
                   header='scalar_value', comments='')

        # Save tetrahedra connectivity if needed
        tetras_file = f"{base_filename}_tetras.csv"
        np.savetxt(tetras_file, mesh['tetrahedra'], delimiter=',',
                   fmt='%d', header='v1,v2,v3,v4', comments='')

        logging.info(f"Saved CSV files for {base_filename}")
        return True
    except Exception as e:
        logging.error(f"Error saving CSV files: {e}")
        traceback.print_exc()
        return False


def extract_principal_fabric_directions(fabric_tensor):
    """
    Extract principal directions (eigenvectors) and values (eigenvalues) from a fabric tensor.

    Parameters:
    -----------
    fabric_tensor : numpy.ndarray
        3x3 fabric tensor

    Returns:
    --------
    tuple
        (eigenvalues, eigenvectors) sorted by decreasing eigenvalue
    """
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(fabric_tensor)

        # Sort by decreasing eigenvalue
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        return eigenvalues, eigenvectors
    except Exception as e:
        logging.error(f"Error extracting principal directions: {e}")
        return None, None


def save_mean_fabric_tensors(results_dict, output_path):
    """
    Save the mean fabric tensors and their principal directions to a CSV file.

    Parameters:
    -----------
    results_dict : dict
        Dictionary containing bone names and their corresponding fabric tensors
    output_path : str
        Path to save the CSV file
    """
    try:
        # Prepare data structure for CSV
        csv_data = []

        for bone_name, fabric_tensors in results_dict.items():
            # Calculate mean fabric tensor across all vertices
            mean_fabric = np.mean(fabric_tensors, axis=0)

            # Extract principal directions and values
            eigenvalues, eigenvectors = extract_principal_fabric_directions(mean_fabric)

            if eigenvalues is not None and eigenvectors is not None:
                # Create row for CSV
                row = {
                    'bone_name': bone_name,
                    # First principal direction (λ1, v1)
                    'lambda1': eigenvalues[0],
                    'v1_x': eigenvectors[0, 0],
                    'v1_y': eigenvectors[1, 0],
                    'v1_z': eigenvectors[2, 0],
                    # Second principal direction (λ2, v2)
                    'lambda2': eigenvalues[1],
                    'v2_x': eigenvectors[0, 1],
                    'v2_y': eigenvectors[1, 1],
                    'v2_z': eigenvectors[2, 1],
                    # Third principal direction (λ3, v3)
                    'lambda3': eigenvalues[2],
                    'v3_x': eigenvectors[0, 2],
                    'v3_y': eigenvectors[1, 2],
                    'v3_z': eigenvectors[2, 2],
                    # Degree of anisotropy (λ1/λ3)
                    'DA': eigenvalues[0] / eigenvalues[2] if eigenvalues[2] > 1e-6 else 1000
                }
                csv_data.append(row)
            else:
                logging.warning(f"Failed to extract principal directions for {bone_name}")

        # Convert to DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False)
        logging.info(f"Saved mean fabric tensor principal directions to {output_path}")

        return True
    except Exception as e:
        logging.error(f"Error saving mean fabric tensors: {e}")
        return False


def align_image_to_mesh(image, mesh):
    """
    Calculates the spatial offset between the geometric center of a morphed mesh
    and the bounding-box center of a trabecular image, and translates the image
    stack to perfectly align with the mesh.
    """
    logging.info("Aligning individual trabecular image to the morphed mesh...")

    # 1. Find the physical center of the bone in the image
    array = sitk.GetArrayFromImage(image)
    if np.sum(array) == 0:
        logging.warning("Image is empty. Skipping alignment.")
        return image

    indices = np.nonzero(array)
    z_min, y_min, x_min = np.min(indices, axis=1)
    z_max, y_max, x_max = np.max(indices, axis=1)

    spacing = image.GetSpacing()
    origin = image.GetOrigin()

    image_center = [
        origin[0] + ((x_min + x_max) / 2.0) * spacing[0],
        origin[1] + ((y_min + y_max) / 2.0) * spacing[1],
        origin[2] + ((z_min + z_max) / 2.0) * spacing[2]
    ]

    # 2. Find the geometric center of the morphed mesh
    mesh_vertices = mesh['vertices']
    mesh_min = np.min(mesh_vertices, axis=0)
    mesh_max = np.max(mesh_vertices, axis=0)
    mesh_center = (mesh_min + mesh_max) / 2.0

    # 3. Calculate translation vector (Image -> Mesh)
    translation_vector = np.array(mesh_center) - np.array(image_center)

    # SimpleITK Resample maps points from the Output space back to the Input space.
    # Therefore, to shift an image by +V, the transform offset must be inverted (-V).
    transform_offset = -translation_vector
    translation_transform = sitk.TranslationTransform(3)
    translation_transform.SetOffset(transform_offset.tolist())

    # 4. Resample the image to its new physical location
    aligned_image = sitk.Resample(
        image,
        image, # Keep the exact same grid dimensions
        translation_transform,
        sitk.sitkNearestNeighbor, # Prevents blurring of binary mask
        0.0,
        image.GetPixelID()
    )

    logging.info(f"Image translated to perfectly engulf mesh by: {translation_vector}")
    return aligned_image


def isoHMA(input_dir, output_dir, iteration=2, cores='detect', reference="reference", method="chma"):
    """
    Execute Step B of the cHMA workflow: Create a canonical mesh and perform analyses.

    Parameters:
    -----------
    input_dir : str
        The output directory of the cHMA function.
    output_dir : str
        Directory to save output files. This should ideally be the same as the input directory.
    iteration : int
        Number of the last iteration from the canonical Holistic Morphometric Analysis desired for analysis.
    """

    # Configure logging
    log_file = os.path.join(output_dir, f"HMA_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Start Logging Progress
    start_time = time.time()
    logging.info(f"Starting isoHMA analysis - Input: {input_dir}, Output: {output_dir}")

    # Set the number of threads for SimpleITK
    if cores == 'detect':
        num_cores = multiprocessing.cpu_count() - 4  # Leave 4 cores free for system operations
    else:
        num_cores = cores
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(num_cores)
    logging.info(f"Using {num_cores} CPU threads for SimpleITK processing")

    create_directories1(output_dir)

    try:
        # =======================================================================================
        # B1. Load the canonical trabecular image
        # =======================================================================================
        logging.info("B1. Loading canonical trabecular image")

        trabecular_canonical_path = os.path.join(input_dir, "Canonical_Bone", "trabecular.tiff")
        canonical_path = os.path.join(input_dir, "Canonical_Bone", "canonical.tiff")

        if not os.path.exists(trabecular_canonical_path):
            logging.error(f"Canonical trabecular image not found at {trabecular_canonical_path}")
            return False

        if not os.path.exists(canonical_path):
            logging.error(f"Canonical bone image not found at {canonical_path}")
            return False

        averaged_trabecular = load_image(trabecular_canonical_path)
        canonical_bone = load_image(canonical_path)

        # =======================================================================================
        # B2. Check for existing mesh or create tetrahedral mesh
        # =======================================================================================

        logging.info("B2. Creating tetrahedral mesh of canonical trabecular image")

        canonical_mesh = None

        # Check for existing mesh files in different formats
        mesh_files = {
            'netgen': os.path.join(input_dir, "Canonical_Bone", "trabecular.vol"),
            'gmsh': os.path.join(input_dir, "Canonical_Bone", "trabecular.msh"),
            'tetgen_node': os.path.join(input_dir, "Canonical_Bone", "trabecular.node"),
            'tetgen_ele': os.path.join(input_dir, "Canonical_Bone", "trabecular.ele"),
            'vtk': os.path.join(input_dir, "Canonical_Bone", "trabecular.vtk")
        }

        # Try loading from NetGen .vol file
        if os.path.exists(mesh_files['netgen']):
            canonical_mesh = load_netgen(mesh_files['netgen'])
            if canonical_mesh and len(canonical_mesh.get("tetrahedra", [])) > 0:
                logging.info("Loaded mesh from NetGen .vol file")
                canonical_tet_collapse, canonical_volume_skew = evaluate_mesh_quality(canonical_mesh)
                if canonical_tet_collapse < 0.9 or canonical_volume_skew > 0.75:
                    logging.warning("Loaded NetGen mesh quality is poor. Will try other formats.")
                    canonical_mesh = None

        # Try loading from Gmsh .msh file
        if canonical_mesh is None and os.path.exists(mesh_files['gmsh']):
            canonical_mesh = load_gmsh_msh_mesh(mesh_files['gmsh'])
            if canonical_mesh and len(canonical_mesh.get("tetrahedra", [])) > 0:
                logging.info("Loaded mesh from Gmsh .msh file")
                canonical_tet_collapse, canonical_volume_skew = evaluate_mesh_quality(canonical_mesh)
                if canonical_tet_collapse < 0.9 or canonical_volume_skew > 0.75:
                    logging.warning("Loaded Gmsh mesh quality is poor. Will try other formats.")
                    canonical_mesh = None

        # Try loading from TetGen files
        if canonical_mesh is None and os.path.exists(mesh_files['tetgen_node']) and os.path.exists(
                mesh_files['tetgen_ele']):
            canonical_mesh = load_tetgen(mesh_files['tetgen_node'], mesh_files['tetgen_ele'])
            if canonical_mesh and len(canonical_mesh.get("tetrahedra", [])) > 0:
                logging.info("Loaded mesh from TetGen files")
                canonical_tet_collapse, canonical_volume_skew = evaluate_mesh_quality(canonical_mesh)
                if canonical_tet_collapse < 0.9 or canonical_volume_skew > 0.75:
                    logging.warning("Loaded TetGen mesh quality is poor. Will try other formats.")
                    canonical_mesh = None

        # Try loading from VTK file
        if canonical_mesh is None and os.path.exists(mesh_files['vtk']):
            canonical_mesh = load_vtk(mesh_files['vtk'])
            if canonical_mesh and len(canonical_mesh.get("tetrahedra", [])) > 0:
                logging.info("Loaded mesh from VTK file")
                canonical_tet_collapse, canonical_volume_skew = evaluate_mesh_quality(canonical_mesh)
                if canonical_tet_collapse < 0.9 or canonical_volume_skew > 0.75:
                    logging.warning("Loaded VTK mesh quality is poor. Will try to create a new mesh.")
                    canonical_mesh = None

        # If no valid mesh was loaded, create a new one
        if canonical_mesh is None or len(canonical_mesh.get("tetrahedra", [])) == 0:
            logging.info("Creating new solid tetrahedral mesh...")

            # First try TetGen (most reliable)
            try:
                canonical_mesh = create_solid_tetrahedral_mesh(
                    averaged_trabecular,
                    method="tetgen",
                    edge_length=1.0,  # 1mm edge length as in Bachmann et al. (2022)
                    output_dir=os.path.join(output_dir, "Canonical_Bone")
                )
                if canonical_mesh:
                    logging.info("Successfully created mesh with TetGen")
            except Exception as e:
                logging.warning(f"TetGen mesh creation failed: {e}")
                canonical_mesh = None

            # Try NetGen if TetGen failed
            if canonical_mesh is None:
                try:
                    canonical_mesh = create_solid_tetrahedral_mesh(
                        averaged_trabecular,
                        method="netgen",
                        edge_length=1.0,
                        output_dir=os.path.join(output_dir, "Canonical_Bone")
                    )
                    if canonical_mesh:
                        logging.info("Successfully created mesh with NetGen")
                except Exception as e:
                    logging.warning(f"NetGen mesh creation failed: {e}")
                    canonical_mesh = None

            # Try GMsh if both failed
            if canonical_mesh is None:
                try:
                    canonical_mesh = create_solid_tetrahedral_mesh(
                        averaged_trabecular,
                        method="gmsh",
                        edge_length=1.0,
                        output_dir=os.path.join(output_dir, "Canonical_Bone")
                    )
                    if canonical_mesh:
                        logging.info("Successfully created mesh with GMsh")
                except Exception as e:
                    logging.warning(f"GMsh mesh creation failed: {e}")
                    canonical_mesh = None

        # If meshing failed
        if canonical_mesh is None or len(canonical_mesh.get("tetrahedra", [])) == 0:
            logging.error("All mesh creation methods failed. Cannot continue with analysis.")
            return False

        # ===========================================================================================
        # B3. Evaluate mesh quality
        # ===========================================================================================

        logging.info("B3. Evaluating final mesh quality.")

        try:
            canonical_tet_collapse, canonical_volume_skew = evaluate_mesh_quality(canonical_mesh)
        except Exception as e:
            logging.error(f"Error evaluating mesh quality: {e}")
            canonical_tet_collapse, canonical_volume_skew = 0, 1  # Default values
            return False
        logging.info(
            f"Canonical Mesh within acceptable tet collapse: {canonical_tet_collapse:.4f}, and volume skew: {canonical_volume_skew:.4f}. Proceeding.")

        # ==========================================================================================
        # B4. Obtain and similarity register trabecular images for HMA
        # ==========================================================================================

        logging.info("B4. Obtaining trabecular images from directory")

        trab_dir = os.path.join(input_dir, "Similarity_Transform", "Trabecular")

        if not os.path.exists(trab_dir) or not os.listdir(trab_dir):
            logging.warning(
                "No trabecular images found. Registering original trabecular images to canonical bone for comparable HMA.")

            os.makedirs(os.path.join(output_dir, "Similarity_Transform", "Trabecular"), exist_ok=True)

            tp = os.path.join(input_dir, "Trabecular")
            tf = [f for f in os.listdir(tp) if f.endswith(".tiff") or f.endswith(".tif")]
            bone_names = [f.replace("_trabecular_resampled.tiff", "").replace("_trabecular_resampled.tif", "") for f
                             in tf]

            logging.info(f"Found {len(bone_names)} Bones")

            ref_path = os.path.join(input_dir, "Similarity_Transform", "Filled", f"{reference}.tiff")
            reference_bone = load_image(ref_path)

            for bone in bone_names:
                try:
                    logging.info(f"Processing {bone}...")

                    # Find trabecular image
                    trab_path = os.path.join(tp, f"{bone}_trabecular_resampled.tiff")
                    if not os.path.exists(trab_path):
                        trab_path = os.path.join(tp, f"{bone}_trabecular_resampled.tif")
                        if not os.path.exists(trab_path):
                            logging.warning(f"No trabecular image found for {bone}. Skipping.")
                            continue
                    trab_image = load_image(trab_path)

                    # Find transform files
                    sim_tfm_path = os.path.join(input_dir, "Similarity_Transform", "Transforms", f"{bone}.tfm")
                    sim_iter_tfm_path = os.path.join(input_dir, "Similarity_Transform", "Transforms2",
                                                     f"{bone}_iter{iteration}.tfm")

                    # Load transforms
                    sim_tfm = sitk.ReadTransform(sim_tfm_path)
                    sim_iter_tfm = sitk.ReadTransform(sim_iter_tfm_path)

                    # Apply Transforms
                    sim_1 = apply_transform(trab_image, sim_tfm, reference_bone)
                    sim_2 = apply_transform(sim_1, sim_iter_tfm, canonical_bone)

                    # Save trabecular image
                    stpath = os.path.join(trab_dir, f"{bone}.tiff")
                    sitk.WriteImage(sim_2, stpath)

                except Exception as e:
                    logging.error(f"Error processing {bone}: {e}")
                    traceback.print_exc()
                    continue
            logging.info(
                "Successfully ensured the compatability of each trabecular image associated with each bone. Proceeding.")
        else:
            trab_files = [f for f in os.listdir(trab_dir) if f.endswith(".tiff") or f.endswith(".tif")]
            bone_names = [f.replace(".tiff", "").replace(".tif", "") for f in trab_files]
            logging.info(f"Found {len(bone_names)} Bones")

        # =======================================================================================
        # B5. Create isotopological meshes and perform HMA for each bone
        # =======================================================================================

        logging.info("B5. Creating isotopological meshes and performing HMA for each bone")

        # Initialize results dictionary
        results = {
            'bone_name': [],
            'tet_collapse': [],
            'volume_skew': [],
            'mean_bv_tv': [],
            'mean_da': []
        }

        fabric_tensors = {}

        for bone in bone_names:
            try:
                logging.info(f"Processing {bone}...")

                if method == "chma":
                    # Find trabecular image
                    trab_path = os.path.join(trab_dir, f"{bone}.tiff")
                    if not os.path.exists(trab_path):
                        trab_path = os.path.join(trab_dir, f"{bone}.tif")
                        if not os.path.exists(trab_path):
                            logging.warning(f"No trabecular image found for {bone}. Skipping.")
                            continue

                    # Load trabecular image
                    trab_image = load_image(trab_path)

                    # Save similarity transform
                    transform_path = os.path.join(input_dir,
                                                  "Similarity_Transform",
                                                  "Transforms2",
                                                  f"{bone}_iter{iteration}.tfm"
                                                  )

                    similarity_transform = sitk.ReadTransform(transform_path)

                    trab_image = apply_transform(
                        trab_image,
                        similarity_transform,
                        canonical_bone
                    )

                    # Morph canonical mesh to individual trabecular space
                    morphed_mesh = isomorph(
                        canonical_mesh,
                        averaged_trabecular,
                        trab_image,
                        input_dir,
                        bone,
                        iteration,
                        debug_dir=os.path.join(output_dir, "Debug")
                    )

                elif method == "schma":
                    # Find trabecular image
                    trab_path = os.path.join(input_dir, "BSpline_Transform", "Trabecular", f"{bone}.tiff")
                    if not os.path.exists(trab_path):
                        trab_path = os.path.join(input_dir, "BSpline_Transform", "Trabecular", f"{bone}.tif")
                        if not os.path.exists(trab_path):
                            logging.warning(f"No trabecular image found for {bone}. Skipping.")
                            continue

                    trab_image = load_image(trab_path)

                    morphed_mesh = align_image_to_mesh(
                        trab_image, morphed_mesh
                    )

                if morphed_mesh is None:
                    logging.warning(f"Failed to morph mesh for {bone}. Skipping.")
                    continue

                # Save morphed mesh
                morphed_mesh_path = os.path.join(output_dir, "Isotopological_Meshes", f"{bone}_mesh.vtk")
                save_mesh_as_vtk(morphed_mesh, morphed_mesh_path)

                # Perform HMA analysis
                hma_results = improved_hma(
                    trab_image,
                    morphed_mesh,
                    grid_spacing=1.5,
                    sphere_diameter=3.0,
                    output_dir=os.path.join(output_dir, "HMA_Results", bone)
                )

                if hma_results is None:
                    logging.warning(f"HMA analysis failed for {bone}")
                    continue

                # Save HMA results including confidence and bone mask
                rbvtv_path = os.path.join(output_dir, "HMA_Results", f"{bone}_rbvtv.vtu")
                rda_path = os.path.join(output_dir, "HMA_Results", f"{bone}_rda.vtu")
                # confidence_path = os.path.join(output_dir, "HMA_Results", f"{bone}_confidence.vtu")
                # bone_path = os.path.join(output_dir, "HMA_Results", f"{bone}_bone.vtu")

                save_mesh_with_scalars(morphed_mesh, hma_results['vertex_r_bv_tv'], rbvtv_path)
                save_mesh_with_scalars(morphed_mesh, hma_results['vertex_r_da'], rda_path)
                # save_mesh_with_scalars(morphed_mesh, hma_results['confidence'], confidence_path)
                # save_mesh_with_scalars(morphed_mesh, hma_results['bone'].astype(float), bone_path)

                # Save CSV files for R
                csv_base_path = os.path.join(output_dir, "HMA_Results", f"{bone}_rbvtv")
                save_mesh_with_scalars_csv(morphed_mesh, hma_results['vertex_r_bv_tv'], csv_base_path)

                csv_base_path = os.path.join(output_dir, "HMA_Results", f"{bone}_rda")
                save_mesh_with_scalars_csv(morphed_mesh, hma_results['vertex_r_da'], csv_base_path)

                csv_base_path = os.path.join(output_dir, "HMA_Results", f"{bone}_confidence")
                save_mesh_with_scalars_csv(morphed_mesh, hma_results['confidence'], csv_base_path)

                csv_base_path = os.path.join(output_dir, "HMA_Results", f"{bone}_bone")
                save_mesh_with_scalars_csv(morphed_mesh, hma_results['bone'].astype(float), csv_base_path)

                # Evaluate mesh quality
                tet_collapse, volume_skew = evaluate_mesh_quality(morphed_mesh)

                # Add to results
                results['bone_name'].append(bone)
                results['tet_collapse'].append(tet_collapse)
                results['volume_skew'].append(volume_skew)
                results['mean_bv_tv'].append(hma_results['mean_bv_tv'])
                results['mean_da'].append(hma_results['mean_da'])
                fabric_tensors[bone] = hma_results['vertex_fabric']

                logging.info(f"Completed processing {bone}")

            except Exception as e:
                logging.error(f"Error processing {bone}: {e}")
                traceback.print_exc()
                continue

        # Save results to CSV
        fabric_tensors_path = os.path.join(output_dir, "Bone_Orientation.csv")
        save_mean_fabric_tensors(fabric_tensors, fabric_tensors_path)
        results_df = pd.DataFrame(results)
        results_path = os.path.join(output_dir, "hma_results.csv")
        results_df.to_csv(results_path, index=False)

        # Calculate total runtime
        end_time = time.time()
        total_time = end_time - start_time
        logging.info(f"isoHMA analysis completed in {total_time / 60:.2f} minutes")
        gc.collect()

        return True

    except Exception as e:
        logging.error(f"Error in isoHMA: {e}")
        traceback.print_exc()
        return False


##########################
# smesh function to assign scalar values to meshes
##########################

def smesh(input_file, pc_loadings, output_file, pc_name="PC1"):
    """
    Add scalar PC loadings to an existing mesh file (.vtk, .vol, or other formats).

    This function supports the cHMA methodology described in Bachmann et al. 2022
    for trabecular bone analysis by adding principal component loadings as scalar
    values to mesh vertices.

    Parameters:
    -----------
    input_file : str
        Path to input mesh file (.vtk, .vol, etc.)
    pc_loadings : array-like
        PC loadings (one per vertex)
    output_file : str
        Path to output VTK file
    pc_name : str
        Name of the PC component (default: "PC1")

    Returns:
    --------
    str
        Path to the created output file
    """
    import numpy as np
    import vtk
    from vtk.util import numpy_support
    import os

    print(f"Reading input mesh file: {input_file}")

    # Check file extension to determine the correct reader
    file_ext = os.path.splitext(input_file)[1].lower()

    if file_ext == '.vtk':
        # For VTK files
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(input_file)
        reader.Update()
        grid = reader.GetOutput()
    elif file_ext == '.vol':
        # For NetGen .vol files
        with open(input_file, 'r') as f:
            lines = f.readlines()

        # Find points section
        try:
            points_idx = lines.index("points\n")
            num_points = int(lines[points_idx + 1])

            # Read vertices
            vertices = np.zeros((num_points, 3))
            for i in range(num_points):
                line = lines[points_idx + 2 + i].strip()
                coords = np.array([float(x) for x in line.split()])
                vertices[i] = coords[:3]

            # Find volumeelements section
            vol_idx = lines.index("volumeelements\n")
            num_elements = int(lines[vol_idx + 1])

            # Read tetrahedra
            tetrahedra = np.zeros((num_elements, 4), dtype=np.int32)
            for i in range(num_elements):
                line = lines[vol_idx + 2 + i].strip()
                parts = np.array([int(x) for x in line.split()])

                if len(parts) >= 8:
                    # Format: type mat domain vol v1 v2 v3 v4
                    tetrahedra[i] = parts[4:8] - 1  # Convert to 0-based indexing
                elif len(parts) >= 5:
                    # Alternative format
                    tetrahedra[i] = parts[-4:] - 1  # Convert to 0-based indexing

            # Create VTK points
            points = vtk.vtkPoints()
            for i in range(len(vertices)):
                points.InsertNextPoint(vertices[i])

            # Create VTK cells
            cells = vtk.vtkCellArray()
            for i in range(len(tetrahedra)):
                tetra = vtk.vtkTetra()
                for j in range(4):
                    tetra.GetPointIds().SetId(j, int(tetrahedra[i, j]))
                cells.InsertNextCell(tetra)

            # Create unstructured grid
            grid = vtk.vtkUnstructuredGrid()
            grid.SetPoints(points)
            grid.SetCells(vtk.VTK_TETRA, cells)
        except Exception as e:
            print(f"Error reading NetGen file: {e}")
            return None
    else:
        # Try a generic reader for other formats
        try:
            reader = vtk.vtkDataSetReader()
            reader.SetFileName(input_file)
            reader.Update()
            grid = reader.GetOutput()

            if not grid or grid.GetNumberOfPoints() == 0:
                # Try unstructured grid reader
                reader = vtk.vtkUnstructuredGridReader()
                reader.SetFileName(input_file)
                reader.Update()
                grid = reader.GetOutput()

            if not grid or grid.GetNumberOfPoints() == 0:
                # Try polydata reader
                reader = vtk.vtkPolyDataReader()
                reader.SetFileName(input_file)
                reader.Update()
                grid = reader.GetOutput()

                # Convert polydata to unstructured grid if needed
                if grid and grid.GetNumberOfPoints() > 0:
                    ug = vtk.vtkUnstructuredGrid()
                    ug.SetPoints(grid.GetPoints())
                    ug.SetCells(grid.GetCellType(0), grid.GetPolys())
                    grid = ug
        except Exception as e:
            print(f"Error reading file with generic reader: {e}")
            return None

    # Check if the grid was successfully loaded
    if not grid or grid.GetNumberOfPoints() == 0:
        print("Failed to read mesh file or mesh file is empty")
        return None

    # Get number of vertices
    num_vertices = grid.GetNumberOfPoints()
    print(f"Mesh has {num_vertices} points and {grid.GetNumberOfCells()} cells")

    # Check dimensions
    if len(pc_loadings) != num_vertices:
        print(f"Warning: Number of PC loadings ({len(pc_loadings)}) doesn't match number of vertices ({num_vertices})")
        print("Attempting to adjust PC loadings...")

        if len(pc_loadings) > num_vertices:
            # Truncate PC loadings
            pc_loadings = pc_loadings[:num_vertices]
            print(f"Truncated PC loadings to {len(pc_loadings)} entries")
        else:
            # Pad PC loadings with zeros
            pc_loadings = np.pad(pc_loadings, (0, num_vertices - len(pc_loadings)), 'constant')
            print(f"Padded PC loadings to {len(pc_loadings)} entries")

    # Add PC loadings as point data
    loadings_array = numpy_support.numpy_to_vtk(
        num_array=pc_loadings,
        deep=True,
        array_type=vtk.VTK_FLOAT
    )
    loadings_array.SetName(pc_name)
    grid.GetPointData().AddArray(loadings_array)
    grid.GetPointData().SetActiveScalars(pc_name)

    # Write VTK file
    print(f"Writing VTK file: {output_file}")
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(grid)
    writer.Write()

    print("VTK file created successfully")
    return output_file

def MCA_Annotate(data_dir, muscle_files_dict):
    """
    :param data_dir: The input directory where the muscle annotations and 2D matrix of vertex and scalars are stored.
    :param muscle_files_dict: List object displaying the file names of the muscle annotations with the preferred name.
    :return: A File called Annotate.csv in the specified data_dir directory
    """
    print(f"Scanning directory: {data_dir}\n")

    # Load the main scalar files
    rbv_path = os.path.join(data_dir, "RBVSC.csv")
    rda_path = os.path.join(data_dir, "RDASC.csv")

    if not os.path.exists(rbv_path) or not os.path.exists(rda_path):
        print("Error: Could not find RBVSC.csv or RDASC.csv.")
        return

    print("Loading massive scalar CSVs... (This will take a few seconds)")
    rbv_df = pd.read_csv(rbv_path)
    rda_df = pd.read_csv(rda_path)

    # Safely set the index to the "ID" column (BU 18, Kebara2, etc.)
    if 'ID' in rbv_df.columns:
        rbv_df.set_index('ID', inplace=True)
    elif not isinstance(rbv_df.index[0], str):
        # Fallback if the column isn't explicitly named 'ID' but is the second column
        rbv_df.set_index(rbv_df.columns[1], inplace=True)

    if 'ID' in rda_df.columns:
        rda_df.set_index('ID', inplace=True)
    elif not isinstance(rda_df.index[0], str):
        rda_df.set_index(rda_df.columns[1], inplace=True)

    # Convert columns to a fast-lookup Set
    rbv_cols_set = set(rbv_df.columns)
    rda_cols_set = set(rda_df.columns)

    all_results = []

    for file_name, muscle_name in muscle_files_dict.items():
        muscle_path = os.path.join(data_dir, file_name)

        if not os.path.exists(muscle_path):
            print(f"Skipping {muscle_name}: Not found.")
            continue

        muscle_df = pd.read_csv(muscle_path)

        if 'vtkOriginalPointIds' in muscle_df.columns:
            raw_ids = muscle_df['vtkOriginalPointIds'].astype(float).astype(int).tolist()
        else:
            print(f"Warning: No valid Point ID column found in {file_name}.")
            continue

        # ---------------------------------------------------------
        # THE CUSTOM MAPPING ENGINE ('scalar_X' + 1)
        # ---------------------------------------------------------
        valid_cols_rbv = []
        valid_cols_rda = []

        for pid in raw_ids:
            # Shift the index by +1 to account for ParaView vs R counting
            shifted_id = pid + 1
            target_col = f"scalar_{shifted_id}"

            # Fallback just in case they actually match perfectly
            exact_col = f"scalar_{pid}"

            if target_col in rbv_cols_set:
                valid_cols_rbv.append(target_col)
            elif exact_col in rbv_cols_set:
                valid_cols_rbv.append(exact_col)

        for pid in raw_ids:
            shifted_id = pid + 1
            target_col = f"scalar_{shifted_id}"
            exact_col = f"scalar_{pid}"

            if target_col in rda_cols_set:
                valid_cols_rda.append(target_col)
            elif exact_col in rda_cols_set:
                valid_cols_rda.append(exact_col)

        print(f"Processing {muscle_name}: Matched {len(valid_cols_rbv)} / {len(raw_ids)} vertices.")

        if len(valid_cols_rbv) == 0:
            print(f"  -> ERROR: Could not find matching columns for {muscle_name}.")
            continue

        # Calculate Means
        rbv_means = rbv_df[valid_cols_rbv].mean(axis=1)
        rda_means = rda_df[valid_cols_rda].mean(axis=1)

        temp_df = pd.DataFrame({
            'ID': rbv_means.index,
            'Muscle_Region': muscle_name,
            'RBV_Mean': rbv_means.values,
            'RDA_Mean': rda_means.values
        })

        # Categorize
        labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        try:
            temp_df['RBV_Intensity'] = pd.qcut(temp_df['RBV_Mean'].dropna(), q=5, labels=labels)
            temp_df['RDA_Intensity'] = pd.qcut(temp_df['RDA_Mean'].dropna(), q=5, labels=labels)
        except ValueError:
            temp_df['RBV_Intensity'] = "N/A"
            temp_df['RDA_Intensity'] = "N/A"

        all_results.append(temp_df)

    if not all_results:
        print("\nNo valid data to combine.")
        return

    long_df = pd.concat(all_results, ignore_index=True)
    wide_df = long_df.pivot(index='ID', columns='Muscle_Region',
                            values=['RBV_Intensity', 'RDA_Intensity', 'RBV_Mean', 'RDA_Mean'])
    wide_df.columns = [f"{muscle}_{metric}" for metric, muscle in wide_df.columns]
    wide_df.reset_index(inplace=True)

    # Make sure we didn't accidentally keep the 'Unnamed: 0' column as a specimen ID
    if 'Unnamed: 0' in wide_df.columns:
        wide_df.drop(columns=['Unnamed: 0'], inplace=True)

    output_path = os.path.join(data_dir, "Categorized_Muscle_Intensities_Wide.csv")
    wide_df.to_csv(output_path, index=False)

    print(f"\nSuccess! MCA Table saved to:\n{output_path}")

"""
Canonical Holistic Morphometric Analysis (cHMA) Implementation

This module implements the cHMA methodology for trabecular bone analysis 
as described in Bachmann et al. 2022.

The workflow consists of two main steps:
1. Step A: Create a canonical bone using filled binary images
2. Step B: Create a canonical mesh and perform analyses

Author: Keeling, Brian Anthony
"""
import os
import numpy as np
import SimpleITK as sitk
import logging
import gc
import traceback
import time
import multiprocessing
import pandas as pd
import skimage.measure
import pyvista as pv
import skimage.measure
from vtk.util.numpy_support import numpy_to_vtk
from scipy.spatial.transform import Rotation as R
from datetime import datetime
from skimage import measure
from scipy import ndimage

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
            logging.error(f"File does not exist: {file_path}")
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

    from scipy import ndimage
    """Fill small holes in a binary image using morphological closing.
    
    Args:
        binary_image_sitk (sitk.Image): Input binary image (1s and 0s).
        radius_mm (float): Radius in millimeters for closing operation.
    
    Returns:
        sitk.Image: Binary image after morphological closing.
    """
    # Get voxel spacing from input image
    spacing = binary_image_sitk.GetSpacing()

    # Convert mm radius to voxel units
    radius_voxels = [int(radius_mm / s + 0.5) for s in spacing]
    
    # Convert image to numpy array
    array = sitk.GetArrayFromImage(binary_image_sitk).astype(np.uint8)
    
    # Create spherical structuring element
    struct = ndimage.generate_binary_structure(3, 1)
    struct = ndimage.iterate_structure(struct, max(radius_voxels))
    
    # Perform morphological closing (dilate + erode)
    closed = ndimage.binary_closing(array, structure=struct).astype(np.uint8)
    
    # Convert back to SimpleITK image
    filled_image = sitk.GetImageFromArray(closed)
    filled_image.CopyInformation(binary_image_sitk)  # Preserve spacing/origin/direction
    return filled_image

def prepare_condyle_image(image):
    """
    Process condyle images by removing background noise and normalizing values.
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
        logging.error(f"Error in prepare_condyle_image: {e}")
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
        registration_method.SetInterpolator(sitk.sitkNearestNeighbor)
        registration_method.SetInitialTransform(initial_transform, inPlace=True)
        
        # Simplified optimizer settings
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=0.9,
            numberOfIterations=100
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

def average_images(image_list, method=2):
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
        avg_image = prepare_condyle_image(avg_image)
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

def bspline_registration(fixed_image, moving_image, control_point_grid=[1.5, 1.5, 1.5], scale_factor=5):
    """Perform B-spline registration with conservative parameters to avoid unrealistic deformations."""
    try:
        logging.info("Starting B-spline registration.")

        # Convert images to float for registration
        #fixed_image = prepare_condyle_image(fixed_image)
        #fixed_image = clean_intensity_image(fixed_image)
        fixed_mask = fill(fixed_image)
        fixed_reg = sitk.Mask(fixed_image, fixed_mask)

        #moving_image = prepare_condyle_image(moving_image)
        #moving_image = clean_intensity_image(moving_image)
        moving_mask = fill(moving_image)
        moving_image = sitk.Mask(moving_image, moving_mask)
        
        fixed_image = sitk.Cast(fixed_mask, sitk.sitkFloat32)
        moving_image = sitk.Cast(moving_mask, sitk.sitkFloat32)
        
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
        #registration_method.SetMetricAsANTSNeighborhoodCorrelation(radius=2)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.75)
        
        #registration_method.SetMetricFixedMask(fixed_image)

        # Use Nearest Neighbor Interpolation
        registration_method.SetInterpolator(sitk.sitkBSpline)
        
        # Add transform constraints to prevent extreme deformations
        #registration_method.SetOptimizerAsLBFGSB(
        #    gradientConvergenceTolerance=1e-9,
        #    numberOfIterations=100, 
        #    maximumNumberOfCorrections=25,
        #    maximumNumberOfFunctionEvaluations=2500,
        #    costFunctionConvergenceFactor=1e+8
        #) 

        registration_method.SetOptimizerAsLBFGS2(solutionAccuracy=1e-4, numberOfIterations=150, deltaConvergenceTolerance=0.01)

        #registration_method.SetOptimizerScalesFromPhysicalShift()
        
        # Use multi-resolution approach with more aggressive smoothing
        #registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        #registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        #registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        #registration_method.AddCommand(sitk.sitkIterationEvent, lambda: iteration_callback(registration_method))
        
        # Execute registration
        bspline_transform = registration_method.Execute(fixed_image, moving_image)

        #print("\nOptimizer's stopping condition, {0}".format(registration_method.GetOptimizerStopConditionDescription()))
        
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
            maximumNumberOfIterations=100,  # Bachmann uses 25
            meanErrorToleranceThreshold=0.0001,  # From Bachmann's paper
            maxErrorToleranceThreshold=0.01,  # From Bachmann's paper
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
        rfix = resample_image(fixed_image, scale_factor)
        fixed_mask = fill(rfix)
        fixed_mask = sitk.Cast(fixed_mask, sitk.sitkUInt8)
        
        rmove = resample_image(moving_image, scale_factor)
        moving_mask = fill(rmove)
        moving_mask = sitk.Cast(moving_mask, sitk.sitkUInt8)

        # Compute Dice coefficient        
        dice_filter = sitk.LabelOverlapMeasuresImageFilter()
        dice_filter.Execute(fixed_mask, moving_mask)
        dice = dice_filter.GetDiceCoefficient()

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

def cHMA(input_dir, output_dir, reference_name="reference", scale_factor=3, max_iterations=5, cp=[1,1,1], cores='detect'):
    """
    Perform Canonical Holistic Morphometric Analysis (cHMA) on trabecular bone images.
    
    Parameters:
    input_dir (str): Directory containing filled and trabecular binary images
    output_dir (str): Directory to save output files
    reference_name (str): Name of the reference condyle (default: "reference")
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
        # A1: Get a list of condyle names from filled images
        # =====================================================================
        
        filled_dir = os.path.join(input_dir, "Filled")
        filled_files = [f for f in os.listdir(filled_dir) if f.endswith(".tiff") or f.endswith(".tif")]
        condyle_names = [f.split("_filled_resampled")[0] for f in filled_files]
        
        logging.info(f"Found {len(condyle_names)} condyles: {', '.join(condyle_names)}")
        
        # Ensure reference condyle is in the list
        if f"{reference_name}_filled_resampled.tiff" not in filled_files:
            logging.error(f"Reference condyle {reference_name} not found in input directory")
            return False

        # =====================================================================
        # A2: Initial similarity transformation
        # =====================================================================
    
        logging.info(f"A2. Performing initial similarity transformation using reference: {reference_name}")
        
        # Load reference image
        reference_image_path = os.path.join(filled_dir, f"{reference_name}_filled_resampled.tiff")
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
            
            # Apply similarity transformation to each condyle
            for condyle in condyle_names:
                try:
                    # Load original image
                    moving_image_path = os.path.join(filled_dir, f"{condyle}_filled_resampled.tiff")
                    moving_image = load_image(moving_image_path)
                    
                    if moving_image is None:
                        logging.warning(f"Failed to load {condyle}. Skipping.")
                        continue
                    
                    # Resample moving image for registration
                    moving_image_rescaled = resample_image(moving_image, scale_factor)
                    if moving_image_rescaled is None:
                        logging.warning(f"Failed to resample {condyle}. Skipping.")
                        continue
    
                    logging.info(f"Starting Similarity Registration for {condyle}.")
                    
                    # Register rescaled moving image to rescaled reference
                    transform = similarity_registration(reference_image_rescaled, moving_image_rescaled)
                    
                    if transform is None:
                        logging.warning(f"Similarity registration failed for {condyle}. Skipping.")
                        continue
                    
                    # Save transform
                    transform_path = os.path.join(output_dir, "Similarity_Transform", "Transforms", f"{condyle}.tfm")
                    sitk.WriteTransform(transform, transform_path)
                    
                    # Apply transform to ORIGINAL (non-rescaled) image
                    transformed_image = apply_transform(moving_image, transform, reference_image)
                    
                    transformed_image_path = os.path.join(output_dir, "Similarity_Transform", "Filled", f"{condyle}.tiff")
                    sitk.WriteImage(transformed_image, transformed_image_path)
                    
                    # Store transform for averaging
                    similarity_transforms[condyle] = transform
                    
                    logging.info(f"Similarity registered {condyle} to reference")
                    
                    # Clean up
                    del moving_image, moving_image_rescaled, transformed_image
                    gc.collect()
                    
                except Exception as e:
                    logging.error(f"Error processing {condyle} for similarity transform: {e}")
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
                reference_avg = prepare_condyle_image(reference_avg)
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

            # Store condyle specific metrics
            metrics = {
                "dice": [],
                "hd": [],
                "msd": []
            }
            
            # Initialize lists to store transforms and transformed images
            similarity_transforms2 = {}
            bspline_transforms = {}
            transformed_images = []
            
            # Condyle Iteration
            for condyle in condyle_names:
                try:
                    # Load original filled image
                    original_image_path = os.path.join(output_dir, "Similarity_Transform", "Filled", f"{condyle}.tiff")
                    original_image = load_image(original_image_path)
                    
                    if original_image is None:
                        logging.warning(f"Failed to load {condyle}. Skipping.")
                        continue
                    
                    # Resample for faster processing
                    original_image_rescaled = resample_image(original_image, scale_factor)
                    
                    if original_image_rescaled is None:
                        logging.warning(f"Failed to resample {condyle}. Skipping.")
                        continue
                    #======================================================
                    # Apply similarity transformation using RESCALED images
                    similarity_transform = similarity_registration(
                        canonical_image_rescaled, 
                        original_image_rescaled
                    )
                    
                    if similarity_transform is None:
                        logging.warning(f"Similarity registration failed for {condyle} in iteration {iteration}. Skipping.")
                        continue

                    #======================================================
                    # Save similarity transform
                    transform_path = os.path.join(
                        output_dir, "Similarity_Transform", "Transforms2", f"{condyle}_iter{iteration}.tfm"
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
                        logging.warning(f"Failed to apply similarity transform for {condyle} in iteration {iteration}. Skipping.")
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
                        logging.warning(f"B-spline registration failed for {condyle} in iteration {iteration}. Skipping.")
                        continue
                        
                    #======================================================
                    # Save B-spline transform
                    
                    bspline_path = os.path.join(
                        output_dir, "BSpline_Transform", "Transforms", f"{condyle}_iter{iteration}.tfm"
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
                        logging.warning(f"Failed to apply B-spline transform to full-res image for {condyle}. Skipping.")
                        continue

                    logging.info(f"Successfully B-Spline Transformed, {condyle}, for iteration: {iteration}.")
                    
                    # Save transformed image
                    transformed_path = os.path.join(output_dir, "BSpline_Transform", "Filled", f"{condyle}_iter{iteration}.tiff")
                    sitk.WriteImage(bspline_full_res, transformed_path)

                    #========================================================
                    # Add to lists for averaging
                    similarity_transforms2[condyle] = similarity_transform
                    bspline_transforms[condyle] = bspline_transform

                    #========================================================
                    # Compute metrics
                    dice, hd, msd = compute_metrics(canonical_image, bspline_full_res)
                    if dice is not None:
                        logging.info(f"Metrics for {condyle}: Dice={dice:.4f}, HD={hd:.4f}, MSD={msd:.4f}")
                        metrics["dice"].append(dice)
                        metrics["hd"].append(hd)
                        metrics["msd"].append(msd)
                    
                    logging.info(f"Successfully processed {condyle} in iteration {iteration}")
                    
                    # Clean up
                    del original_image, original_image_rescaled, similarity_image_rescaled
                    del similarity_image, bspline_full_res
                    gc.collect()
                    
                except Exception as e:
                    logging.error(f"Error processing {condyle} in iteration {iteration}: {e}")
                    traceback.print_exc()
                    continue

            # =====================================================================
            # A6: Create the Canonical Bone
            # =====================================================================

            logging.info(f"A6. Creating Canonical Bone for Iteration {iteration}")
            
            #==================================================
            # Average BSpline transformed images
            gc.collect()

           # try: 
           #     for condyle in condyle_names:
           #         transformed_path = os.path.join(output_dir, "BSpline_Transform", "Filled", f"{condyle}_iter{iteration}.tiff")
           #         bspline_full_res = load_image(transformed_path)
           #         transformed_images.append(bspline_full_res)
           #         del bspline_full_res
           # except Exception as e:
           #         logging.error(f"Error retrieving transformed condyle images in iteration {iteration}: {e}")
           #         traceback.print_exc()
            
            # Check if we have enough transformed images
           # if len(transformed_images) < len(condyle_names) // 2:
           #     logging.error(f"Too few successful transformations in iteration {iteration}. Stopping.")
           #     break
            
           # gc.collect()
           # averaged_image = average_images(transformed_images)
            
           # if averaged_image is None:
           #     logging.error(f"Failed to average transformed images in iteration {iteration}")
           #     break
           # del transformed_images
            
           # logging.info(f"Successfully averaged transformed images for iteration {iteration}")

            #==================================================
            # Average BSpline Transforms through displacement field conversion
            gc.collect()
            logging.info("Converting BSpline Transforms to Displacement Fields")
            
            try:
                # Make displacement fields from B-spline transforms
                displacement_fields = []
                for condyle, bspline_transform in bspline_transforms.items():
                    try:
                        displacement_field = sitk.TransformToDisplacementField(
                            bspline_transform,
                            sitk.sitkVectorFloat64,
                            canonical_image.GetSize(),
                            canonical_image.GetOrigin(),
                            canonical_image.GetSpacing(),
                            canonical_image.GetDirection()
                        )
                        displacement_fields.append(displacement_field)
                    except Exception as e:
                        logging.warning(f"Error converting B-spline to displacement field for {condyle}: {e}")
                        continue
            
                logging.info("Creating averaged inverted transform")
    
                if displacement_fields:
                    #============================
                    # Average displacement fields
                    avg_displacement_field = average_displacement_fields(displacement_fields)
                    del displacement_fields
                    #============================
                    # Invert with better parameters
                    inverted_transform = invert_displacement_field(avg_displacement_field)
                    logging.info("Successfully created averaged inverted transform")
                    #============================
                    # Apply transform with extra safeguards
                    logging.info("Creating canonical image")
                    #inverted_image = apply_transform(averaged_image, inverted_transform, interpolator=sitk.sitkNearestNeighbor)
                    inverted_image = apply_transform(canonical_image, inverted_transform, interpolator=sitk.sitkNearestNeighbor)
                    #============================
                    # Verify inverted image has content
                    stats = sitk.StatisticsImageFilter()
                    stats.Execute(inverted_image)
                    if stats.GetSum() < 1:
                        logging.warning("Inverted image is empty! Using averaged image instead.")
                        new_canonical_image = averaged_image
                    else:
                        new_canonical_image = inverted_image
                        logging.info("Successfully created the canonical image")
                else:
                    logging.error("No valid displacement fields for inversion. Stopping analysis.")
                    break
                    
            except Exception as e:
                logging.warning(f"Error in displacement field processing: {e}")
                logging.warning("Using cleaned averaged image as canonical")
                new_canonical_image = averaged_image

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
        condyle_names = [f.split("_filled_resampled")[0] for f in filled_files]
        
        reference_image_path = os.path.join(filled_dir, f"{reference_name}_filled_resampled.tiff")
        reference_image = chma.load_image(reference_image_path)
                
        trabecular_dir = os.path.join(input_dir, "Trabecular")
        transformed_trabecular_images = []
        
        for condyle in condyle_names:
            # Load trabecular image
            trabecular_path = os.path.join(trabecular_dir, f"{condyle}_trabecular_resampled.tiff")
            trabecular_image = chma.load_image(trabecular_path)
        
            # Load initial similarity transform to align condyles
            first_sim_path = os.path.join(
                output_dir, "Similarity_Transform", "Transforms",
                f"{condyle}.tfm"
            )
            
            # Load final transforms from last iteration
            last_iteration = 2
            similarity_path = os.path.join(
                output_dir, "Similarity_Transform", "Transforms2", 
                f"{condyle}_iter{last_iteration}.tfm"
            )
            bspline_path = os.path.join(
                output_dir, "BSpline_Transform", "Transforms", 
                f"{condyle}_iter{last_iteration}.tfm"
            )
        
            # Read transform paths
            first_sim_transform = sitk.ReadTransform(first_sim_path)
            similarity_transform = sitk.ReadTransform(similarity_path)
            bspline_transform = sitk.ReadTransform(bspline_path)
        
            #=========================================================
            # Apply transforms
            
            # First similarity
            trabecular_first = chma.apply_transform(trabecular_image, first_sim_transform, reference_image)
            
            # Second similarity
            trabecular_similarity = chma.apply_transform(trabecular_first, similarity_transform, canonical_image)
        
            tspath = os.path.join(output_dir, "Similarity_Transform", "Trabecular", f"{condyle}.tiff")
            sitk.WriteImage(trabecular_similarity, tspath)
            
            # Then B-spline
            trabecular_bspline = chma.apply_transform(trabecular_similarity, bspline_transform, canonical_image)
        
            # Save transformed trabecular image
            transformed_path = os.path.join(output_dir, "BSpline_Transform", "Trabecular", f"{condyle}.tiff")
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
        averaged_trabecular = chma.average_images(transformed_trabecular_images) 
        
        del transformed_trabecular_images
        
        if averaged_trabecular is None:
            logging.error("Failed to average transformed trabecular images")
            return False
            
        # Save canonical trabecular image
        trabecular_canonical_path = os.path.join(output_dir, "Canonical_Bone", "trabecular.tiff")
        sitk.WriteImage(averaged_trabecular, trabecular_canonical_path)
        
        # Calculate total runtime
        end_time = time.time()
        total_time = end_time - start_time
        logging.info(f"cHMA analysis completed in {total_time/60:.2f} minutes")
        gc.collect()
        
        return True
        
    except Exception as e:
        logging.error(f"Unexpected error in cHMA function: {e}")
        traceback.print_exc()
        return False
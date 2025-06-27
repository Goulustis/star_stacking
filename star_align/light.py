import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, variance
from loguru import logger
from tqdm import tqdm
from functools import partial

import os
import cv2
import glob
import os.path as osp


from star_align.utils import parallel_map


def classify_stars(image, patch_size=5, sigma_threshold=2.0):
    """
    Classify pixels as background or stars using local statistics.
    
    Parameters:
    -----------
    image : 2D numpy array
        Input grayscale image
    patch_size : int
        Size of the local patch for computing statistics
    sigma_threshold : float
        Number of standard deviations for threshold
        
    Returns:
    --------
    mask : 2D numpy array (boolean)
        True for star pixels, False for background pixels
    visualization : 2D numpy array
        Original image with stars highlighted
    """
    # Convert image to float if it's not already
    image_float = image.astype(float)
    
    # Compute local mean using uniform filter
    local_mean = uniform_filter(image_float, size=patch_size)
    
    # Compute local variance
    # Calculate the squared difference first
    squared_diff = (image_float - local_mean) ** 2
    # Apply uniform filter to get local variance
    local_variance = uniform_filter(squared_diff, size=patch_size)
    
    # Compute local standard deviation
    local_std = np.sqrt(local_variance)
    
    # Compute the absolute difference between the pixel value and the local mean
    diff = np.abs(image_float - local_mean)
    
    # Classify pixels as stars if they deviate more than threshold*std from the mean
    star_mask = diff > (sigma_threshold * local_std)
    
    # Create visualization
    visualization = np.copy(image_float)
    
    # Normalize the image for display
    visualization = (visualization - np.min(visualization)) / (np.max(visualization) - np.min(visualization))
    return star_mask
    

def fit_light_pollution_model_rgb(image, mask, patch_size=32):
    """
    Fit a linear gradient model to background pixels for light pollution estimation in RGB images.
    
    Parameters:
    -----------
    image : ndarray
        Input RGB image (3D array with shape [height, width, 3])
    mask : ndarray
        Binary mask where 1 indicates background pixels and 0 indicates star pixels
    patch_size : int
        Size of patches to process the image in
    
    Returns:
    --------
    model : ndarray
        Estimated light pollution model with same shape as input image
    """
    h, w, channels = image.shape
    model = np.zeros_like(image, dtype=float)
    
    # Process each color channel separately
    for c in range(channels):
        channel_image = image[:, :, c]
        channel_model = np.zeros((h, w), dtype=float)
        
        # Process image in patches
        for y in tqdm(range(0, h, patch_size),desc="fitting pollution"):
            for x in range(0, w, patch_size):
                # Define current patch boundaries
                y_end = min(y + patch_size, h)
                x_end = min(x + patch_size, w)
                
                # Extract current patch
                patch = channel_image[y:y_end, x:x_end]
                patch_mask = mask[y:y_end, x:x_end]
                
                # Fit model to current patch
                patch_model = fit_patch_model(patch, patch_mask)
                
                # Store the model for current patch
                channel_model[y:y_end, x:x_end] = patch_model
        
        # Store the channel model
        model[:, :, c] = channel_model
    
    return model


def fit_patch_model(patch:np.ndarray, patch_mask:np.ndarray):
    """
    Fit a linear gradient model to a single patch using Weighted Least Squares.
    
    Parameters:
    -----------
    patch : ndarray
        Image patch (2D array)
    patch_mask : ndarray
        Binary mask for the patch
    
    Returns:
    --------
    model_patch : ndarray
        Estimated model for the patch
    """
    h, w = patch.shape
    n_pixels = h * w
    
    # Flatten the patch and mask
    y = patch.flatten()
    weights = patch_mask.flatten()
    
    # If all pixels are masked, return zeros
    if np.sum(weights) < 3:  # Need at least 3 points to fit a plane
        return np.zeros_like(patch)
    
    # Create feature matrix X
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()
    
    X = np.column_stack([np.ones(n_pixels), x_coords, y_coords])
    
    # Create weight matrix W (diagonal)
    W = np.diag(weights)
    
    # Solve for beta using the normal equations: (X^T W X) beta = X^T W y
    XtW = X.T @ W
    XtWX = XtW @ X
    XtWy = XtW @ y
    
    # Handle potential singularity with pseudo-inverse
    try:
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]
    
    # Compute the model: X * beta
    model_values = X @ beta
    
    # Reshape back to patch dimensions
    model_patch = model_values.reshape(h, w)
    
    return model_patch


def remove_light_single(img_f:str, out_dir:str):
    logger.info(f"Processing {img_f}")
    image = cv2.imread(img_f, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    
    # Run the classification
    mask = classify_stars(image, patch_size=256, sigma_threshold=2)
    mask = ~mask  # Invert mask to get background pixels
    image = cv2.imread(img_f).astype(np.float32) / 255.0
    
    base_name = f"{osp.basename(img_f).split('.')[0]}"
    model_f = osp.join(out_dir, f"estimated_model_{base_name}.npy")
    if not osp.exists(model_f):
        # Fit light pollution model
        estimated_model = fit_light_pollution_model_rgb(image, mask, patch_size=24)
        # Save estimated light pollution as .npy
        np.save(model_f, estimated_model.astype(np.float32))
    else:
        estimated_model = np.load(model_f)
    
    # Show corrected image
    corrected_image = image - estimated_model

    corrected_image = (corrected_image*(2**16 - 1)).astype(np.uint16)
    cv2.imwrite(osp.join(out_dir, f"corrected_image_{base_name}.png"), corrected_image)


def remove_light_pollution(img_dir:str = "raw_imgs/Stacking 11pm-2", 
                           work_dir:str = "pre_proc"):


    out_dir = osp.join(work_dir, osp.basename(img_dir))
    os.makedirs(out_dir, exist_ok=True)

    img_fs = sorted(glob.glob(f"{img_dir}/*.JPG"))
    
    rm_light_fn = partial(remove_light_single, out_dir=out_dir)
    parallel_map(rm_light_fn, img_fs, max_threads=8, show_pbar=True, desc="Processing images")
    return out_dir


if __name__ == "__main__":
    remove_light_pollution()
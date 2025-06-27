import cv2
import numpy as np
from scipy.optimize import least_squares

def calculate_homography(pts1, pts2, f, img_size):
    K = np.array([
        [f, 0, img_size[0] / 2],
        [0, f, img_size[1] / 2],
        [0, 0, 1]
    ])
    
    pts1_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), K, None)
    pts2_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), K, None)
    
    H, _ = cv2.findHomography(pts1_norm, pts2_norm, cv2.RANSAC)
    return H

def reprojection_error_focal_length(f, pts_pairs, img_size):
    total_error = []
    
    for pts1, pts2 in pts_pairs:
        H = calculate_homography(pts1, pts2, f, img_size)
        
        # Project points using the calculated homography
        pts1_homog = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
        pts2_proj = (H @ pts1_homog.T).T
        
        # Normalize homogeneous coordinates
        pts2_proj /= pts2_proj[:, 2].reshape(-1, 1)
        
        # Calculate reprojection error
        errors = pts2_proj[:, :2] - pts2
        total_error.extend(errors.ravel())
        
    return np.array(total_error)

# Example image size (width, height)
# img_size = (1920, 1080)
img_size = cv2.imread("pre_proc/Stacking 10pm-1/corrected_image_DSC03803.JPG").shape[:2][::-1]  # (width, height)

# Prepare your point correspondences for multiple image pairs
# pts_pairs is a list of tuples, each containing two arrays of corresponding points from two images
# e.g., [(pts1_a, pts2_a), (pts1_b, pts2_b), ...]

# Example placeholder for point correspondences
pts_pairs = [
    # (pts1, pts2) for the first image pair
    (np.array([[100, 150], [200, 250], [300, 350], [400, 450]]),
     np.array([[120, 170], [220, 270], [320, 370], [420, 470]])),
    # Next pair...
]

# Initial guess for focal length
initial_focal_length = 1000

# Optimize focal length
result = least_squares(
    reprojection_error_focal_length,
    x0=initial_focal_length,
    args=(pts_pairs, img_size)
)

optimized_focal_length = result.x[0]
print("Optimized Focal Length:", optimized_focal_length)

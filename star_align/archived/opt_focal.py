import cv2
import numpy as np
from scipy.optimize import least_squares
from typing import Tuple
import scipy.optimize as optim

def calculate_homography(pts1, pts2, f: np.ndarray, img_size: Tuple[int, int]):
    """Calculate homography between two sets of points using the provided focal length."""
    f = f if type(f) == float else f.item()
    K = np.array([
        [f, 0, img_size[0] / 2],
        [0, f, img_size[1] / 2],
        [0, 0, 1]
    ])
    
    # Convert points to normalized coordinates
    pts1_homog = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
    pts2_homog = np.hstack([pts2, np.ones((pts2.shape[0], 1))])
    
    # Apply inverse of K to normalize points
    pts1_norm = (np.linalg.inv(K) @ pts1_homog.T).T
    pts2_norm = (np.linalg.inv(K) @ pts2_homog.T).T
    
    # Extract x,y coordinates (normalized)
    pts1_norm = pts1_norm[:, :2]
    pts2_norm = pts2_norm[:, :2]
    
    # Find homography between normalized points
    H_norm, _ = cv2.findHomography(pts1_norm, pts2_norm, cv2.RANSAC, 5.0)
    
    if H_norm is None:
        return None
    
    # Transform homography back to pixel coordinates
    H = K @ H_norm @ np.linalg.inv(K)
    
    return H

def reprojection_error_focal_length(f, pts_pairs, img_size):
    """Calculate reprojection error for a given focal length."""
    if f <= 0:  # Ensure focal length is positive
        return np.array([1e6])  # Return a large error for invalid focal lengths
    
    total_error = []
    
    for pts1, pts2 in pts_pairs:
        H = calculate_homography(pts1, pts2, f, img_size)
        
        if H is None:
            # If homography estimation fails, add a large error
            total_error.extend([1e3] * (pts1.shape[0] * 2))
            continue
        
        # Project points using the calculated homography
        pts1_homog = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
        pts2_proj = (H @ pts1_homog.T).T
        
        # Normalize homogeneous coordinates
        pts2_proj = pts2_proj[:, :2] / pts2_proj[:, 2:]
        
        # Calculate squared reprojection errors
        errors = np.sqrt(np.sum((pts2_proj - pts2)**2, axis=1))
        total_error.extend(errors)
        
    return np.array(total_error).mean()


def opt_focal(img_f:str, 
              tracks: np.ndarray, 
              n_pairs:int = 30,
              foc_init:float = 1000.):
    img = cv2.imread(img_f)
    h, w = img.shape[:2]
    img_size = (w, h)

    train_pairs = []
    idxs = np.random.choice(len(track)-1, (n_pairs, 2), replace=True)
    for idx1, idx2  in idxs:
        if idx1 == idx2:
            continue
        pts1, pts2 = tracks[idx1], tracks[idx2]
        val_cond = ~np.isnan(pts1[:,0]) & ~np.isnan(pts2[:, 0])
        pts1, pts2 = pts1[val_cond], pts2[val_cond]
        train_pairs.append((pts1, pts2))

    # result = least_squares(
    #     reprojection_error_focal_length,
    #     x0=foc_init,
    #     args=(train_pairs, img_size)
    # )

    # Set bounds for focal length
    min_focal = 0.1 * max(img_size)  # Lower bound
    max_focal = 5.0 * max(img_size)  # Upper bound
    # bounds = [(min_focal, max_focal)]
    bounds = [(foc_init, max_focal)]

    # result = optim.minimize(
    #     reprojection_error_focal_length,
    #     x0=[foc_init],  # Powell's method expects an array
    #     args=(train_pairs, img_size),
    #     method='Powell',
    #     bounds=bounds,
    #     options={
    #         'disp': True,  # Display convergence messages
    #         'ftol': 1e-6,  # Function tolerance for convergence
    #         'maxiter': 100  # Maximum iterations
    #     }
    # )

    # print(f"Optimization result: {result.message}")
    # print(f"Initial focal length: {foc_init}, Optimized focal length: {result.x[0]}")
    # print(f"Initial cost: {np.mean(reprojection_error_focal_length(foc_init, train_pairs, img_size)**2)}")
    # print(f"Final cost: {np.mean(result.fun**2)}")
    
    # print(f"final error:{reprojection_error_focal_length(result.x[0], train_pairs, img_size)}")

    initial_error = reprojection_error_focal_length(foc_init, train_pairs, img_size)
    print(f"Initial reprojection error: {initial_error}")

    # Using brute force search to find optimal focal length
    result = optim.brute(
        reprojection_error_focal_length,
        ranges=bounds,  # Specify the range for the focal length
        args=(train_pairs, img_size),
        Ns=100,  # Adjust the resolution of search
        full_output=True,
        finish=None  # We do not use a finish procedure
    )
    optimized_focal = result[0]
    final_error = reprojection_error_focal_length(optimized_focal, train_pairs, img_size)
    
    print(f"Optimized focal length: {optimized_focal}")
    print(f"opt error {result[1]}")
    print(f"Final MSE: {final_error}")
    print(f"Error reduction: {(initial_error - final_error) / initial_error * 100:.2f}%")


    # return result.x[0]
    return result[0]

if __name__ == "__main__":
    np.random.seed(9000)
    track = np.load("track.npy")
    img_f = "pre_proc/Stacking 11pm-2/corrected_image_DSC03687.JPG"

    focal = opt_focal(img_f, track, foc_init=90., n_pairs=16)
    print(focal)
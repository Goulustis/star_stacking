import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from loguru import logger
import glob

from star_finder.findStars import find

def find_matching_min(stars1, stars2, min_dist=30, ret_msk:bool=False):


    dist = (stars1[:, np.newaxis] - stars2[np.newaxis, :])**2
    dist = np.sqrt(np.sum(dist, axis=2))
    
    stars1_idxs = np.arange(len(stars1))
    # stars2_idxs = np.arange(len(stars2))

    matches = np.stack([stars1_idxs, np.argmin(dist, axis=1)], axis=1)
    matches_dists = dist.min(axis=1)
    good_cond = matches_dists < min_dist

    if ret_msk:
        return matches, good_cond
    else:    
        good_matches = matches[good_cond]

    return good_matches


def visualize_correspondences(img1, img2, points1, points2, matches):
    """
    Visualize the matched points between two images.
    
    Parameters:
        img1: First image
        img2: Second image
        points1: List of points from the first image
        points2: List of points from the second image
        matches: List of index pairs (i, j) where points1[i] corresponds to points2[j]
    """
    # Create a new image that combines both input images

    # Randomly select 32 matches if there are more than 32 matches
    if len(matches) > 32:
        matches = matches[np.random.choice(len(matches), 32, replace=False)]
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    combined_image = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    combined_image[:h1, :w1] = img1
    combined_image[:h2, w1:w1 + w2] = img2
    combined_image[:, w1:w1 + 32] = [0, 0, 255]
    
    # Plot matches
    plt.figure(figsize=(32, 16))  # Increase figure size for higher resolution
    plt.imshow(combined_image)
    
    for i, j in matches:
        # Coordinates in the original images
        pt1 = (int(points1[i][0]), int(points1[i][1]))
        pt2 = (int(points2[j][0]) + w1, int(points2[j][1]))
        
        # Draw circles at point locations
        plt.scatter(*pt1, marker='o', color='red', s=40)
        plt.scatter(*pt2, marker='o', color='red', s=40)
        
        # Draw line between matched points
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'g-', linewidth=1)
    
    plt.title(f"Point Correspondences: {len(matches)} matches")
    plt.axis('off')
    plt.savefig("correspondences.png")
    plt.show()


def find_matches_from_imgs(img1, img2, min_dist=30, ret_msk:bool=False):
    """
    Find matching stars between two images based on minimum distance.
    
    Parameters:
        img1: First image
        img2: Second image
        min_dist: Minimum distance threshold for matching stars
        ret_msk: If True, keep matches that are further than min_dist
    
    Returns:
        matches: Array of shape (N, 2) where N is the number of matches,
                  each row contains indices (i, j) where i is from img1 and j is from img2
    """
    stars1 = find(img1)
    stars2 = find(img2)

    matches = find_matching_min(stars1, stars2, min_dist, ret_msk)
    
    good_kpt1 = stars1[matches[:, 0]]
    good_kpt2 = stars2[matches[:, 1]]

    return good_kpt1, good_kpt2, matches


def build_track(img_dir:str, save_f = "track.npy"):
    img_fs = sorted(glob.glob(f"{img_dir}/*image*.png"))

    good_kpt1, good_kpt2, _ = find_matches_from_imgs(img_fs[0], img_fs[1], min_dist=30, ret_msk=False)
    track = [good_kpt1, good_kpt2]

    kpt1 = good_kpt2
    for i in tqdm(range(2, len(img_fs))):
        kpts2 = find(img_fs[i])
        matches, msk = find_matching_min(kpt1, kpts2, min_dist=30, ret_msk=True)
        good_kpt = kpts2[matches[:, 1]]
        good_kpt[~msk] = np.nan
        track.append(good_kpt)
        kpt1 = good_kpt

        logger.info(f"num kpt:{msk.sum()}")
    
    track = np.stack(track, axis=0)

    if save_f is not None:
        np.save(save_f, track)

    return track
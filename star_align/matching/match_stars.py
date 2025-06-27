import numpy as np
from scipy.spatial import KDTree, distance
from collections import defaultdict
import matplotlib.pyplot as plt
from itertools import combinations

from loguru import logger

def create_triangles(stars):
    """
    Create triangles from star coordinates.
    
    Args:
        stars: List of (x, y) coordinates of stars
        
    Returns:
        triangles: List of triangles, each represented as (idx1, idx2, idx3, [side_lengths])
        star_to_triangles: Dictionary mapping star index to list of triangle indices it belongs to
    """
    # Convert to numpy array for faster computation
    stars = np.array(stars)
    n_stars = len(stars)
    
    # Find 100 nearest neighbors for each star
    kdtree = KDTree(stars)
    triangles = []
    star_to_triangles = defaultdict(list)
    
    # Use a set to track unique triangles (sorted vertex indices)
    unique_triangles = set()
    
    for i in range(n_stars):
        # Query for 101 closest points (including the point itself)
        distances, indices = kdtree.query(stars[i], k=min(101, n_stars))
        indices = indices[distances < 128]
        # Skip the first index as it's the star itself
        neighbors = indices[1:101] if len(indices) > 1 else indices
        
        # Create triangles with the star and its neighbors
        for j, k in combinations(neighbors, 2):
            # Sort the vertices to create a canonical representation
            vertices = tuple(sorted([i, j, k]))
            
            # Skip if this triangle has already been processed
            if vertices in unique_triangles:
                continue
                
            # Add to set of unique triangles
            unique_triangles.add(vertices)
            
            # Calculate side lengths
            side1 = distance.euclidean(stars[vertices[0]], stars[vertices[1]])
            side2 = distance.euclidean(stars[vertices[0]], stars[vertices[2]])
            side3 = distance.euclidean(stars[vertices[1]], stars[vertices[2]])
            
            # Sort side lengths to ensure invariance to vertex ordering
            sides = sorted([side1, side2, side3])
            
            # Store triangle as (vertex1, vertex2, vertex3, [side_lengths])
            triangle_idx = len(triangles)
            triangles.append((vertices[0], vertices[1], vertices[2], sides))
            
            # Map each star to triangles it belongs to
            star_to_triangles[vertices[0]].append(triangle_idx)
            star_to_triangles[vertices[1]].append(triangle_idx)
            star_to_triangles[vertices[2]].append(triangle_idx)
    
    logger.info(f"Created {len(triangles)} unique triangles from {n_stars} stars.")
    return triangles, star_to_triangles


def find_matching_stars(stars1, stars2, distance_threshold=1.0, min_triangle_matches=3):
    """
    Find matching stars between two images - optimized version.
    
    Args:
        stars1: List of (x, y) coordinates of stars in first image
        stars2: List of (x, y) coordinates of stars in second image
        distance_threshold: Maximum distance for triangle matching
        min_triangle_matches: Minimum number of matching triangles required
        
    Returns:
        matches: List of (idx1, idx2) pairs where idx1 is the index in stars1
                and idx2 is the index in stars2
    """
    # Create triangles for both star sets
    triangles1, star_to_triangles1 = create_triangles(stars1)
    triangles2, star_to_triangles2 = create_triangles(stars2)
    
    # Extract side lengths for KD-Tree construction
    sides1 = np.array([t[3] for t in triangles1])
    sides2 = np.array([t[3] for t in triangles2])
    
    # Build KD-Tree for the second set of triangles
    kdtree = KDTree(sides2)
    
    # Vectorized KD-tree query for all triangles in first image at once
    distances, match_indices = kdtree.query(sides1, k=1)
    
    # Store potential matches using numpy for faster processing
    # Initialize a counter matrix to track correspondence votes
    n_stars1 = len(stars1)
    n_stars2 = len(stars2)
    vote_matrix = np.zeros((n_stars1, n_stars2), dtype=int)
    
    # Only process good matches based on distance threshold
    good_matches = np.where(distances < distance_threshold)[0]
    
    # Process all good matches
    for i in good_matches:
        triangle1 = triangles1[i]
        triangle2 = triangles2[match_indices[i]]
        
        # Add votes for each possible vertex correspondence
        for v1 in triangle1[:3]:
            for v2 in triangle2[:3]:
                vote_matrix[v1, v2] += 1
    
    # Find the best match for each star in the first image
    final_matches = []
    for i in range(n_stars1):
        if np.max(vote_matrix[i]) >= min_triangle_matches:
            j = np.argmax(vote_matrix[i])
            final_matches.append((i, j))
    
    return final_matches



def visualize_matches(stars1, stars2, matches):
    """
    Visualize the matching stars between two images.
    
    Args:
        stars1: List of (x, y) coordinates of stars in first image
        stars2: List of (x, y) coordinates of stars in second image
        matches: List of (idx1, idx2) pairs where idx1 is the index in stars1
                and idx2 is the index in stars2
    """
    plt.figure(figsize=(12, 6))
    
    # Plot stars in first image
    plt.subplot(1, 2, 1)
    plt.scatter([p[0] for p in stars1], [p[1] for p in stars1], c='blue', label='Stars in Image 1')
    
    # Highlight matched stars in first image
    matched_stars1 = [stars1[m[0]] for m in matches]
    plt.scatter([p[0] for p in matched_stars1], [p[1] for p in matched_stars1], 
                c='red', label='Matched Stars')
    plt.title('Image 1')
    plt.legend()
    
    # Plot stars in second image
    plt.subplot(1, 2, 2)
    plt.scatter([p[0] for p in stars2], [p[1] for p in stars2], c='blue', label='Stars in Image 2')
    
    # Highlight matched stars in second image
    matched_stars2 = [stars2[m[1]] for m in matches]
    plt.scatter([p[0] for p in matched_stars2], [p[1] for p in matched_stars2], 
                c='red', label='Matched Stars')
    plt.title('Image 2')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def main():
    from star_finder.findStars import find
    from loguru import logger
    # Example star coordinates (these would come from your star detection algorithm)
    # Format: [(x1, y1), (x2, y2), ...]
    logger.info("Starting star matching process 1")
    stars1 = find("pre_proc/Stacking 10pm-1/corrected_image_DSC03803.JPG")
    
    # Second image with some rotation, translation, and maybe some noise
    # stars2 = [(15, 15), (25, 25), (35, 15), (45, 45), (55, 35), (65, 65)]
    logger.info("Starting star matching process 2")
    stars2 = find("pre_proc/Stacking 10pm-1/corrected_image_DSC03804.JPG")
    
    # Find matching stars
    logger.info("Finding matching stars")
    matches = find_matching_stars(stars1, stars2)
    logger.info(f"Found {len(matches)} matching stars")
    
    # Print matches
    for idx1, idx2 in matches:
        logger.info(f"Star {idx1} in image 1 matches star {idx2} in image 2")
    
    # Visualize matches
    visualize_matches(stars1, stars2, matches)

if __name__ == "__main__":
    main()
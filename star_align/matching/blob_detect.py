import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from loguru import logger
import os

def difference_of_gaussians(image, sigma_values):
    """
    Compute difference of Gaussians for multiple scales.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input grayscale image
    sigma_values : list
        List of sigma values for Gaussian filters
    
    Returns:
    --------
    dog_stack : numpy.ndarray
        Stack of DoG images
    """
    # Initialize stack for DoG images
    dog_stack = np.zeros((len(sigma_values) - 1, *image.shape), dtype=np.float32)
    
    # Compute DoG for each pair of consecutive sigma values
    for i in range(len(sigma_values) - 1):
        g1 = gaussian_filter(image, sigma_values[i])
        g2 = gaussian_filter(image, sigma_values[i + 1])
        dog_stack[i] = g2 - g1
    
    return dog_stack

def extract_stars(dog_stack, sigma_values, num_stars=50, min_distance=5):
    """
    Extract stars from DoG image stack.
    
    Parameters:
    -----------
    dog_stack : numpy.ndarray
        Stack of DoG images
    sigma_values : list
        List of sigma values used to create the DoG stack
    num_stars : int
        Number of stars to extract
    min_distance : int
        Minimum distance between stars
    
    Returns:
    --------
    stars : list
        List of tuples (y, x, radius) for each star
    """
    # Create a copy of the stack to modify
    stack = dog_stack.copy()
    
    # Create an empty list to store star information
    stars = []
    
    # Calculate radius factor (mapping from sigma to actual radius)
    # The radius of a blob detected with sigma is approximately 1.414 * sigma
    radius_factors = [1.414 * (sigma_values[i+1] + sigma_values[i])/2 for i in range(len(sigma_values)-1)]
    
    for _ in range(num_stars):
        # Find the brightest pixel in the stack
        max_value = np.max(stack)
        
        # If all pixels are zero (or very small), break
        if max_value < 1e-6:
            break
        
        # Find the indices of the maximum value
        indices = np.where(stack == max_value)
        scale_idx, y, x = indices[0][0], indices[1][0], indices[2][0]
        
        # Calculate the radius based on the scale
        radius = radius_factors[scale_idx]
        
        # Add the star to our list
        stars.append((y, x, radius))
        
        # Create a mask to remove overlapping blobs
        mask_radius = int(radius + min_distance)
        
        # Apply the mask to all scales in the stack
        for i in range(stack.shape[0]):
            # Create a circular mask
            mask = np.zeros_like(stack[i])
            cv2.circle(mask, (x, y), mask_radius, 1, -1)
            
            # Apply mask (set pixels in the circle to zero)
            stack[i][mask > 0] = 0
    
    return stars

def visualize_stars(image, stars):
    """
    Visualize detected stars on the original image.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Original image
    stars : list
        List of tuples (y, x, radius) for each star
    
    Returns:
    --------
    result : numpy.ndarray
        Image with circles drawn around detected stars
    """
    # Convert to color image if it's grayscale
    if len(image.shape) == 2:
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        result = image.copy()
    
    # Draw circles for each star
    for y, x, radius in stars:
        cv2.circle(result, (int(x), int(y)), int(radius), (0, 255, 0), 1)
    
    return result

# Example usage
def main():
    # Load an image
    image = cv2.imread('pre_proc/Stacking 11pm-2/corrected_image_DSC03687.JPG', cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Error loading image")
        return
    
    # Normalize image to [0, 1]
    image = image.astype(np.float32) / 255.0
    image[image < 2] = 0
    
    # Define sigma values for DoG
    sigma_values = [1, 2, 4, 8, 16]
    
    # Compute DoG
    logger.info("Computing Difference of Gaussians...")
    dog_stack = difference_of_gaussians(image, sigma_values)

    # Save DoG stack to temporary folder
    logger.info("Saving DoG stack to 'tmp_dog' folder...")
    tmp_dog_folder = '/home/yons/projects/stars/tmp_dog'
    os.makedirs(tmp_dog_folder, exist_ok=True)
    
    for i, dog_image in enumerate(dog_stack):
        filename = os.path.join(tmp_dog_folder, f'dog_scale_{i}.png')
        # cv2.imwrite(filename, (dog_image - dog_image.min()) / (dog_image.max() - dog_image.min()) * 255)
        cv2.imwrite(filename, (dog_image/dog_image.max() * 255).astype(np.uint8))
    
    # # Extract stars
    # logger.info("Extracting stars...")
    # stars = extract_stars(dog_stack, sigma_values, num_stars=100)
    
    # # Visualize results
    # result = visualize_stars(image, stars)
    
    # # Display or save the result
    # cv2.imwrite('detected_stars.jpg', result * 255)
    # cv2.imshow('Detected Stars', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

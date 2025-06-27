import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def viz_track(img_fs, track, output_dir = 'track_visualization_frames'):
    """
    Visualizes tracks of points across multiple frames on the corresponding images.
    
    Parameters:
    track (np.ndarray): shape = (n_frames, n_pnts, 2); if track 'i' stopped, the points will be [np.nan, np.nan]
    img_fs (list[str]): list of image file paths of length 'n_frames'
    
    Returns:
    None: Displays the visualization and saves frames in an output directory
    """
    n_frames, n_pnts, _ = track.shape
    
    # Verify that the number of frames matches the number of images
    assert len(img_fs) == n_frames, "Number of frames in track doesn't match number of images"
    
    # Create a colormap for the tracks
    colors = plt.cm.jet(np.linspace(0, 1, n_pnts))
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # For each frame
    for frame_idx in range(n_frames):
        # Read the image
        img = cv2.imread(img_fs[frame_idx])
        if img is None:
            print(f"Warning: Could not read image {img_fs[frame_idx]}")
            continue
            
        # Plot tracks up to current frame
        for point_idx in range(n_pnts):
            # Check if the track is lost in the current frame
            current_pt = track[frame_idx, point_idx]
            if np.isnan(current_pt).any():
                # Skip this track for this frame if it's lost
                continue
                
            color = colors[point_idx][:3]
            # Convert to BGR for OpenCV
            color_bgr = (int(color[2]*255), int(color[1]*255), int(color[0]*255))
            
            # Draw the entire track history for this point up to the current frame
            # First collect all valid points in the track history
            valid_track_points = []
            for t in range(frame_idx + 1):  # Include all frames up to current
                pt = track[t, point_idx]
                if not np.isnan(pt).any():
                    valid_track_points.append((t, tuple(map(int, pt))))
            
            # Draw lines connecting consecutive valid points
            for i in range(1, len(valid_track_points)):
                _, pt1 = valid_track_points[i-1]
                _, pt2 = valid_track_points[i]
                cv2.line(img, pt1, pt2, color_bgr, 2)
            
            # Draw all points in the track history
            for _, pt in valid_track_points[:-1]:  # All points except the current one
                cv2.circle(img, pt, 3, color_bgr, -1)  # Smaller circles for history
            
            # Draw the current point with a larger circle
            _, current_pt = valid_track_points[-1]
            cv2.circle(img, current_pt, 5, color_bgr, -1)  # Larger circle for current point
        
        # Add frame counter
        cv2.putText(img, f"Frame: {frame_idx+1}/{n_frames}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save the frame
        output_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(output_path, img)
        
        # Display progress
        if frame_idx % 10 == 0:
            print(f"Processing frame {frame_idx+1}/{n_frames}")
    
    print(f"All frames saved to '{output_dir}' directory")
    print("Visualization complete!")


        


if __name__ == "__main__":
    import os.path as osp
    import glob 

    img_fs = sorted(glob.glob("/home/yons/projects/stack_stars/pre_proc/Stacking 11pm-2/*image*.JPG"))
    track = np.load("/home/yons/projects/stack_stars/track.npy")
    viz_track(img_fs, track)
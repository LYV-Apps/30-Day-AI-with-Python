import cv2
import matplotlib.pyplot as plt 
import numpy as np 
import os
from thirty_days_pyai_helpers.print import slow_print, slow_print_header, slow_print_error, print_intro
from sklearn.cluster import KMeans 

# Function to detect dominant colors in an image
def detect_dominant_colors(image_path, num_colors=5):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("⚠️ Image not found or unable to load.")

    # Convert the image from BGR to RGB (OpenCV uses BGR format by default)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to a 2D array of pixels (height * width, 3)
    pixels = rgb_image.reshape(-1, 3)

    # Apply K-Means clustering to find dominant colors
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(pixels)

    # Get the RGB values of the cluster centers (dominant colors)
    dominant_colors = kmeans.cluster_centers_.astype(int)

    # Count the number of pixels in each cluster
    labels = kmeans.labels_
    color_counts = np.bincount(labels)

    # Normalize counte to create a color bar
    total_pixels = sum(color_counts)
    color_proportions = color_counts / total_pixels

    return dominant_colors, color_proportions

# Function to display the dominant colors
def display_dominant_colors(dominant_colors, color_proportions, image_path):
    # Print RGB values of the dominant colors: 
    slow_print_header("Dominant Colors (RGB):")
    for i, color in enumerate(dominant_colors):
        slow_print(f"Color {i+1}: {color}")

    # Create a single figure with 2 sublplots (stacked vertically)
    fig = plt.figure(figsize=(8,6))

    # Subplot 1: Original Image
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax1 = plt.subplot(2,1,1)
    ax1.imshow(rgb_image)
    ax1.axis('off')
    ax1.set_title("Original Image")
    

    # Subplot 2: Dominant Colors
    ax2 = plt.subplot(2,1,2)
    x_pos = 0

    # Plot a rectangle for each dominant color
    for color, proportion in zip(dominant_colors, color_proportions):
        width = proportion # Width proportional to the color's frequency
        ax2.add_patch(plt.Rectangle((x_pos, 0), width, 1, color=color/255.0))
        x_pos += width

    # Remove axes for a clean look
    ax2.set_xlim(0,1)
    ax2.set_ylim(0,1)
    ax2.axis('off')
    ax2.set_title("Dominant Colors")
    
    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the combined figure
    plt.show()

def list_image_paths():
    # Get the directory of the current Python script
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate to the parent directory
    parent_dir = os.path.dirname(current_script_dir)
    
    # Path to the images folder in the parent directory
    images_dir = os.path.join(current_script_dir, 'images')
    
    # Check if the images directory exists
    if not os.path.exists(images_dir):
        slow_print(f"Images directory not found: {images_dir}")
        return []
    
    # List all files in the images directory
    image_files = os.listdir(images_dir)
    
    # Get full paths for image files
    image_paths = [os.path.join(images_dir, img) for img in image_files]
    
    # Print out the paths
    slow_print_header("Image Paths:")
    for index, path in enumerate(image_paths):
        slow_print(f"{index + 1}: {os.path.basename(path)}")
    
    return image_paths

# Main execution
if __name__ == "__main__":
    print_intro(2, "Image Color Detector", "Colorful Images, Colorful Life!")
    images = list_image_paths()
    image_path = "images/grok_field_of_flowers.jpg"

    selected_image = int(input(f"Select an image (1-{len(images)}): "))

    try: 
        # Detect dominant colors (default: 5 colors)
        dominant_colors, color_proportions = detect_dominant_colors(images[selected_image-1], num_colors=5)

        # Display the results
        display_dominant_colors(dominant_colors, color_proportions, images[selected_image-1])
        
    except Exception as e:
        slow_print_error(f"An error occurred: {e}")


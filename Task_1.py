import numpy as np
import cv2
import matplotlib.pyplot as plt

# Function to generate an image with 2 objects and a total of 3 pixel values
def generate_image():
    image = np.zeros((250, 250), dtype=np.uint8)
    
    # Drawing a filled circle with pixel value of 200
    cv2.circle(image, (100, 100), 50, 200, -1)
    
    # Drawing a filled rectangle with pixel value of 150
    cv2.rectangle(image, (150, 150), (200, 200), 150, -1)
    
    # Setting the background to pixel value
    image[image == 0] = 50
    
    return image

# Function to add Gaussian noise to the image
def add_noise(image):
    noise_intensity = np.sqrt(500)
    noise_gaussian = np.random.normal(0, noise_intensity, image.shape)
    noisy_image = noise_gaussian + image
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

# Function to implement Otsu's algorithm
def otsu_thresholding(image):
    hist, bins = np.histogram(image.ravel(), 256, [0, 256])
    total_pixels = image.shape[0] * image.shape[1]
    sum_pixel_values = np.sum(image)
    sum_background = 0
    weight_background = 0
    max_variance = 0.0
    optimal_threshold = 0
    for i in range(256):
        sum_background += i * hist[i]
        weight_background += hist[i]
        if weight_background == 0:
            continue
        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break
        mean_background = sum_background / weight_background
        mean_foreground = (sum_pixel_values - sum_background) / weight_foreground
        between_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if between_variance >= max_variance:
            optimal_threshold = i
            max_variance = between_variance
    segmented_image = cv2.threshold(image, optimal_threshold, 255, cv2.THRESH_BINARY)[1]
    return segmented_image, optimal_threshold

# Generate image
image = generate_image()

# Add Gaussian noise
noisy_image = add_noise(image)

# Implement Otsu's algorithm
segmented_image, threshold = otsu_thresholding(noisy_image)

# Display results
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(image, cmap='gray', vmin=0, vmax=255)
axs[0].set_title("Original Image with\nCircle and Rectangle")
axs[1].imshow(noisy_image, cmap='gray', vmin=0, vmax=255)
axs[1].set_title("Image with Gaussian Noise")
axs[2].imshow(segmented_image, cmap='gray', vmin=0, vmax=255)
axs[2].set_title(f"Segmented Image Using Otsu's Algorithm\n(Threshold = {threshold})")
plt.show()

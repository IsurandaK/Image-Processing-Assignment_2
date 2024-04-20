import numpy as np
import cv2
import matplotlib.pyplot as plt

# Function to generate an image with a circle and a rectangle
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

# Function to implement region growing for image segmentation
def region_growing(img, seeds, pixel_range):
    mask = np.zeros(img.shape, dtype=np.uint8)
    for seed in seeds:
        queue = [seed]
        while queue:
            current_pixel = queue.pop(0)
            if mask[current_pixel] == 0 and np.abs(int(img[current_pixel[0], current_pixel[1]]) - int(img[seed[0], seed[1]])) <= pixel_range:
                mask[current_pixel] = 255
                neighbors = [(current_pixel[0] + dx, current_pixel[1] + dy) for dx in range(-1, 2) for dy in range(-1, 2) if not (dx == 0 and dy == 0)]
                for neighbor in neighbors:
                    if 0 <= neighbor[0] < img.shape[0] and 0 <= neighbor[1] < img.shape[1]:
                        queue.append(neighbor)
    return mask

# Generate image
image = generate_image()

# Add Gaussian noise
noisy_image = add_noise(image)

# Define seeds and pixel range
seeds = [(100, 100), (175, 175)]  # Center of circle and center of rectangle
pixel_range = 50

# Apply region growing for segmentation
segmented_image = region_growing(noisy_image, seeds, pixel_range)

# Display results
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(image, cmap='gray', vmin=0, vmax=255)
axs[0].set_title("Original Image")
axs[1].imshow(noisy_image, cmap='gray', vmin=0, vmax=255)
axs[1].set_title("Noisy Image")
axs[2].imshow(segmented_image, cmap='gray', vmin=0, vmax=255)
axs[2].set_title("Segmented Image using Region Growing")
plt.show()

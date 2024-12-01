import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import pywt
import imageio
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
import os

# Ensure output directories exist
os.makedirs('iterations', exist_ok=True)
os.makedirs('images', exist_ok=True)

# Step 1: Load and Prepare the MRI Image
# Using Shepp-Logan Phantom as a simulated MRI image
image_size = 256
original_image = shepp_logan_phantom()
original_image = resize(original_image, (image_size, image_size), mode='reflect', anti_aliasing=True)

plt.imshow(original_image, cmap='gray')
plt.title('Original MRI Image')
plt.axis('off')
plt.savefig('images/original_image.png', bbox_inches='tight')
plt.close()

# Step 2: Perform Full k-space Acquisition (2D FFT)
full_k_space = fftshift(fft2(original_image))

# Step 3: Simulate Undersampling in k-space
# Create a variable-density random undersampling mask
def undersampling_mask(size, undersampling_factor):
    mask = np.zeros(size)
    # Fully sample the center of k-space
    center_fraction = 0.08
    center_size = int(size[0] * center_fraction)
    center_start = size[0] // 2 - center_size // 2
    center_end = center_start + center_size
    mask[center_start:center_end, :] = 1

    # Randomly sample the rest of k-space
    prob = undersampling_factor
    random_mask = np.random.rand(size[0], size[1]) < prob
    mask = np.logical_or(mask, random_mask)
    return mask.astype(float)

undersampling_factor = 0.3  # Adjust to undersample 30% of the peripheral k-space
mask = undersampling_mask(full_k_space.shape, undersampling_factor)

# Apply the undersampling mask
undersampled_k_space = full_k_space * mask

# Step 4: Conventional Reconstruction (Zero-filled IFFT)
zero_filled_reconstruction = np.abs(ifft2(ifftshift(undersampled_k_space)))

plt.imshow(zero_filled_reconstruction, cmap='gray')
plt.title('Zero-filled Reconstruction')
plt.axis('off')
plt.savefig('images/zero_filled_reconstruction.png', bbox_inches='tight')
plt.close()

# Step 5: Compressed Sensing Reconstruction
# Define the Soft Thresholding function
def soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

# Parameters for Iterative Reconstruction
max_iterations = 50
lambda_reg = 0.01  # Regularization parameter
wavelet = 'db4'    # Daubechies 4 wavelet
levels = 4         # Levels of wavelet decomposition

# Initialize the reconstruction with zero-filled image
reconstruction = zero_filled_reconstruction.copy()

# Iterative Reconstruction Loop
for iteration in range(max_iterations):
    # Step 5.1: Enforce Data Consistency
    reconstruction_k_space = fftshift(fft2(reconstruction))
    reconstruction_k_space = undersampled_k_space + (1 - mask) * reconstruction_k_space
    reconstruction = np.abs(ifft2(ifftshift(reconstruction_k_space)))

    # Step 5.2: Sparsity Promotion via Wavelet Thresholding
    coeffs = pywt.wavedec2(reconstruction, wavelet=wavelet, level=levels)
    coeffs_thresh = list()
    for coeff_detail in coeffs:
        if isinstance(coeff_detail, tuple):
            coeffs_thresh.append(tuple(soft_threshold(c, lambda_reg) for c in coeff_detail))
        else:
            coeffs_thresh.append(soft_threshold(coeff_detail, lambda_reg))
    reconstruction = pywt.waverec2(coeffs_thresh, wavelet=wavelet)

    # Save the reconstruction at every iteration
    plt.imshow(reconstruction, cmap='gray')
    plt.title(f'Iteration {iteration + 1}')
    plt.axis('off')
    plt.savefig(f'iterations/reconstruction_{iteration + 1}.png', bbox_inches='tight')
    plt.close()

    # Optionally, monitor convergence by printing the iteration number
    print(f'Iteration {iteration + 1} completed.')

# Step 6: Visualize the Improvements
# Create a movie (GIF) from the saved images
images = []
for iteration in range(1, max_iterations + 1):
    filename = f'iterations/reconstruction_{iteration}.png'
    images.append(imageio.imread(filename))

# Save the movie
imageio.mimsave('images/reconstruction_movie.gif', images, fps=5)

# Step 7: Display the Final Reconstruction
plt.imshow(reconstruction, cmap='gray')
plt.title('Final CS Reconstruction')
plt.axis('off')
plt.savefig('images/final_cs_reconstruction.png', bbox_inches='tight')
plt.close()

# Step 8: Compare with Original Image
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(reconstruction, cmap='gray')
plt.title('Final CS Reconstruction')
plt.axis('off')

plt.savefig('images/comparison.png', bbox_inches='tight')
plt.close()
